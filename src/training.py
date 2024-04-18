import typing,os,re,logging,json,inspect
from timeit import default_timer as timer
from pathlib import Path
import pickle as pkl
import numpy as np

from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch.profiler

import utils
import errors
import visualization
from models.modeling_utils import BaseModel, BaseConfig
from mapping import registry
from tokenizers import BaseTokenizer
from simulated_annealing import anneal
from optimization import ConstantLRSchedule, WarmupConstantSchedule, WarmupLinearSchedule, WarmupCosineSchedule

try:
    from apex import amp
    import amp_C
    import apex_C
    from apex.amp import _amp_state
    from apex.parallel.distributed import flat_dist_call
    from apex.parallel.distributed import DistributedDataParallel as DDP
    APEX_FOUND = True
except ImportError:
    APEX_FOUND = False

logger = logging.getLogger(__name__)

MetricsDict = typing.Dict[str, float]
LossAndMetrics = typing.Tuple[float, MetricsDict]
OutputDict = typing.Dict[str, typing.Any]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)

class ForwardRunner:

    def __init__(self,
                 model: BaseModel,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 local_rank: int = -1):

        self.model = model
        self.device = device
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.local_rank = local_rank

        forward_arg_keys = inspect.getfullargspec(model.forward).args
        forward_arg_keys = forward_arg_keys[1:]  # remove self argument
        self._forward_arg_keys = forward_arg_keys
        #assert 'input_ids' in self._forward_arg_keys

    def initialize_distributed_model(self):
        if self.local_rank != -1:
            if not self.fp16:
                self.model = DDP(self.model)
            else:
                flat_dist_call([param.data for param in self.model.parameters()],
                               torch.distributed.broadcast, (0,))
        elif self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)

    def forward(self,
                batch: typing.Dict[str, torch.Tensor],
                return_outputs: bool = False,
                no_loss: bool = False):
        # Filter out batch items that aren't used in this model
        # Requires that dataset keys match the forward args of the model
        # Useful if some elements of the data are only used by certain models
        # e.g. PSSMs / MSAs and other evolutionary data
        batch = {name: tensor for name, tensor in batch.items()
                 if name in self._forward_arg_keys}
        if self.device.type == 'cuda':
            # for name, val in batch.items():
            #     if isinstance(val,torch.Tensor):
            #         batch[name] = val.cuda(device=self.device, non_blocking=True)
            #     elif isinstance(val, typing.List):
            #         batch[name] = [tensor.cuda(device=self.device, non_blocking=True) for tensor in val]

            batch = {name: tensor.cuda(device=self.device, non_blocking=True)
                     for name, tensor in batch.items()}
        
        outputs = self.model(**batch)

        if no_loss:
            return outputs

        if isinstance(outputs[0], tuple) and len(outputs[0]) == 2:
            # model also returned metrics
            loss, metrics = outputs[0]
        else:
            # no metrics and loss
            loss = None
            metrics = {}

        if self.n_gpu > 1 and loss is not None :  # pytorch DataDistributed doesn't mean scalars
            loss = loss.mean()
            metrics = {name: metric.mean() for name, metric in metrics.items()}

        if return_outputs:
            return loss, metrics, outputs
        else:
            return loss, metrics

    def train(self):
        self.model.train()
        return self

    def eval(self):
        self.model.eval()
        return self

class BackwardRunner(ForwardRunner):

    def __init__(self,
                 model: BaseModel,
                 optimizer: optim.Optimizer,  # type: ignore
                 gradient_accumulation_steps: int = 1,
                 device: torch.device = torch.device('cuda:0'),
                 n_gpu: int = 1,
                 fp16: bool = False,
                 fp16_opt_level: str = 'O1',
                 local_rank: int = -1,
                 max_grad_norm: float = 1.0,
                 warmup_steps: int = 0,
                 num_train_optimization_steps: int = 1000000,
                 lr_scheduler: str = 'constant',
                 num_epoch: int = 50):

        super().__init__(model, device, n_gpu, fp16, local_rank)
        self.fp16_opt_level = fp16_opt_level
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self._global_step = 0
        self._local_rank = local_rank
        self._overflow_buf = torch.cuda.IntTensor([0])  # type: ignore
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._delay_accumulation = fp16 and local_rank != -1
        if lr_scheduler == 'constant':
            self.scheduler = ConstantLRSchedule(self.optimizer)
        elif lr_scheduler == 'warmupConstant':
            self.scheduler = WarmupConstantSchedule(
                self.optimizer, warmup_steps)
        elif lr_scheduler == 'warmupLinear':
            self.scheduler = WarmupLinearSchedule(
                self.optimizer, warmup_steps, num_train_optimization_steps)
        elif lr_scheduler == 'warmupCosine':
            self.scheduler = WarmupCosineSchedule(
                self.optimizer, warmup_steps, num_train_optimization_steps, cycles=.5)
        elif lr_scheduler == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, num_train_optimization_steps // (num_epoch*4), T_mult=2, eta_min=1e-8, verbose=False)

    def initialize_fp16(self):
        if self.fp16:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level=self.fp16_opt_level, loss_scale="dynamic") # master_weights=True
            _amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    def resume_from_checkpoint(self, checkpoint_dir: str, checkpoint_epoch: int) -> int:
        if checkpoint_epoch is None:
            logger.info("loading checkpoint from {}checkpoint.bin".format(checkpoint_dir))
            checkpoint = torch.load(
                os.path.join(checkpoint_dir, 'checkpoint.bin'), map_location=self.device)
        else:
            logger.info("loading checkpoint from {}checkpoint_{}.bin".format(checkpoint_dir,checkpoint_epoch))
            checkpoint = torch.load(
                os.path.join(checkpoint_dir, 'checkpoint_{}.bin'.format(checkpoint_epoch)), map_location=self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.fp16:
            self.optimizer._lazy_init_maybe_master_weights()
            self.optimizer._amp_stash.lazy_init_called = True
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved in zip(
                    amp.master_params(self.optimizer), checkpoint['master params']):
                param.data.copy_(saved.data)
            amp.load_state_dict(checkpoint['amp'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint.keys():
            start_gloStep = checkpoint['global_step'] + 1
        else:
            start_gloStep = 0
        return start_epoch, start_gloStep

    def save_state(self, 
                   save_directory: typing.Union[str, Path],
                   epoch_id: int,
                   save_freq: typing.Union[str, int],
                   save_freq_opt_checkpoint: typing.Union[str, int],
                   save_checkpoint: bool,
                   num_train_epochs: int,
                   num_evals_no_improvement: int):
        save_directory = Path(save_directory)
        if not save_directory.exists():
            save_directory.mkdir()
        else:
            assert save_directory.is_dir(), "Save path should be a directory"
        model_to_save = getattr(self.model, 'module', self.model)
        model_to_save.save_pretrained(save_directory, epoch_id, save_freq, num_train_epochs, num_evals_no_improvement)
        optimizer_state: typing.Dict[str, typing.Any] = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch_id,
            'global_step': self.global_step}
        if APEX_FOUND:
            optimizer_state['master params'] = list(amp.master_params(self.optimizer))
            try:
                optimizer_state['amp'] = amp.state_dict()
            except AttributeError:
                pass
        if save_checkpoint:
            if isinstance(save_freq_opt_checkpoint, int):
                if (((epoch_id + 1) % save_freq_opt_checkpoint == 0) or ((epoch_id + 1) == num_train_epochs)) and num_evals_no_improvement == 0:
                    torch.save(optimizer_state, save_directory / 'checkpoint_{}.bin'.format(epoch_id))
                    torch.save(optimizer_state, save_directory / 'checkpoint.bin')
                elif ((epoch_id + 1) % save_freq_opt_checkpoint == 0) or ((epoch_id + 1) == num_train_epochs) or (epoch_id == 0):
                    torch.save(optimizer_state, save_directory / 'checkpoint_{}.bin'.format(epoch_id))
                elif num_evals_no_improvement == 0:
                    torch.save(optimizer_state, save_directory / 'checkpoint.bin')
            else:
                torch.save(optimizer_state, save_directory / 'checkpoint.bin')


    def backward(self, loss) -> None:
        if not self._delay_accumulation:
            loss = loss / self.gradient_accumulation_steps
        if self.fp16:
            with amp.scale_loss(loss, self.optimizer,
                                delay_overflow_check=self._delay_accumulation) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def step(self) -> None:
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, norm_type=2.0, error_if_nonfinite=False)
        if self._local_rank == -1:
            self._step()
        elif not self.fp16:
            # TODO: Can you do this allreduce after accumulation also?
            self._step()
        else:
            self._step_distributed_fp16()

    def _step(self) -> None:
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()  # type: ignore
        self._global_step += 1

    def _step_distributed_fp16(self) -> None:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        scaler = _amp_state.loss_scalers[0]
        master_grads = [p.grad for p in amp.master_params(self.optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        # allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else \
            # torch.float32
        allreduce_dtype = torch.float16
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [master_grads, allreduced_views],
            scaler.loss_scale() / (
                torch.distributed.get_world_size() * self.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        self._overflow_buf.zero_()
        amp_C.multi_tensor_scale(
            65536,
            self._overflow_buf,
            [allreduced_views, master_grads],
            1. / scaler.loss_scale())
        # 5. update loss scale
        scaler = _amp_state.loss_scalers[0]
        old_overflow_buf = scaler._overflow_buf
        scaler._overflow_buf = self._overflow_buf
        had_overflow = scaler.update_scale()
        scaler._overfloat_buf = old_overflow_buf
        # 6. call optimizer step function
        if had_overflow == 0:
            self._step()
        else:
            # Overflow detected, print message and clear gradients
            logger.info(f"Gradient overflow.  Skipping step, reducing loss scale to "
                        f"{scaler.loss_scale()}")
            if _amp_state.opt_properties.master_weights:
                for param in self.optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in self.model.parameters():
            param.grad = None

    @property
    def global_step(self) -> int:
        return self._global_step
    
    @property
    def set_global_step(self, gloStep) -> None:
        self._global_step = gloStep

def run_train_epoch(epoch_id: int,
                    train_loader: DataLoader,
                    runner: BackwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    num_log_iter: int = 20,
                    gradient_accumulation_steps: int = 1,
                    log_dir: str = None,
                    pytorch_profiler: bool = False,
                    start_epoch: int = None) -> LossAndMetrics:
    if viz is None:
        viz = visualization.DummyVisualizer()
    smoothing = 1 - 1 / num_log_iter
    accumulator = utils.MetricsAccumulator(smoothing)

    torch.set_grad_enabled(True)
    runner.train()

    def make_log_str(step: int, time: float, forward_time: float=0, backward_time: float=0, data_time: float=0) -> str:
        ep_percent = epoch_id + step / len(train_loader)
        if runner.scheduler is not None:
            #curr_lr = runner.scheduler.get_lr()[0]  # type: ignore
            curr_lr = runner.scheduler.get_last_lr()[0] # type: ignore
        else:
            curr_lr = runner.optimizer.param_groups[0]['lr']

        print_str = []
        print_str.append(f"[Ep: {ep_percent:.2f}]")
        print_str.append(f"[Iter: {runner.global_step}]")
        print_str.append(f"[Time: {time:5.2f}s; F/B/D: {forward_time:.1f}/{backward_time:.1f}/{data_time:.1f}]")
        print_str.append(f"[Loss: {accumulator.loss():.5g}]")

        for name, value in accumulator.metrics().items():
            print_str.append(f"[{name.capitalize()}: {value:.5g}]")

        print_str.append(f"[LR: {curr_lr:.5g}]")
    
        ## GPU mem inspect
        curr_device_idx = torch.cuda.current_device()
        mem_divider = 1.049e+6 # byte to MiB
        ma_mib = torch.cuda.memory_allocated(curr_device_idx) // mem_divider
        max_ma_mib = torch.cuda.max_memory_allocated(curr_device_idx) // mem_divider
        mr_mib = torch.cuda.memory_reserved(curr_device_idx) // mem_divider
        max_mr_mib = torch.cuda.max_memory_reserved(curr_device_idx) // mem_divider
        free_mem, total_mem = torch.cuda.mem_get_info(curr_device_idx)
        active_mem, free_mem, total_mem = (total_mem - free_mem) // mem_divider, free_mem // mem_divider, total_mem // mem_divider
        print_str.append(f"[Mem({curr_device_idx}): {int(ma_mib)}(ma),{int(max_ma_mib)}(mma),{int(mr_mib)}(mr),{int(max_mr_mib)}(mmr),{int(active_mem)}/{int(free_mem)}/{int(total_mem)}(a/f/t)]")
    
        return ''.join(print_str)
    
    if pytorch_profiler and epoch_id == start_epoch:
        # setup pytorch profiler
        pfer = torch.profiler.profile(
            #schedule=torch.profiler.schedule(skip_first=0, wait=max(1,int(len(train_loader)*0.1)), warmup=max(1,int(len(train_loader)*0.1)), active=min(3,int(len(train_loader)*0.2)), repeat=2),
            schedule=torch.profiler.schedule(skip_first=30, wait=10, warmup=10, active=3, repeat=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=False,
            profile_memory=True,
            with_stack=True)
        # set timer
        pfer.start()
        start_t = timer()
        for step, batch in enumerate(train_loader):
            loss, metrics = runner.forward(batch)  # type: ignore
            runner.backward(loss)
            accumulator.update(loss, metrics, step=False)
            if (step + 1) % gradient_accumulation_steps == 0:
                runner.step()
                viz.log_metrics(accumulator.step(), "train", runner.global_step)
                viz.log_scalars("lr","train", runner.scheduler.get_last_lr()[0], runner.global_step)
                if runner.global_step % num_log_iter == 0:
                    end_t = timer()
                    logger.info(make_log_str(step, end_t - start_t))
                    start_t = end_t
            pfer.step()
        pfer.stop()
    else:
        forward_t,backward_t,data_t = 0,0,0
        start_t = timer()
        data_start = timer()
        for step, batch in enumerate(train_loader):
            data_t += timer() - data_start
            forward_start = timer()
            loss, metrics = runner.forward(batch)  # type: ignore
            forward_t += timer() - forward_start
            
            backward_start = timer()
            runner.backward(loss)
            accumulator.update(loss, metrics, step=False)
            backward_t += timer() - backward_start
            
            if (step + 1) % gradient_accumulation_steps == 0:
                backward_start = timer()
                runner.step()
                backward_t += timer() - backward_start
                
                viz.log_metrics(accumulator.step(), "train", runner.global_step)
                viz.log_scalars("lr","train", runner.scheduler.get_last_lr()[0], runner.global_step)
                if runner.global_step % num_log_iter == 0:
                    end_t = timer()
                    logger.info(make_log_str(step, end_t - start_t, forward_t, backward_t, data_t))
                    forward_t,backward_t,data_t = 0,0,0
                    start_t = end_t
            
            data_start = timer()

    final_print_str = f"Train: [Loss: {accumulator.final_loss():.5g}]"
    for name, value in accumulator.final_metrics().items():
        final_print_str += f"[{name.capitalize()}: {value:.5g}]"
    logger.info(final_print_str)
    return accumulator.final_loss(), accumulator.final_metrics()

def run_valid_epoch(epoch_id: int,
                    valid_loader: DataLoader,
                    runner: ForwardRunner,
                    viz: typing.Optional[visualization.TAPEVisualizer] = None,
                    is_master: bool = True) -> typing.Tuple[float, typing.Dict[str, float]]:
    num_batches = len(valid_loader)
    accumulator = utils.MetricsAccumulator()

    torch.set_grad_enabled(False)
    runner.eval()
    
    # !!This need to be corrected in utils.MetricsAccumulator():
    # The total perplexity is calculated in the way like: 1/2*(s1/n1 + s2/n2)
    # But should be (s1+s2)/(n1+n2)
    for batch in tqdm(valid_loader, desc='Running Eval', total=num_batches,
                      disable=not is_master, leave=False):
        loss, metrics = runner.forward(batch)  # type: ignore
        accumulator.update(loss, metrics)

    # Reduce loss across all processes if multiprocessing
    eval_loss = utils.reduce_scalar(accumulator.final_loss())
    metrics = {name: utils.reduce_scalar(value)
               for name, value in accumulator.final_metrics().items()}

    print_str = f"Evaluation: [Loss: {eval_loss:.5g}]"
    for name, value in metrics.items():
        print_str += f"[{name.capitalize()}: {value:.5g}]"

    metrics['loss'] = eval_loss
    if viz is not None:
        #viz.log_metrics(metrics, "val", getattr(runner, 'global_step', epoch_id))
        viz.log_metrics(metrics, "val", epoch_id)
        logger.info("** Visualization log saved after epoch {} **".format(epoch_id))


    logger.info(print_str)

    return eval_loss, metrics

def run_eval_epoch(eval_loader: DataLoader,
                   runner: ForwardRunner,
                   metric_names: typing.Sequence[str],
                   metric_functions: typing.Sequence[typing.Callable],
                   data_dir: str,
                   task: str,
                   from_pretrained: str,
                   pretrained_epoch: typing.Union[int,str],
                   model_config: BaseConfig,
                   split: str,
                   eval_save_dir: str,
                   output_pred: bool = False,
                   is_master: bool = True,
                   **kwargs) -> typing.Union[typing.Dict,typing.Tuple]:  #typing.List[typing.Dict[str, typing.Any]]:
    torch.set_grad_enabled(False)
    runner.eval()
    
    ## antibody task related
    mlm_mask_stragy = kwargs.get('mlm_mask_stragy', None)
    mlm_maskStragy_id = f'_{mlm_mask_stragy}' if mlm_mask_stragy is not None else ''
    #embed_modelNm = kwargs.get('embed_modelNm', None)
    
    ## mutagenesis
    mutgsis_set = kwargs.get('mutgsis_set', None)

    # load some config params
    num_layers = model_config.num_hidden_layers
    num_heads = model_config.num_attention_heads
    head_selector = model_config.head_selector

    save_outputs = []
    metric_values = {} # Dict[str,Any]
    accumulator = utils.MetricsAccumulator() # (moving) average across batches

    ## initialize metric_values
    for name in metric_names:
        if name == 'perplexity_subClass_AB':
            metric_values[name] = [0.,0.,0.] # [ece, ppl, makedToken_count]
        elif name == 'antibody_HL_likelihood':
            metric_values[name] = {
                'seq_ids': [], # [bs,]
                'mut_names': [], # [bs,]
                'aa_logits': [], # [bs,l_max,n_aa]
                'wt_aa_ids': [], # [bs,l_max]
                'mut_aa_ids': [], # [bs,l_max]
            }
        elif name in ['embed_antibody','embed_antibody_internal']:
            metric_values[name] = [] # save data for tSNE
        else:
            metric_values[name] = [0., 0.]

    for batch in tqdm(eval_loader, desc='Evaluation', total=len(eval_loader),
                      disable=not is_master):
        loss, metrics, outputs = runner.forward(batch, return_outputs=True) # type: ignore
        
        if task == 'antibody_mlm_seqConcate':
            pred_token_logits = outputs[1].cpu().numpy() #[bs,l_max,n_token]
            pred_subClassHLPair_logits = outputs[2].cpu().numpy() #[bs,n_subClassPair]
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            targets = batch['targets'].cpu().numpy() #[bs,l_max]
            predictions = pred_token_logits
        elif task == 'antibody_embed_seqConcate':
            hidden_states_transform_token = outputs[0].cpu().numpy() #[bs,l_max,hidden_d]
            hidden_states_transform_subClassHLPair = outputs[1].cpu().numpy() #[bs,hidden_d]
            hidden_states_encoder_lastLayer = outputs[2][-1].cpu().numpy()  #[bs,l_max,hidden_d]
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            entityH = batch['entityH'] #tuple, [bs,]
            entityL = batch['entityL'] #tuple, [bs,]
            input_masks = batch['input_mask'].cpu().numpy() #[bs,l_max]
            token_type_ids = batch['token_type_ids'].cpu().numpy() #[bs,l_max]
        elif task == 'antibody_mlm_seqIndiv':
            pred_token_logits_VH = outputs[1].cpu().numpy() #[bs,l_max_VH,n_token]
            pred_token_logits_VL = outputs[2].cpu().numpy() #[bs,l_max_VL,n_token]
            pred_subClassHLPair_logits = outputs[3].cpu().numpy() #[bs,n_subClassPair]
            targets_VH = batch['targets_VH'].cpu().numpy() #[bs,l_max_VH]
            targets_VL = batch['targets_VL'].cpu().numpy() #[bs,l_max_VL]
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            targets = np.concatenate((targets_VH,targets_VL),axis=1) #[bs,l_max_VH+l_max_VL]
            predictions = np.concatenate((pred_token_logits_VH,pred_token_logits_VL),axis=1) #[bs,l_max_VH+l_max_VL,n_token]
        elif task == 'antibody_embed_seqIndiv':
            hidden_states_transform_token_VH = outputs[0].cpu().numpy() #[bs,l_VH,hidden_d], hidden vec in pred_head
            hidden_states_transform_token_VL = outputs[1].cpu().numpy() #[bs,l_VL,hidden_d], hidden vec in pred_head
            hidden_states_transform_subClassHLPair = outputs[2].cpu().numpy() #[bs,hidden_d]
            hidden_states_encoder_lastLayer_VH = outputs[7].cpu().numpy() #[bs,l_VH,hidden_d], after crossAttention
            hidden_states_encoder_lastLayer_VL = outputs[9].cpu().numpy() #[bs,l_VL,hidden_d], after crossAttention
            targets_subClassHLPair = batch['subClassHLPair'].cpu().numpy() #[bs,]
            entityH = batch['entityH'] #tuple, [bs,]
            entityL = batch['entityL'] #tuple, [bs,]
            input_masks_VH = batch['input_mask_VH'].cpu().numpy() #[bs,l_max]
            input_masks_VL = batch['input_mask_VL'].cpu().numpy() #[bs,l_max]
        elif task  == 'antibody_mutation_MLM':
            concate_stragy = kwargs.get('concate_stragy',None)
            assert concate_stragy is not None
            if concate_stragy == 'seqConcate':
                pred_aa_logits = outputs[1].cpu().numpy() #[bs,l_max,n_token]
                labels_wt = batch['targets'].cpu().numpy() #[bs,l_max]
                labels_mut = batch['targets_mut'].numpy()
                mut_names = batch['mut_names']
                seq_ids = batch['seq_ids']
            elif concate_stragy == 'seqIndiv':
                pred_aa_logits_VH = outputs[1].cpu().numpy() #[bs,l_max_VH,n_token]
                pred_aa_logits_VL = outputs[2].cpu().numpy() #[bs,l_max_VL,n_token]
                pred_aa_logits = np.concatenate((pred_aa_logits_VH,pred_aa_logits_VL),axis=1)
                labels_wt_VH = batch['targets_VH'].numpy() #[bs,l_max_VH]
                labels_wt_VL = batch['targets_VL'].numpy() #[bs,l_max_VL]
                labels_mut_VH = batch['targets_mut_VH'].numpy()
                labels_mut_VL = batch['targets_mut_VL'].numpy()
                labels_wt = np.concatenate((labels_wt_VH,labels_wt_VL),axis=1)
                labels_mut = np.concatenate((labels_mut_VH,labels_mut_VL),axis=1)
                mut_names = batch['mut_names']
                seq_ids = batch['seq_ids']
            else:
                Exception(f'invalid concate_stragy: {concate_stragy}')
        else:
            metric_tuple = outputs[0][1]
            predictions = outputs[1].cpu().numpy() # mlm: [bs,l_max,n_token]; fitness:[bs,] 
            targets = batch['targets'].cpu().numpy() # mlm: [bs,l_max]; fitness:[bs,]
            hiddenMats, attentionMats = None, None
            #hiddenMats = np.transpose([hiddenL.cpu().numpy() for hiddenL in outputs[2]], (1,0,2,3)) # [bs,n_layer,L_max,hidden_d]
            #attentionMats = np.transpose([attenL.cpu().numpy() for attenL in outputs[3]], (1,0,2,3,4)) # [bs, n_layer, n_head,L_max,L_max]
        
            # targets_contact, valid_mask, seq_length are co existing
            if "targets_contact" in batch.keys():
                target_contacts = batch['targets_contact'].cpu().numpy() #size: [bs,l_max,l_max]
                valid_masks = batch ['valid_mask'].cpu().numpy() # size: [bs, l_max]
                seq_lengths = batch['seq_length'].cpu().numpy() # size: [bs,]
                loss_mlm = np.log(metric_tuple['perplexity'].cpu().numpy()) # size: [n_gpu,] batch mean on each gpu
                if 'nonCon_att_Fnorm2' in metric_tuple.keys(): 
                  loss_fnorm2 = metric_tuple['nonCon_att_Fnorm2'].cpu().numpy() # size: [n_gpu,]
                  loss_fnorm2_local = metric_tuple['nonCon_att_Fnorm2_local'].cpu().numpy() # size: [n_gpus, ]
                  loss_fnorm2_local_nor = metric_tuple['nonCon_att_Fnorm2_local_nor'].cpu().numpy() # size: [n_gpu, ]
                elif 'con_att_Fnorm2' in metric_tuple.keys(): 
                  loss_fnorm2 = metric_tuple['con_att_Fnorm2'].cpu().numpy() # size: [n_gpu,]
                  loss_fnorm2_local = metric_tuple['con_att_Fnorm2_local'].cpu().numpy() # size: [n_gpu, ]
                  loss_fnorm2_local_nor = metric_tuple['con_att_Fnorm2_local_nor'].cpu().numpy() # size: [n_gpu, ]
                else:
                  pass
            else:
                if 'perplexity' in metric_tuple.keys():
                    loss_mlm = np.log(metric_tuple['perplexity'].cpu().numpy())
        
        ## save needed outputs 
        if output_pred:
            if "targets_contact" in batch.keys():
                for pred, target, attentMat, tar_cont, seq_len in zip(predictions, targets, attentionMats, target_contacts, seq_lengths):
                    save_outputs.append({"prediction": pred, "target": target, "attenMat": attentMat, "tar_contMap": tar_cont, "seq_length": seq_len})
            else:
                 for pred, target in zip(predictions, targets):
                    save_outputs.append({"prediction": pred, "target": target})
        
        ## loop over metrics and fill in metric_values
        for name, metric in zip(metric_names, metric_functions):
            if name == 'perplexity_subClass_AB':
                metric_values[name] = list(map(sum, zip(metric_values[name],metric(targets_subClassHLPair, pred_subClassHLPair_logits, normalize=False))))
            elif name == 'antibody_HL_likelihood':
                for bs_i in range(len(mut_names)):
                    metric_values[name]['seq_ids'].append(seq_ids[bs_i])
                    metric_values[name]['mut_names'].append(mut_names[bs_i])
                    metric_values[name]['aa_logits'].append(pred_aa_logits[bs_i])
                    metric_values[name]['wt_aa_ids'].append(labels_wt[bs_i])
                    metric_values[name]['mut_aa_ids'].append(labels_mut[bs_i])
            elif re.search(r'accuracy.*_subClass_AB',name) is not None:
                metric_values[name] = list(map(sum, zip(metric_values[name], metric(targets_subClassHLPair, pred_subClassHLPair_logits, normalize=False))))
            elif name in ['embed_antibody', 'embed_antibody_internal']:
                for bs_i in range(len(entityH)):
                    #if len(hidden_states_pooled[bs_i,:]) != model_config.hidden_size:
                    #    logger.info('embed size abnormal: {}'.format(len(hidden_states_pooled[bs_i,:])))
                    #    logger.info('task: {}, mlm_mask_stragy: {}'.format(task,mlm_mask_stragy))
                    ## extract hidden vec of seq
                    if task == 'antibody_embed_seqConcate':
                        token_type_lmax = token_type_ids[bs_i]
                        input_mask_lmax = input_masks[bs_i].astype(bool)
                        hidden_states_lastLayer_seq = hidden_states_encoder_lastLayer[bs_i,input_mask_lmax,:] #[n_pos,hidden_d]
                        hidden_states_transform_token_seq = hidden_states_transform_token[bs_i,input_mask_lmax,:]
                        token_type_seq = token_type_lmax[input_mask_lmax] #[n_pos,]
                        token_type_VH = token_type_seq == 0
                        token_type_VH[0] = False
                        token_type_VH[-1] = False
                        token_type_VL = token_type_seq == 1
                        token_type_VL[-1] = False
                        hidden_states_lastLayer_token_VH_seq = hidden_states_lastLayer_seq[token_type_VH,:] #[len_VH,hidden_d]
                        hidden_states_lastLayer_token_VL_seq = hidden_states_lastLayer_seq[token_type_VL,:] #[len_VL,hidden_d]
                        hidden_states_transform_token_VH_seq = hidden_states_transform_token_seq[token_type_VH,:] 
                        hidden_states_transform_token_VL_seq = hidden_states_transform_token_seq[token_type_VL,:]
                    elif task == 'antibody_embed_seqIndiv':
                        input_mask_VH = input_masks_VH[bs_i].astype(bool)
                        input_mask_VL = input_masks_VL[bs_i].astype(bool)
                        hidden_states_lastLayer_token_VH_seq = hidden_states_encoder_lastLayer_VH[bs_i,input_mask_VH,:][1:-1,:]
                        hidden_states_lastLayer_token_VL_seq = hidden_states_encoder_lastLayer_VL[bs_i,input_mask_VL,:][1:-1,:]
                        hidden_states_transform_token_VH_seq = hidden_states_transform_token_VH[bs_i,input_mask_VH,:][1:-1,:]
                        hidden_states_transform_token_VL_seq = hidden_states_transform_token_VL[bs_i,input_mask_VL,:][1:-1,:]
                    metric_values[name].append({'entityH': entityH[bs_i],
                                                'entityL': entityL[bs_i],
                                                'hidden_states_lastLayer_token_VL': hidden_states_lastLayer_token_VL_seq.tolist(),
                                                'hidden_states_lastLayer_token_VH': hidden_states_lastLayer_token_VH_seq.tolist(),
                                                'hidden_states_transform_token_VL': hidden_states_transform_token_VL_seq.tolist(),
                                                'hidden_states_transform_token_VH': hidden_states_transform_token_VH_seq.tolist(),
                                                'subClass_pair': int(targets_subClassHLPair[bs_i])})
            else:
                metric_values[name] = list(map(sum, zip(metric_values[name], metric(targets, predictions, normalize=False))))

    # get final value of each metric
    metric_outputs = {}
    for name, value in metric_values.items():
        if name == 'perplexity_subClass_AB':
            metric_outputs[f'AB_subClass_ece{mlm_maskStragy_id}'] = np.exp(value[0] / value[2])
            metric_outputs[f'AB_subClass_ppl{mlm_maskStragy_id}'] = value[1] / value[2]
        elif name == 'antibody_HL_likelihood':
            metric_fun = registry.get_metric('antibody_HL_likelihood')
            mut_seq_ids, mut_seq_names, likelihood_ratios = metric_fun(value)
            return mut_seq_ids, mut_seq_names, likelihood_ratios
        elif name == 'embed_antibody_internal':
            pretrain_set = re.split('/',from_pretrained)[-3]
            antibody_straty_set = re.split('/',from_pretrained)[-2]
            Path('{}/embeddings/{}'.format(data_dir,pretrain_set)).mkdir(parents=True, exist_ok=True)  
            with open('{}/embeddings/{}/{}.json'.format(data_dir,pretrain_set,antibody_straty_set),'w') as jfl:
              json.dump(value,jfl)
        elif name == 'embed_antibody':
            Path('{}/embeddings'.format(data_dir)).mkdir(parents=True, exist_ok=True)  
            with open('{}/embeddings/{}.json'.format(data_dir,split),'w') as jfl:
              json.dump(value,jfl)
        else:
            metric_outputs[f'{name}{mlm_maskStragy_id}'] = value[0] / value[1]
    
    if output_pred:
        return (metric_outputs, save_outputs)
    else:
        return (metric_outputs, None)

def run_train(model_type: str,
              task: str,
              learning_rate: float = 1e-4,
              batch_size: int = 1024,
              num_train_epochs: int = 10,
              num_train_total_epochs: int = 300,
              num_log_iter: int = 20,
              fp16: bool = False,
              fp16_opt_level: str = 'O1',
              warmup_steps: int = 10000,
              gradient_accumulation_steps: int = 1,
              loss_scale: int = 0,
              max_grad_norm: float = 1.0,
              exp_name: typing.Optional[str] = None,
              from_pretrained: typing.Optional[str] = None,
              pretrained_epoch: int = None,
              log_dir: str = './logs',
              pytorch_profiler: bool = False,
              eval_freq: int = 1,
              save_freq: typing.Union[int, str] = 1,
              save_freq_opt_checkpoint: typing.Union[int, str] = 'improvement',
              model_config_file: typing.Optional[str] = None, # only when 'from_pretrained' is None, config load from this file 
              extra_config_file: typing.Optional[str] = None, # after config load from 'from_pretrained', extra params here
              data_dir: str = './data',
              data_format: str = 'lmdb',
              train_split: str = 'train',
              valid_split: str = 'valid',
              output_dir: str = './results',
              no_cuda: bool = False,
              seed: int = 42,
              local_rank: int = -1,
              tokenizer: str = 'pfam',
              num_workers: int = 8,
              debug: bool = False,
              log_level: typing.Union[str, int] = logging.INFO,
              patience: int = -1,
              resume_from_checkpoint: bool = False,
              mlm_mask_stragy: str = 'vanilla',
              balancing: bool = True,
              lr_scheduler: str = 'constant',
              save_checkpoint: bool = True,
              neighbor_strategy: str = 'knn',
              knn_value: int = 20,
              dist_cutoff: float = 12.0) -> None:

    # SETUP AND LOGGING CODE #
    input_args = locals() # the dictionary of current local symbol table
    device, n_gpu, is_master, is_global_master = utils.setup_distributed(local_rank, no_cuda)

    exp_dir = utils.get_expname(exp_name, task, model_type)
    
    ## if 'best' is given as pretrained_epoch
    if isinstance(pretrained_epoch, str) and pretrained_epoch.lower() == 'best':
        pretrained_epoch = None

    if is_global_master:
        save_path = Path(output_dir) / exp_dir
        # save all the hidden parameters.
        save_path.mkdir(parents=True, exist_ok=True)
        with (save_path / 'args.json').open('w') as f:
            json.dump(input_args, f)
    else:
        save_path = None 

    utils.barrier_if_distributed()
    utils.setup_logging(is_master, is_global_master, save_path, log_level)
    utils.set_random_seeds(seed, n_gpu)

    if isinstance(tokenizer, str):
        tokenizer = BaseTokenizer(vocab=tokenizer)
    
    vocab_num = tokenizer.vocab_size

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained, extra_config_file, pretrained_epoch)
    model.resize_token_embeddings(vocab_num) ## append 'X' token; take care of tie_weights, resize mlm-head bias module
    model = model.to(device)

    # setup the datasets , model_config
    train_dataset = utils.setup_dataset(task, data_dir, train_split, tokenizer, data_format, in_memory=False, mlm_mask_stragy=mlm_mask_stragy, neighbor_strategy=neighbor_strategy, knn_value=knn_value, dist_cutoff=dist_cutoff, model_config=model.config)
    valid_dataset = utils.setup_dataset(task, data_dir, valid_split, tokenizer, data_format, in_memory=False, mlm_mask_stragy=mlm_mask_stragy, neighbor_strategy=neighbor_strategy, knn_value=knn_value, dist_cutoff=dist_cutoff, model_config=model.config)
    train_loader = utils.setup_loader(
        train_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers, balancing=balancing)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        gradient_accumulation_steps, num_workers)

    num_train_optimization_steps = utils.get_num_train_optimization_steps(
        train_dataset, batch_size, num_train_total_epochs)

    optimizer = utils.setup_optimizer(model, learning_rate)
    
    # setup log recorder
    ## only master gpu of each node has valid viz, others are dummy viz
    rank = int(os.environ["RANK"]) if 'RANK' in os.environ else 0
    viz = visualization.get(log_dir, exp_dir, local_rank, rank, debug=debug)
    viz.log_config(input_args)
    viz.log_config(model.config.to_dict())
    viz.watch(model)

    if 'RANK' in os.environ:
        logger.info(
            f"device: {device}, "
            f"n_gpu: {n_gpu}, "
            f"distributed_training: {local_rank != -1}, "
            f"local rank: {os.environ['LOCAL_RANK']}; world rank: {os.environ['RANK']}; world size: {os.environ['WORLD_SIZE']}, "
            f"16-bits training: {fp16}")
    else:
        logger.info(
            f"device: {device}, "
            f"n_gpu: {n_gpu}, "
            f"distributed_training: {local_rank != -1}, "
            f"16-bits training: {fp16}")

    runner = BackwardRunner(
        model, optimizer, gradient_accumulation_steps, device, n_gpu,
        fp16, fp16_opt_level,local_rank, max_grad_norm, warmup_steps, num_train_optimization_steps, lr_scheduler,num_train_total_epochs)

    runner.initialize_fp16()
    if resume_from_checkpoint:
        assert from_pretrained is not None
        start_epoch, start_gloStep = runner.resume_from_checkpoint(from_pretrained,pretrained_epoch)
    else:
        start_epoch, start_gloStep = 0, 0
    
    runner.initialize_distributed_model()
    runner._global_step = start_gloStep # set starting value of global steps

    if isinstance(save_freq, str) and save_freq != 'improvement':
        raise ValueError(
            f"Only recongized string value for save_freq is 'improvement'"
            f", received: {save_freq}")

    if save_freq == 'improvement' and eval_freq <= 0:
        raise ValueError("Cannot set save_freq to 'improvement' and eval_freq < 0")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num epochs (current cycle) = %d, %d to %d" % (num_train_epochs-start_epoch,start_epoch,num_train_epochs-1))
    logger.info("  Num total epochs = %d", num_train_total_epochs)
    logger.info("  Num total steps = %d", num_train_optimization_steps)
    logger.info("  Num parameters = %d", num_trainable_parameters)

    best_val_loss = float('inf')
    best_val_epoch = 0
    num_evals_no_improvement = 0
    
    def do_save(epoch_id: int, num_evals_no_improvement: int) -> bool:
        if not is_global_master:
            return False
        ## condition on 'save_freq'
        if isinstance(save_freq, int): # also save the best model so far
            return ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs) or (num_evals_no_improvement == 0)
        else:
            return num_evals_no_improvement == 0

    utils.barrier_if_distributed()

    # ACTUAL TRAIN/EVAL LOOP #

    with utils.wrap_cuda_oom_error(local_rank, batch_size, n_gpu, gradient_accumulation_steps):
        # before train, do one round of evaluation first
        # before 0th epoch, evaluate random initilized model
        # before kth epoch, a fix to last epoch's log not saved by TB in last round
        #_, _ = run_valid_epoch(start_epoch-1, valid_loader, runner, viz, is_master)

        for epoch_id in range(start_epoch, num_train_epochs):
            # save untrained model at epoch_id = 0
            if epoch_id == 0 and is_global_master:
                logger.info("** ** * Saving untrained model before epoch 0 ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id, save_freq, save_freq_opt_checkpoint, save_checkpoint, num_train_epochs, num_evals_no_improvement)
                logger.info(f"Saving model checkpoint to {save_path}")

            run_train_epoch(epoch_id, train_loader, runner,
                            viz, num_log_iter, gradient_accumulation_steps, log_dir=f'{log_dir}/{exp_dir}', pytorch_profiler=pytorch_profiler,start_epoch=start_epoch)
            if eval_freq > 0 and (epoch_id + 1) % eval_freq == 0:
                val_loss, _ = run_valid_epoch(epoch_id, valid_loader, runner, viz, is_master)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch_id
                    num_evals_no_improvement = 0
                else:
                    num_evals_no_improvement += 1
            
            # Save trained model
            if do_save(epoch_id, num_evals_no_improvement):
                logger.info("** ** * Saving trained model ** ** * ")
                # Only save the model itself
                runner.save_state(save_path, epoch_id, save_freq, save_freq_opt_checkpoint, save_checkpoint, num_train_epochs, num_evals_no_improvement)
                logger.info(f"Saving model checkpoint to {save_path}")

            utils.barrier_if_distributed()
            if patience > 0 and num_evals_no_improvement >= patience:
                logger.info(f"Finished training at epoch {epoch_id} because no "
                            f"improvement for {num_evals_no_improvement} epochs.")
                logger.log(35, f"Best Val Loss (early-stopping): {best_val_loss} at epoch {best_val_epoch}")
                
                if local_rank != -1:
                    # If you're distributed, raise this error. It sends a signal to
                    # the master process which lets it kill other processes and terminate
                    # without actually reporting an error. See utils/distributed_utils.py
                    # for the signal handling code.
                    raise errors.EarlyStopping
                else:
                    break
    logger.info(f"Finished training after {num_train_epochs} epochs.")
    
    if best_val_loss != float('inf') and is_global_master:
        logger.log(35, f"Best Val Loss: {best_val_loss} at epoch {best_val_epoch}")
    
    # close SummaryWriter in tensorBoardX
    # if tensorboard writer is not closed, EOFError will be raised in multiprocess setting
    if hasattr(viz,'close_logger'):
       viz.close_logger()

def run_eval(model_type: str,
             task: str,
             from_pretrained: str,
             pretrained_epoch: typing.Union[str, int] = None,
             split: str = 'holdout',
             batch_size: int = 1024,
             model_config_file: typing.Optional[str] = None,
             extra_config_file: typing.Optional[str] = None,
             data_dir: str = './data',
             eval_save_dir: str = './eval_results',
             data_format: str = 'lmdb',
             no_cuda: bool = False,
             local_rank: int = -1,
             seed: int = 42,
             tokenizer: str = 'pfam',
             num_workers: int = 8,
             debug: bool = False,
             metrics: typing.Tuple[str, ...] = (),
             log_level: typing.Union[str, int] = logging.INFO,
             mutgsis_set: str = None,
             mlm_mask_stragy: str = None,
             embed_modelNm: str = None,
             neighbor_strategy: str = 'knn',
             knn_value: int = 20,
             dist_cutoff: float = 8.0) -> typing.Dict[str, float]:

    # for solving `RuntimeError: received 0 items of ancdata`
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    local_rank = -1  #not support torch.distributed.launch for evaluation
    device, n_gpu, is_master, is_global_master = utils.setup_distributed(local_rank, no_cuda)
    logger.info(f"device: {device}, n_gpu: {n_gpu}")
    utils.setup_logging(is_master, is_global_master, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    if isinstance(tokenizer, str):
        tokenizer = BaseTokenizer(vocab=tokenizer)

    ## initilize useful variables ##
    vocab_num = tokenizer.vocab_size
    dt_nm = re.split('/', data_dir)[-1]
    ## if 'best' is given as pretrained_epoch
    if isinstance(pretrained_epoch, str) and pretrained_epoch.lower() == 'best':
        pretrained_epoch = None
    ## if embed_modelNm is given, at default use best epoch
    if embed_modelNm is not None:
        if re.search(r'rp75', embed_modelNm):
            pretrained_epoch = 224
        elif re.search(r'rp15', embed_modelNm):
            pretrained_epoch = 729
        else:
            Exception(f'invalid embed_modelNm {embed_modelNm}')

    

    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained, extra_config_file, pretrained_epoch)
    model.resize_token_embeddings(vocab_num) ## append 'X' token; take care of tie_weights, resize mlm-head bias module

    model = model.to(device)
    model_config = model.config # instance of BaseConfig

    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()
    valid_dataset = utils.setup_dataset(task, data_dir, split, tokenizer,
        data_format, in_memory=False, mutgsis_set=mutgsis_set, mlm_mask_stragy=mlm_mask_stragy, neighbor_strategy=neighbor_strategy, knn_value=knn_value, dist_cutoff=dist_cutoff, model_config=model_config)
    valid_loader = utils.setup_loader(
        valid_dataset, batch_size, local_rank, n_gpu,
        1, num_workers)

    metric_functions = []
    for name in metrics:
      if 'precision' in name:
        metric_functions.append(registry.get_metric('contact_precision'))
      elif 'contact_background' in name:
        metric_functions.append(registry.get_metric('contact_background_prec'))
      elif 'att_fnorm2' in name:
        continue
      elif 'loss' in name:
        continue
      elif 'all_pred_distribution' in name:
        metric_functions.append(registry.get_metric('all_pred_distribution'))
      else:
        metric_functions.append(registry.get_metric(name))
    metrics_to_save, outputs_to_save = run_eval_epoch(valid_loader, runner, metrics, metric_functions,
                                                      data_dir=data_dir, task=task,
                                                      from_pretrained=from_pretrained,
                                                      pretrained_epoch=pretrained_epoch,
                                                      model_config=model_config,
                                                      split=split,
                                                      eval_save_dir=eval_save_dir,
                                                      output_pred=False,
                                                      is_master=is_master,
                                                      mlm_mask_stragy=mlm_mask_stragy,
                                                      embed_modelNm=embed_modelNm,
                                                      mutgsis_set=mutgsis_set)
    
    if metrics_to_save:
      logger.info(f"eval_report*> {';'.join(f'{name}: {val}' for name, val in metrics_to_save.items())}")
    
      eval_path = f"{eval_save_dir}/{task}/predictions/{re.split('/',from_pretrained)[-1]}"
      Path(eval_path).mkdir(parents=True, exist_ok=True)
      ## antibody seq model
      mlm_maskStragy_id = f'_{mlm_mask_stragy}' if mlm_mask_stragy is not None else ''
      if pretrained_epoch is None:
        with (Path(eval_path) / f'results_metrics_{dt_nm}_{split}{mlm_maskStragy_id}.json').open('w') as f:
          json.dump(metrics_to_save, f, cls=NumpyEncoder)
      else:
        with (Path(eval_path) / f'results_metrics_{dt_nm}_{split}_{pretrained_epoch}{mlm_maskStragy_id}.json').open('w') as f:
          json.dump(metrics_to_save, f, cls=NumpyEncoder)

    if outputs_to_save is not None:
      if pretrained_epoch is None:
        with (Path(eval_path) / f'output_predictions_{dt_nm}_{split}.pkl').open('wb') as f:
          pkl.dump(outputs_to_save, f)
      else:
        with (Path(eval_path) / f'output_predictions_{dt_nm}_{split}_{pretrained_epoch}.pkl').open('wb') as f:
          pkl.dump(outputs_to_save, f)

    return metrics_to_save

def run_antibody_engineer(model_type: str,
                        task: str,
                        from_pretrained: str,
                        pretrained_epoch: typing.Union[str, int] = None,
                        batch_size: int = 1024,
                        model_config_file: typing.Optional[str] = None,
                        extra_config_file: typing.Optional[str] = None,
                        eval_save_dir: str = './eval_results',
                        no_cuda: bool = False,
                        local_rank: int = -1,
                        seed: int = 42,
                        tokenizer: str = 'pfam',
                        num_workers: int = 8,
                        log_level: typing.Union[str, int] = logging.INFO,
                        sa_config_path: str = None,
                        with_func_surrogates: bool = False,
                        seq_info_file: str = None,
                        mut_mode: str = 'both',
                        wt_anchor: bool = False,
                        assign_edit_num: bool = False,
                        ):
    """
    Pipeline w/o surrogate:
        1. mutate seq (make N random edits)
        2. get probability from AbLM
        3. accept or reject the proposed seq
        4. save intermediate results
    Pipeline w/ surrogate:
        1. mutate seq (make N random edits)
        2.1 get probability from AbLM
        2.2 get embeddings from AbLM
        2.3 get function probability from surrogate models
        3. accept or reject the proposed seq
        4. save intermediate results
    """
    # for solving `RuntimeError: received 0 items of ancdata`
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    local_rank = -1  #not support torch.distributed.launch for evaluation
    device, n_gpu, is_master, is_global_master = utils.setup_distributed(local_rank, no_cuda)
    logger.info(f"device: {device}, n_gpu: {n_gpu}")
    utils.setup_logging(is_master, is_global_master, save_path=None, log_level=log_level)
    utils.set_random_seeds(seed, n_gpu)

    if isinstance(tokenizer, str):
        tokenizer = BaseTokenizer(vocab=tokenizer)

    ## initilize useful variables ##
    vocab_num = tokenizer.vocab_size
    ## if 'best' is given as pretrained_epoch
    if isinstance(pretrained_epoch, str) and pretrained_epoch.lower() == 'best':
        pretrained_epoch = None

    print('>Load configs')
    with open(f'{sa_config_path}','r') as f:
        config_dict = json.load(f)
    config_dict['mut_mode'] = mut_mode
    config_dict['wt_anchor'] = wt_anchor
    config_dict['assign_edit_num'] = assign_edit_num

    ## append model name
    from_pretrained = f"{from_pretrained}/{config_dict['mdl_name']}"
    model = registry.get_task_model(model_type, task, model_config_file, from_pretrained, extra_config_file, pretrained_epoch)
    model.resize_token_embeddings(vocab_num) ## append 'X' token; take care of tie_weights, resize mlm-head bias module

    model = model.to(device)
    model_config = model.config # instance of BaseConfig

    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()

    metric_functions = []
    if with_func_surrogates:
        metrics = ['embed_antibody','antibody_HL_likelihood']
    else:
        metrics = ['antibody_HL_likelihood']
    for name in metrics:
        metric_functions.append(registry.get_metric(name))
    
    dataLoader_cfg = {'batch_size':batch_size,'local_rank':local_rank,'n_gpu':n_gpu,'num_workers':num_workers,'concate_stragy':config_dict['concate_stragy'],'tokenizer':tokenizer}
    run_eval_epoch_args = {'runner':runner,'metrics':metrics,'metric_functions':metric_functions,'from_pretrained':from_pretrained,'pretrained_epoch':pretrained_epoch,'model_config':model_config, 'eval_save_dir':eval_save_dir,'output_pred':False,'is_master':is_master,'mlm_mask_stragy':config_dict['mlm_mask_stragy'],'nmut_threshold':config_dict['nmut_threshold']}

    anneal_history = anneal(config_dict,seq_info_file,data_cfg=dataLoader_cfg,eval_cfg=run_eval_epoch_args)

    seq_set_nm = seq_info_file.split('.')[0]
    with open(f"{eval_save_dir}/outputs/{seq_set_nm}_{config_dict['output_file']}", 'wb') as handle:
        pkl.dump(anneal_history, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return 