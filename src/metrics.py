from typing import Sequence, Union, Tuple, Dict
import numpy as np
from scipy.special import softmax
from mapping import registry
import torch
from torch import nn
import matplotlib as mpl
mpl.use('Agg')

@registry.register_metric('accuracy_subClass_AB')
@registry.register_metric('accuracy')
def accuracy(target: Union[Sequence[int], Sequence[Sequence[int]]],
             prediction: Union[Sequence[float], Sequence[Sequence[float]]],
             normalize: bool = False,
             **kwargs) -> Union[float, Tuple[float, float]]:
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('accuracy_top3_subClass_AB')
@registry.register_metric('accuracy_top3')
def accuracy_top3(target: Union[Sequence[int], Sequence[Sequence[int]]],
                  prediction: Union[Sequence[float], Sequence[Sequence[float]]],
                  normalize: bool = False,
                  **kwargs) -> Union[float, Tuple[float, float]]:
    topK = 3
    if isinstance(target[0], int):
        # non-sequence case
        label_array = np.asarray(label)
        pred_array = np.asarray(score)
        pred_max_k = pred_array.argsort(axis=-1)[:, -topK:][:, ::-1]
        match_array = np.logical_or.reduce(pred_max_k==label_array, axis=-1)
        acc_score_topk = match_array.sum() / match_array.size

        return acc_score_topk
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            mask = label_array != -1
            label_mask_arr = label_array[mask].reshape(-1,1)
            pred_mask_arr = pred_array[mask]
            pred_max_k = pred_mask_arr.argsort(axis=-1)[:, -topK:][:, ::-1]
            match_array = np.logical_or.reduce(pred_max_k==label_mask_arr, axis=-1)
            correct += match_array.sum()
            total += match_array.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('accuracy_top5_subClass_AB')
@registry.register_metric('accuracy_top5')
def accuracy_top5(target: Union[Sequence[int], Sequence[Sequence[int]]],
                  prediction: Union[Sequence[float], Sequence[Sequence[float]]],
                  normalize: bool = False,
                  **kwargs) -> Union[float, Tuple[float, float]]:
    topK = 5
    if isinstance(target[0], int):
        # non-sequence case
        label_array = np.asarray(label)
        pred_array = np.asarray(score)
        pred_max_k = pred_array.argsort(axis=-1)[:, -topK:][:, ::-1]
        match_array = np.logical_or.reduce(pred_max_k==label_array, axis=-1)
        acc_score_topk = match_array.sum() / match_array.size

        return acc_score_topk
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            mask = label_array != -1
            label_mask_arr = label_array[mask].reshape(-1,1)
            pred_mask_arr = pred_array[mask]
            pred_max_k = pred_mask_arr.argsort(axis=-1)[:, -topK:][:, ::-1]
            match_array = np.logical_or.reduce(pred_max_k==label_mask_arr, axis=-1)
            correct += match_array.sum()
            total += match_array.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('accuracy_top10_subClass_AB')
@registry.register_metric('accuracy_top10')
def accuracy_top10(target: Union[Sequence[int], Sequence[Sequence[int]]],
                   prediction: Union[Sequence[float], Sequence[Sequence[float]]],
                   normalize: bool = False,
                   **kwargs) -> Union[float, Tuple[float,float]]:
    topK = 10
    if isinstance(target[0], int):
        # non-sequence case
        label_array = np.asarray(label)
        pred_array = np.asarray(score)
        pred_max_k = pred_array.argsort(axis=-1)[:, -topK:][:, ::-1]
        match_array = np.logical_or.reduce(pred_max_k==label_array, axis=-1)
        acc_score_topk = match_array.sum() / match_array.size

        return acc_score_topk
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score)
            mask = label_array != -1
            label_mask_arr = label_array[mask].reshape(-1,1)
            pred_mask_arr = pred_array[mask]
            pred_max_k = pred_mask_arr.argsort(axis=-1)[:, -topK:][:, ::-1]
            match_array = np.logical_or.reduce(pred_max_k==label_mask_arr, axis=-1)
            correct += match_array.sum()
            total += match_array.size
        if normalize:
          return correct / total
        else:
          return (correct, total)

@registry.register_metric('perplexity_subClass_AB')
@registry.register_metric('perplexity')
def perplexity(target: Union[Sequence[int], Sequence[Sequence[int]]],
               prediction: Union[Sequence[float], Sequence[Sequence[float]]],
               normalize: bool = False,
               **kwargs) -> Union[float,Tuple[float,float]]:
  '''
  ECE and perplexity evaluated as token level

  * ECE: exp(mean(per_maskedToken_ce list))
  * perplexity: mean(exp(per_maskedToken_ce list))
  '''
  maskedToken_count = 0.0
  #ce_total = 0
  ece_ce_sum_nn = 0.0  # accumulating sum of ce
  ppl_expCE_sum_nn = 0.0 # accumulating sum of exp(ce)
  for label, score in zip(target, prediction):
    label_array = np.asarray(label)
    #pred_array = np.asarray(score)
    mask = label_array != -1 #[L_max,]
    label_mask_arr = label_array[mask] #[k_maskedTokens,]
    #pred_mask_arr = pred_array[mask] # logit score
    #pred_mask_arr_prob = softmax(pred_mask_arr,axis=-1) # convert to probability
    
    if len(label_mask_arr) == 0:
      #print('no masked pos. label_array: {}'.format(label_array))
      continue
    else:
      # scipy version cross entropy
      #ce_sum = log_loss(label_mask_arr,pred_mask_arr_prob,normalize=False,labels=np.arange(pred_mask_arr.shape[-1]))
      #ce_total += ce_sum
      
      maskedToken_count += label_mask_arr.shape[0]
      # pytorch version cross entropy
      ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
      if len(label_array.shape) == 0:
        score_tensor = torch.from_numpy(np.array(score).reshape((1,-1)))
        label_tensor = torch.from_numpy(np.array(label).reshape(1))
      else:
        score_tensor = torch.from_numpy(score)
        label_tensor = torch.from_numpy(label)
      ce_values_nn = ce_loss(score_tensor,label_tensor).numpy() #[L_max,]
      ece_ce_sum_nn += np.sum(ce_values_nn[mask])
      ppl_expCE_sum_nn += np.sum(np.exp(ce_values_nn[mask]))

  if normalize:
    return (np.exp(ece_ce_sum_nn / maskedToken_count), ppl_expCE_sum_nn / maskedToken_count)
  else:
    return (ece_ce_sum_nn,ppl_expCE_sum_nn,maskedToken_count)

@registry.register_metric('save_embedding')
@registry.register_metric('embed_antibody')
@registry.register_metric('embed_antibody_internal')
def none_metric():
  return

@registry.register_metric('antibody_HL_likelihood')
def antibody_HL_pair_prob(value: dict,
                          **kwargs) -> float:
  """Evaluate sequence probability (MLM way) of antibody heavy-light paired chains

  Args:
    value: dict of keys
              seq_ids: seq interger identifier
              mut_names: mutation names, e.g. 'H-A5R:H20D&L-K12H:P52R'
              aa_logits: AA pred logits, size [L,num_token]
              wt_aa_ids: AA wt ids, size [L,]
              mut_aa_ids: AA mut ids, size [L,]
  Returns:
    mutation names
    zero-shot fitness preditions
  """
  class_channel_last = kwargs.get('class_channel_last',True)
  fitness_pred_list = []
  for bs_i in range(len(value['mut_names'])):
    aa_logits = value['aa_logits'][bs_i]
    wt_aa_ids = value['wt_aa_ids'][bs_i]
    mut_aa_ids = value['mut_aa_ids'][bs_i]
    aa_masks = wt_aa_ids != -1
    if class_channel_last:
      aa_mask_logits = aa_logits[aa_masks,:] #[n_mask,aa_tokens]
      aa_mask_logits = np.transpose(aa_mask_logits)
    else:
      aa_mask_logits = aa_logits[:,aa_masks] #[aa_tokens,n_mask]

    ## un-normalized to 20 AA tokens
    wt_mask_ids = wt_aa_ids[aa_masks]
    mut_mask_ids = mut_aa_ids[aa_masks]
    aa_mask_softmax = softmax(aa_mask_logits,axis=0)
    prob_wt, prob_mut = 1.0,1.0
    for p in range(aa_mask_softmax.shape[1]):
      prob_wt *= aa_mask_softmax[:,p][wt_mask_ids[p]]
      prob_mut *= aa_mask_softmax[:,p][mut_mask_ids[p]]
    fitness_pred_list.append(np.log((prob_mut / prob_wt).item()))

    ## normalized to 20 AA tokens (after taking ratio, 29-normalized is the same as 20-normalized)
    # aa_mask_20_logits = aa_mask_logits[PFAM_VOCAB_20AA_IDX,:]
    # wt_mask_ids = wt_aa_ids[aa_masks]
    # mut_mask_ids = mut_aa_ids[aa_masks]
    # aa_mask_softmax = softmax(aa_mask_20_logits,axis=0)
    # prob_wt, prob_mut = 1.0,1.0
    # for p in range(aa_mask_softmax.shape[1]):
    #   prob_wt *= aa_mask_softmax[:,p][PFAM_VOCAB_20AA_IDX_MAP[wt_mask_ids[p]]]
    #   prob_mut *= aa_mask_softmax[:,p][PFAM_VOCAB_20AA_IDX_MAP[mut_mask_ids[p]]]
    # if math.isfinite(value['fitness_gt'][bs_i]):
    #   fitness_unSV_list_20renor.append(np.log((prob_mut / prob_wt).item()))
    # ## process mutation name
    # fitness_unSV_list_save_20renor.append(np.log((prob_mut / prob_wt).item()))  

  return value['seq_ids'], value['mut_names'], fitness_pred_list