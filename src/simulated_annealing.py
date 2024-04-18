import random, json, copy
import numpy as np
import pandas as pd
from typing import List

# self-defined modules
import utils

### constants ###
# Standard order AA alphabet
AA_ALPHABET_STANDARD_ORDER = 'ARNDCQEGHILKMFPSTWYV'
# SA related
SIM_ANNEAL_K = 1

def make_n_random_edits(seq:str,
                        nedits:int,
                        alphabet:str=AA_ALPHABET_STANDARD_ORDER,
                        design_pos_list:List=None,
                        design_pos_prob:List=None,
                        mut_curr_pos: List=None,
                        wt_seq:str=None):
    """Make n random missense mutations to the input seq
    """
    if wt_seq is not None:
        assert len(wt_seq) == len(seq), 'mut and wt seq length not matching'
    
    lseq = list(seq)
    lalphabet = list(alphabet)
    mut_pos = []
    # Create non-redundant list of positions to mutate.
    l = copy.deepcopy(design_pos_list) # position indices are zero-based
    nedits = min(len(l), nedits)
    
    if design_pos_prob is None:
        random.shuffle(l)
        pos_to_mutate = l[:nedits]
        pos_to_mutate = np.sort(pos_to_mutate) # sort proposed postions
    else:
        pos_to_mutate = np.random.choice(l, nedits, replace=False, p=design_pos_prob)
        pos_to_mutate = np.sort(pos_to_mutate)

    for i in range(nedits):
        pos = pos_to_mutate[i]
        aa_to_choose_from = list(set(lalphabet) - set([lseq[pos]]))
        lseq[pos] = aa_to_choose_from[np.random.randint(len(aa_to_choose_from))]
        mut_pos.append(pos)
    
    return "".join(lseq), mut_pos

def propose_seq_batch(seqs:List,
                      design_pos_list:List=None,
                      design_pos_prob:List=None,
                      seqs_extra:List=None,
                      design_pos_list_extra:List=None,
                      design_pos_prob_extra:List=None,
                      num_muts_per_seq:List=None,
                      uniform_Nedit_list: List=None,
                      mut_mode: str='both',
                      h_mutPos: List = None,
                      l_mutPos: List = None,
                      h_wt_seq: str=None,
                      l_wt_seq: str=None):
    """Propose a new batch of sequences
    For antibody, "seqs" is heavy chain and "seqs_extra" is light chain.
    """
    mut_seqs = []
    mut_pos = [] # ordered
    mut_seqs_extra = []
    mut_pos_extra = []
    if seqs_extra is not None:
        assert len(seqs) == len(seqs_extra) # check pair matching
    # unroll N_edit_list
    if uniform_Nedit_list:
        edit_pos_list, edit_prob_list = [], []
        for edit_pair in uniform_Nedit_list:
            edit_pos_list.append(edit_pair[0])
            edit_prob_list.append(edit_pair[1])
    else:
        edit_pos_list, edit_prob_list = None, None
    for i,s in enumerate(seqs):
        if edit_pos_list:
            n_edits = np.random.choice(edit_pos_list,size=1,replace=False,p=edit_prob_list)[0]
        else:
            n_edits = np.random.poisson(num_muts_per_seq[i]-1) + 1
        if mut_mode in ['both','heavy']:
            mut_seq, pos_list = make_n_random_edits(s, n_edits, design_pos_list=design_pos_list, design_pos_prob=design_pos_prob)
            mut_seqs.append(mut_seq)
            mut_pos.append(pos_list)
        else:
            mut_seqs.append(s)
            mut_pos.append([])
        if seqs_extra is not None:
            if edit_pos_list:
                n_edits = np.random.choice(edit_pos_list,size=1,replace=False,p=edit_prob_list)[0]
            else:
                n_edits = np.random.poisson(num_muts_per_seq[i]-1) + 1
            if mut_mode in ['both','light']:
                mut_seq, pos_list = make_n_random_edits(seqs_extra[i], n_edits, design_pos_list=design_pos_list_extra, design_pos_prob=design_pos_prob_extra)
                mut_seqs_extra.append(mut_seq)
                mut_pos_extra.append(pos_list)
            else:
                mut_seqs_extra.append(seqs_extra[i])
                mut_pos_extra.append([])
        
    if seqs_extra is not None:
        return mut_seqs, mut_pos, mut_seqs_extra, mut_pos_extra
    else:
        return mut_seqs, mut_pos

def acceptance_function_prob(f_proposal, f_current, k, T):
    """Calculate probability for acceptance
    """
    ap = np.exp((f_proposal - f_current)/(k*T))
    ap[ap > 1] = 1
    return ap

def acceptance_likelihood_prob(ratio, k, T):
    """Calculate probability for acceptance
    """
    alpha = 0.
    ap = np.exp((ratio-alpha)/(k*T))
    ap[ap > 1] = 1
    return ap

def load_init_seq_info(seq_info_path:str,seq_info_file:str):
    """Load starting sequence information
    """
    with open(f'{seq_info_path}/{seq_info_file}','r') as f:
        seq_info = json.load(f)
    h_str = seq_info['heavy_chain_seq']
    l_str = seq_info['light_chain_seq']
    h_design_pos_list = seq_info['heavy_design_pos']
    l_design_pos_list = seq_info['light_design_pos']
    h_design_pos_prob = seq_info['heavy_design_pos_prob']
    l_design_pos_prob = seq_info['light_design_pos_prob']
    return h_str, h_design_pos_list, h_design_pos_prob, l_str, l_design_pos_list, l_design_pos_prob

def assemble_state_seq_df(h_state_seqs:List,
                          h_mut_seqs:List,
                          h_mut_pos:List,
                          l_state_seqs:List,
                          l_mut_seqs: List,
                          l_mut_pos:List):
    """ convert seq List data to DataFrame
    keys: stateSeqVH,stateSeqVL,mutSeqVH,mutSeqVL,mutRelaIdxVH,mutRelaIdxVL,subclassH,subclassL
    """
    input_args = locals()
    num_trajc = len(h_state_seqs)
    for nm, arg in input_args.items():
        assert len(arg) == num_trajc, f'{nm} size {len(arg)} not match {num_trajc}'
    
    dummy_subClass = ['unknown']*num_trajc
    h_mut_num = [len(mut_pos) for mut_pos in h_mut_pos]
    l_mut_num = [len(mut_pos) for mut_pos in l_mut_pos]

    state_seq_df = pd.DataFrame(list(zip(h_state_seqs,l_state_seqs,h_mut_seqs,l_mut_seqs,h_mut_pos,l_mut_pos,h_mut_num,l_mut_num,dummy_subClass,dummy_subClass,list(range(num_trajc)))),columns=['stateSeqVH','stateSeqVL','mutSeqVH','mutSeqVL','mutRelaIdxVH','mutRelaIdxVL','mutNumVH','mutNumVL','subclassH','subclassL','seq_id'])

    return state_seq_df

def update_position_posterior():
    """Update the saliency map over positions
    """
    
    return

def get_AbLM_likelihood(h_state_seqs: List,
                        h_mut_seqs: List,
                        h_mut_pos: List,
                        l_state_seqs:List,
                        l_mut_seqs: List,
                        l_mut_pos: List,
                        data_cfg: dict,
                        eval_cfg: dict,
                        ):
    """setup dataloader and run likelihood inference
    """
    # to avoid circular import
    from training import run_eval_epoch
    
    #'stateSeqVH','stateSeqVL','mutSeqVH','mutSeqVL','mutRelaIdxVH','mutRelaIdxVL','mutNumVH','mutNumVL','subclassH','subclassL','seq_id'
    state_seqs_df = assemble_state_seq_df(h_state_seqs,h_mut_seqs,h_mut_pos,l_state_seqs,l_mut_seqs,l_mut_pos)
    
    valid_dataset = utils.setup_dataset(task='antibody_mutation_MLM', data_dir=None, split=None, tokenizer=data_cfg['tokenizer'], data_format=None, in_memory=None, concate_stragy=data_cfg['concate_stragy'], df_object=state_seqs_df)
    valid_loader = utils.setup_loader(valid_dataset, data_cfg['batch_size'], data_cfg['local_rank'], data_cfg['n_gpu'], 1, data_cfg['num_workers'])

    mut_seq_ids, mut_seq_names, likelihood_ratios = run_eval_epoch(valid_loader, eval_cfg['runner'], eval_cfg['metrics'], eval_cfg['metric_functions'], data_dir=None, task='antibody_mutation_MLM', from_pretrained=eval_cfg['from_pretrained'], pretrained_epoch=eval_cfg['pretrained_epoch'], model_config=eval_cfg['model_config'], split=None, eval_save_dir=eval_cfg['eval_save_dir'], output_pred=eval_cfg['output_pred'], is_master=eval_cfg['is_master'], concate_stragy=data_cfg['concate_stragy'])

    likelihood_ratio_df = pd.DataFrame(list(zip(mut_seq_ids,mut_seq_names,likelihood_ratios)),columns=['seq_id','seq_name','likelihood_ratio'])

    state_seqs_ratio_df = state_seqs_df.merge(likelihood_ratio_df,on=['seq_id'])
    assert len(state_seqs_ratio_df.index) == len(state_seqs_df.index),'state seq num un-consistent after LM eval'
    
    # filter out seqs with mutations larger then threshold
    nmut_threshold = eval_cfg['nmut_threshold']
    state_seqs_ratio_df.loc[(state_seqs_ratio_df['mutNumVH'] > nmut_threshold) | (state_seqs_ratio_df['mutNumVL'] > nmut_threshold),'likelihood_ratio'] = -np.inf
    
    return state_seqs_ratio_df

def anneal(config_dict: dict,
           seq_info_file: str,
           k:int = SIM_ANNEAL_K,
           data_cfg: dict=None,
           eval_cfg: dict=None,
           ):
    # unroll configs
    num_trajc = config_dict['num_trajc']
    sa_n_iter = config_dict['sa_n_iter']
    T_max = config_dict['T_max']
    temp_decay_rate = config_dict['temp_decay_rate']
    mut_mode = config_dict['mut_mode']
    wt_anchor = config_dict['wt_anchor']
    assign_edit_num = config_dict['assign_edit_num']

    ### modified based on Low-N paper
    print('>Initializing')
    h_seq, h_design_pos_list, h_design_pos_prob, l_seq, l_design_pos_list, l_design_pos_prob = load_init_seq_info(seq_info_path=config_dict['seq_info_path'],seq_info_file=seq_info_file)
    h_init_batch = [h_seq]*num_trajc
    l_init_batch = [l_seq]*num_trajc
    h_state_seqs = copy.deepcopy(h_init_batch)
    l_state_seqs = copy.deepcopy(l_init_batch)
    h_mutNum_batch = [0]*num_trajc
    l_mutNum_batch = [0]*num_trajc
    h_state_mutPos = [[]]*num_trajc
    l_state_mutPos = [[]]*num_trajc
    state_ratios = [-1*np.inf]*num_trajc

    # append initial seqs and likelihood to history
    h_seq_history = [copy.deepcopy(h_init_batch)]
    l_seq_history = [copy.deepcopy(l_init_batch)]
    ratio_history = [copy.deepcopy(state_ratios)]
    h_mutPos_history = [copy.deepcopy(h_state_mutPos)]
    l_mutPos_history = [copy.deepcopy(l_state_mutPos)]
    h_mutNum_history = [copy.deepcopy(h_mutNum_batch)]
    l_mutNum_history = [copy.deepcopy(l_mutNum_batch)]

    # mutation rate
    mu_muts_per_seq = 10*np.random.rand(num_trajc) + 3
    if assign_edit_num:
        uniform_Nedit_list = [[1,50/1000],[2,300/1000],[3,650/1000]]
    else:
        uniform_Nedit_list = None

    # start iteration
    for i in range(sa_n_iter):
        print('>Iteration:', i)

        print('\tProposing sequences.')
        h_proposal_seqs, h_proposal_mutPos, l_proposal_seqs, l_proposal_mutPos = propose_seq_batch(h_init_batch,h_design_pos_list,h_design_pos_prob,l_init_batch,l_design_pos_list,l_design_pos_prob,mu_muts_per_seq,uniform_Nedit_list=uniform_Nedit_list,mut_mode=mut_mode)
        
        print('\tCalculating likelihood.')
        if wt_anchor:
            proposal_ratios_df = get_AbLM_likelihood(h_init_batch,h_proposal_seqs,h_proposal_mutPos,l_init_batch,l_proposal_seqs,l_proposal_mutPos,data_cfg,eval_cfg)
        else:
            proposal_ratios_df = get_AbLM_likelihood(h_state_seqs,h_proposal_seqs,h_proposal_mutPos,l_state_seqs,l_proposal_seqs,l_proposal_mutPos,data_cfg,eval_cfg)
        
        proposal_ratios = proposal_ratios_df.sort_values(by='seq_id')['likelihood_ratio'].to_numpy()

        print('\tMaking acceptance/rejection decisions.')
        aprob = acceptance_likelihood_prob(proposal_ratios, k, T_max*(temp_decay_rate**i))
        
        print(f'\t**max prob:{max(aprob):.2f}')
        # Make sequence acceptance/rejection decisions
        for j, ap in enumerate(aprob):
            if np.random.rand() < ap:
                # accept
                h_state_seqs[j] = copy.deepcopy(h_proposal_seqs[j])
                l_state_seqs[j] = copy.deepcopy(l_proposal_seqs[j])
                state_ratios[j] = copy.deepcopy(proposal_ratios[j])
                h_state_mutPos[j] = copy.deepcopy(h_proposal_mutPos[j])
                l_state_mutPos[j] = copy.deepcopy(l_proposal_mutPos[j])
                h_mutNum_batch[j] = len(h_proposal_mutPos[j])
                l_mutNum_batch[j] = len(l_proposal_mutPos[j])
            # else do nothing (reject)
            
        h_seq_history.append(copy.deepcopy(h_state_seqs))
        l_seq_history.append(copy.deepcopy(l_state_seqs))
        ratio_history.append(copy.deepcopy(state_ratios))
        h_mutPos_history.append(copy.deepcopy(h_state_mutPos))
        l_mutPos_history.append(copy.deepcopy(l_state_mutPos))
        h_mutNum_history.append(copy.deepcopy(h_mutNum_batch))
        l_mutNum_history.append(copy.deepcopy(l_mutNum_batch))
        
    return {
        'h_seq_history': h_seq_history,
        'l_seq_history': l_seq_history,
        'ratio_history': ratio_history,
        'h_mutPos_history': h_mutPos_history,
        'l_mutPos_history': l_mutPos_history,
        'h_mutNum_history': h_mutNum_history,
        'l_mutNum_history': l_mutNum_history,
        'h_init_seq': h_seq,
        'l_init_seq': l_seq,
        'T_max': T_max,
        'mu_muts_per_seq': mu_muts_per_seq,
        'k': k,
        'n_iter': sa_n_iter,
        'temp_decay_rate': temp_decay_rate,
    }