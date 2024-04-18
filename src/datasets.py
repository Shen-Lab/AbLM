from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy, deepcopy
from pathlib import Path
import pickle as pkl
import logging
import random
from itertools import chain

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform

from tokenizers import BaseTokenizer, ab_H_subclass, ab_L_subclass, ab_HL_subclass
from mapping import registry

logger = logging.getLogger(__name__)


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    in_memr_type = kwargs.get('in_memr_type','dataframe')
    
    if data_file is not None:
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if data_file.suffix == '.lmdb':
            return LMDBDataset(data_file, *args, **kwargs)
        elif data_file.suffix in {'.fasta', '.fna', '.ffn', '.faa', '.frn'}:
            return FastaDataset(data_file, *args, **kwargs)
        elif data_file.suffix == '.json':
            return JSONDataset(data_file, *args, **kwargs)
        elif data_file.is_dir():
            return NPZDataset(data_file, *args, **kwargs)
        else:
            raise ValueError(f"Unrecognized datafile type {data_file.suffix}")
    else:
        if in_memr_type == 'dataframe':
            df_object = kwargs.get('df_object',None)
            if df_object is None:
                raise ValueError(df_object)
            else:
                return DataFrameDataset(df_object, *args, **kwargs)

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.amax([seq.shape for seq in sequences], axis=0).tolist()
    #shape = [batch_size] + [475]*len(sequences[0].shape)


    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)
    else:
        raise ValueError(f'invalid element type {dtype}')

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

class FastaDataset(Dataset):
    """Creates a dataset from a fasta file.
    Args:
        data_file (Union[str, Path]): Path to fasta file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        from Bio import SeqIO
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        # if in_memory:
        cache = list(SeqIO.parse(str(data_file), 'fasta'))
        num_examples = len(cache)
        self._cache = cache
        # else:
            # records = SeqIO.index(str(data_file), 'fasta')
            # num_examples = len(records)
#
            # if num_examples < 10000:
                # logger.info("Reading full fasta file into memory because number of examples "
                            # "is very low. This loads data approximately 20x faster.")
                # in_memory = True
                # cache = list(records.values())
                # self._cache = cache
            # else:
                # self._records = records
                # self._keys = list(records.keys())

        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        # if self._in_memory and self._cache[index] is not None:
        record = self._cache[index]
        # else:
            # key = self._keys[index]
            # record = self._records[key]
            # if self._in_memory:
                # self._cache[index] = record

        item = {'id': record.id,
                'primary': str(record.seq),
                'protein_length': len(record.seq)}
        return item

class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=126, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))
        
        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = None
        self._in_memory = in_memory
        self._num_examples = num_examples
        self.data_file = str(data_file)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        
        if self._env is None:
            self._env = lmdb.open(self.data_file, max_readers=126, readonly=True,
                        lock=False, readahead=False, meminit=False)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        self._env = None
        return item

class DataFrameDataset(Dataset):
    """Creates a dataset from dataframe.
    """

    def __init__(self,
                 df_in_memr: pd.DataFrame,
                 **kwargs):

        self.df = df_in_memr
        self._num_examples = len(df_in_memr.index)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        
        item = self.df.iloc[index].to_dict()
        return item

class JSONDataset(Dataset):
    """Creates a dataset from a json file. Assumes that data is
       a JSON serialized list of record, where each record is
       a dictionary.
    Args:
        data_file (Union[str, Path]): Path to json file.
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self, data_file: Union[str, Path], in_memory: bool = True):
        import json
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        records = json.loads(data_file.read_text())
        # *insufficient shared memory*: python list has 'copy-on-write' which
        # will gradually use up memory. Check this post:
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-445446603
        # Convert python distionary list to pandas dataFrame
        if not isinstance(records, list):
            raise TypeError(f"TAPE JSONDataset requires a json serialized list, "
                            f"received {type(records)}")

        records = pd.DataFrame(records)
        self._records = records
        self._num_examples = len(records)

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        item = dict(self._records.loc[index,:])
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = str(index)
        return item

class NPZDataset(Dataset):
    """Creates a dataset from a directory of npz files.
    Args:
        data_file (Union[str, Path]): Path to directory of npz files
        in_memory (bool): Dummy variable to match API of other datasets
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = True,
                 split_files: Optional[Collection[str]] = None):
        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)
        if not data_file.is_dir():
            raise NotADirectoryError(data_file)
        file_glob = data_file.glob('*.npz')
        if split_files is None:
            file_list = list(file_glob)
        else:
            split_files = set(split_files)
            if len(split_files) == 0:
                raise ValueError("Passed an empty split file set")

            file_list = [f for f in file_glob if f.name in split_files]
            if len(file_list) != len(split_files):
                num_missing = len(split_files) - len(file_list)
                raise FileNotFoundError(
                    f"{num_missing} specified split files not found in directory")

        if len(file_list) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_file}")

        self._file_list = file_list

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)

        item = dict(np.load(self._file_list[index]))
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset to contain a list of dictionary "
                            f"records, received record of type {type(item)}")
        if 'id' not in item:
            item['id'] = self._file_list[index].stem
        return item

@registry.register_task('antibody_mlm_seqConcate')
class ABSeqConcateMaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        
        data_path = Path(data_path)
        data_file = f"HL_pair_{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokens_pair = tokens_VH + ['<sep>'] + tokens_VL
        tokens = self.tokenizer.add_special_tokens(tokens_pair)
        if self.mask_stragy == 'vanilla':
            masked_tokens, labels = self._apply_bert_mask(tokens)
            masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        elif self.mask_stragy == 'cdr_vanilla':
            masked_tokens, labels = self._apply_bert_mask_cdrVani(tokens,len(tokens_VH),len(tokens_VL),
                                        [item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']],[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids = np.array(self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)
        elif self.mask_stragy == 'cdr_margin':
            ## size change: [L,] -> [6,L] (each seq mask one whole cdr region, 3+3)
            masked_tokens, labels = self._apply_bert_mask_cdr_margin(tokens,len(tokens_VH),len(tokens_VL),
                                        [item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']],[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens], np.int64)
        elif self.mask_stragy == 'cdr_pair':
            ## size change: [L,] -> [9,L] (each seq mask one whole cdr region, 3*3)
            masked_tokens, labels = self._apply_bert_mask_cdr_pair(tokens,len(tokens_VH),len(tokens_VL),
                                        [item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']],[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens], np.int64)
        else:
            raise ValueError('Unrecognized MLM mask strategy: {}'.format(self.mask_stragy))
        input_mask = np.ones_like(masked_token_ids)
        ## token_type_ids(segment_id)
        ## <cls> VH <sep> VL <sep> <pad> ...
        ##   0  {0}   0  {1}   1     1 ...
        aug_size = 1 if len(masked_token_ids.shape) == 1 else masked_token_ids.shape[0]
        token_type_ids = np.array([[0] + [0]*len(tokens_VH) + [0] + [1]*len(tokens_VL) + [1]]*aug_size, np.int64).squeeze()
        subClassH = np.array([ab_H_subclass[item['subclassH']]]*aug_size).squeeze() if item['subclassH'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
        subClassL = np.array([ab_L_subclass[item['subclassL']]]*aug_size).squeeze() if item['subclassL'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
        if item['subclassH'].lower() != 'unknown' and item['subclassL'].lower() != 'unknown':
            subClassHLPair = np.array([ab_HL_subclass['{}-{}'.format(item['subclassH'],item['subclassL'])]]*aug_size).squeeze()
        else:
            subClassHLPair = np.array([-1]*aug_size).squeeze()
        return masked_token_ids, input_mask, labels, token_type_ids, subClassH, subClassL, subClassHLPair

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, lm_label_ids, token_type_ids, subClassH, subClassL, subClassHLPair = tuple(zip(*batch))
        
        ## reshape mini-batch of cdrOne masked seqs
        ## e.g. input_ids: tuple -> list, [bs,augment_size,...] -> [bs*augment_size,...]
        if self.mask_stragy == 'cdr_margin' or self.mask_stragy == 'cdr_pair':
            input_ids = list(chain(*input_ids))
            input_mask = list(chain(*input_mask))
            lm_label_ids = list(chain(*lm_label_ids))
            token_type_ids = list(chain(*token_type_ids))
            subClassH = list(chain(*subClassH))
            subClassL = list(chain(*subClassL))
            subClassHLPair = list(chain(*subClassHLPair))

        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        token_type_ids = torch.from_numpy(pad_sequences(token_type_ids, 1))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))
        subClassH = torch.LongTensor(np.array(subClassH))
        subClassL = torch.LongTensor(np.array(subClassL))
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'targets': lm_label_ids,
                'token_type_ids': token_type_ids,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)
                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass
                masked_tokens[i] = token    
        return masked_tokens, labels

    def _apply_bert_mask_cdrVani(self, tokens: List[str],
                                 lenVH: int, lenVL: int, 
                                 cdrHIdx: List[List], cdrLIdx: List[List]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrIdx_list = list(range(cdrHIdx[0][0]+1,cdrHIdx[0][1]+2)) + \
                      list(range(cdrHIdx[1][0]+1,cdrHIdx[1][1]+2)) + \
                      list(range(cdrHIdx[2][0]+1,cdrHIdx[2][1]+2)) + \
                      list(range(cdrLIdx[0][0]+lenVH+2,cdrLIdx[0][1]+lenVH+3)) + \
                      list(range(cdrLIdx[1][0]+lenVH+2,cdrLIdx[1][1]+lenVH+3)) + \
                      list(range(cdrLIdx[2][0]+lenVH+2,cdrLIdx[2][1]+lenVH+3))
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            ## non-cdr region, only change residue
            if i not in cdrIdx_list:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        #token = self.tokenizer.mask_token
                        pass
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token 
            else:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token
        return masked_tokens, labels
    
    def _apply_bert_mask_cdr_margin(self, tokens: List[str],
                                    lenVH: int, lenVL: int, 
                                    cdrHIdx: List[List], cdrLIdx: List[List]) -> Tuple[List[List], List[List]]:
        masked_tokens_aug = []
        labels_aug = []
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrHIdx_list = [list(range(cdrHIdx[0][0]+1,cdrHIdx[0][1]+2)),list(range(cdrHIdx[1][0]+1,cdrHIdx[1][1]+2)),list(range(cdrHIdx[2][0]+1,cdrHIdx[2][1]+2))]
        cdrLIdx_list = [list(range(cdrLIdx[0][0]+lenVH+2,cdrLIdx[0][1]+lenVH+3)),list(range(cdrLIdx[1][0]+lenVH+2,cdrLIdx[1][1]+lenVH+3)),list(range(cdrLIdx[2][0]+lenVH+2,cdrLIdx[2][1]+lenVH+3))]
        cdrAllIdx_list = cdrHIdx_list + cdrLIdx_list
        for cdr_i in range(len(cdrAllIdx_list)): 
            cdr_range = cdrAllIdx_list[cdr_i]
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                # also stop token is used as separation token between H/L
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                ## mask cdr regions in H/L chain
                if i in cdr_range:
                    labels[i] = self.tokenizer.convert_token_to_id(token) 
                    token = self.tokenizer.mask_token
                    masked_tokens[i] = token
                else:
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15
                        labels[i] = self.tokenizer.convert_token_to_id(token)
                        if prob < 0.8:
                            # 80% random change to mask token
                            #token = self.tokenizer.mask_token
                            pass
                        elif prob < 0.9:
                            # 10% chance to change to random token(not special tokens)
                            token = self.tokenizer.convert_id_to_token(
                                random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                        else:
                            # 10% chance to keep current token
                            pass
                        masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

    def _apply_bert_mask_cdr_pair(self, tokens: List[str],
                                    lenVH: int, lenVL: int, 
                                    cdrHIdx: List[List], cdrLIdx: List[List]) -> Tuple[List[List], List[List]]:
        masked_tokens_aug = []
        labels_aug = []
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrHIdx_list = [list(range(cdrHIdx[0][0]+1,cdrHIdx[0][1]+2)),list(range(cdrHIdx[1][0]+1,cdrHIdx[1][1]+2)),list(range(cdrHIdx[2][0]+1,cdrHIdx[2][1]+2))]
        cdrLIdx_list = [list(range(cdrLIdx[0][0]+lenVH+2,cdrLIdx[0][1]+lenVH+3)),list(range(cdrLIdx[1][0]+lenVH+2,cdrLIdx[1][1]+lenVH+3)),list(range(cdrLIdx[2][0]+lenVH+2,cdrLIdx[2][1]+lenVH+3))]
        for cdrh in range(len(cdrHIdx_list)):
            for cdrl in range(len(cdrLIdx_list)): 
                cdrh_range = cdrHIdx_list[cdrh]
                cdrl_range = cdrLIdx_list[cdrl]
                masked_tokens = copy(tokens)
                labels = np.zeros([len(tokens)], np.int64) - 1
                for i, token in enumerate(tokens):
                    # Tokens begin and end with start_token and stop_token, ignore these
                    # also stop token is used as separation token between H/L
                    if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                        continue
                    ## mask cdr regions in H/L chain
                    if i in cdrh_range or i in cdrl_range:
                        labels[i] = self.tokenizer.convert_token_to_id(token) 
                        token = self.tokenizer.mask_token
                        masked_tokens[i] = token
                    else:
                        prob = random.random()
                        if prob < 0.15:
                            prob /= 0.15
                            labels[i] = self.tokenizer.convert_token_to_id(token)
                            if prob < 0.8:
                                # 80% random change to mask token
                                #token = self.tokenizer.mask_token
                                pass
                            elif prob < 0.9:
                                # 10% chance to change to random token(not special tokens)
                                token = self.tokenizer.convert_id_to_token(
                                    random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                            else:
                                # 10% chance to keep current token
                                pass
                            masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

@registry.register_task('antibody_embed_seqConcate')
class ABSeqConcateEmbedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}" # 'HL_pair_{split}'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokens_pair = tokens_VH + ['<sep>'] + tokens_VL
        tokens_pair = self.tokenizer.add_special_tokens(tokens_pair)
        token_ids = np.array(self.tokenizer.convert_tokens_to_ids(tokens_pair), np.int64)
        input_mask = np.ones_like(token_ids)
        ## token_type_ids(segment_id)
        ## <cls> VH <sep> VL <sep> <pad> ...
        ##   0  {0}   0  {1}   1     1 ...
        token_type_ids = np.array([[0] + [0]*len(tokens_VH) + [0] + [1]*len(tokens_VL) + [1]], np.int64).squeeze()
        subClassH_str = item['subclassH'] if 'subclassH' in item.keys() else 'unknown'
        subClassL_str = item['subclassL'] if 'subclassL' in item.keys() else 'unknown'
        subClassH = ab_H_subclass[subClassH_str]
        subClassL = ab_L_subclass[subClassL_str]
        subClassHLPair = ab_HL_subclass['{}-{}'.format(subClassH_str,subClassL_str)]
        return token_ids, input_mask, token_type_ids, subClassH, subClassL, subClassHLPair, item['entityH'], item['entityL']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, input_mask, token_type_ids, subClassH, subClassL, subClassHLPair, entityH, entityL = tuple(zip(*batch))
        
        input_ids = torch.from_numpy(pad_sequences(input_ids, 0))
        input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
        token_type_ids = torch.from_numpy(pad_sequences(token_type_ids, 1))
        # ignore_index is -1
        subClassH = torch.LongTensor(np.array(subClassH))
        subClassL = torch.LongTensor(np.array(subClassL))
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids': input_ids,
                'input_mask': input_mask,
                'token_type_ids': token_type_ids,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair,
                'entityH': entityH,
                'entityL': entityL}

@registry.register_task('antibody_mlm_seqIndiv')
class ABSeqIndivMaskedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
       Heavy and light chain seqs are encoded individually
       Self-attention for intra-seq; across-attention for inter-seq
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        #print('**mask_stragy:{}**'.format(self.mask_stragy))
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"HL_pair_{split}.{file_format}"
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokensSE_VH = self.tokenizer.add_special_tokens(tokens_VH)
        tokensSE_VL = self.tokenizer.add_special_tokens(tokens_VL)
        masked_token_ids_VH, masked_token_ids_VL = None, None
        if self.mask_stragy == 'vanilla':
            masked_tokens_VH, labels_VH = self._apply_bert_mask(tokensSE_VH)
            masked_tokens_VL, labels_VL = self._apply_bert_mask(tokensSE_VL)
            masked_token_ids_VH = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VH), np.int64)
            masked_token_ids_VL = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VL), np.int64)
        elif self.mask_stragy == 'cdr_vanilla':
            masked_tokens_VH, labels_VH = self._apply_bert_mask_cdrVani(tokensSE_VH,[item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']])
            masked_tokens_VL, labels_VL = self._apply_bert_mask_cdrVani(tokensSE_VL,[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids_VH = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VH), np.int64)
            masked_token_ids_VL = np.array(
                self.tokenizer.convert_tokens_to_ids(masked_tokens_VL), np.int64)
        elif self.mask_stragy == 'cdr_margin':
            ## size change: [L,] -> [6,L] (each seq mask one whole cdr region)
            masked_tokens_VH_cdr, labels_VH_cdr = self._apply_bert_mask_cdrOne(tokensSE_VH,[item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']])
            masked_tokens_VH_noise, labels_VH_noise = self._apply_bert_mask_noise(tokensSE_VH)
            masked_tokens_VH = np.concatenate((masked_tokens_VH_cdr,masked_tokens_VH_noise), axis=0)
            labels_VH = np.concatenate((labels_VH_cdr,labels_VH_noise),axis=0)

            masked_tokens_VL_cdr, labels_VL_cdr = self._apply_bert_mask_cdrOne(tokensSE_VL,[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_tokens_VL_noise, labels_VL_noise = self._apply_bert_mask_noise(tokensSE_VL)
            masked_tokens_VL = np.concatenate((masked_tokens_VL_cdr,masked_tokens_VL_noise), axis=0)
            labels_VL = np.concatenate((labels_VL_noise,labels_VL_cdr),axis=0)

            masked_token_ids_VH = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VH], np.int64)
            masked_token_ids_VL = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VL], np.int64)
        elif self.mask_stragy == 'cdr_pair':
            ## size change: [L,] -> [9,L] (each seq mask one whole cdr region)
            masked_tokens_VH_unpair, labels_VH_unpair = self._apply_bert_mask_cdrOne(tokensSE_VH,[item['cdr1HIdx'],item['cdr2HIdx'],item['cdr3HIdx']])
            masked_tokens_VL_unpair, labels_VL_unpair = self._apply_bert_mask_cdrOne(tokensSE_VL,[item['cdr1LIdx'],item['cdr2LIdx'],item['cdr3LIdx']])
            masked_token_ids_VH = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VH_unpair], np.int64)
            masked_token_ids_VL = np.array([self.tokenizer.convert_tokens_to_ids(masked_tokens_i) for masked_tokens_i in masked_tokens_VL_unpair], np.int64)
            masked_token_ids_VH = np.repeat(masked_token_ids_VH,[3,3,3],axis=0) #[a1,a1,a1,b1,b1,b1,c1,c1,c1]
            masked_token_ids_VL = np.tile(masked_token_ids_VL,(3,1)) #[a2,b2,c2,a2,b2,c2,a2,b2,c2]
            labels_VH = np.repeat(labels_VH_unpair,[3,3,3],axis=0)
            labels_VL = np.tile(labels_VL_unpair,(3,1))
        else:
            raise ValueError('Unrecognized MLM mask strategy: {}'.format(self.mask_stragy))
        assert (masked_token_ids_VH is not None) and (masked_token_ids_VL is not None)
        input_mask_VH = np.ones_like(masked_token_ids_VH)
        input_mask_VL = np.ones_like(masked_token_ids_VL)
        aug_size = 1 if len(masked_token_ids_VH.shape) == 1 else masked_token_ids_VH.shape[0]
        subClassH = np.array([ab_H_subclass[item['subclassH']]]*aug_size).squeeze()
        subClassL = np.array([ab_L_subclass[item['subclassL']]]*aug_size).squeeze()
        subClassHLPair = np.array([ab_HL_subclass['{}-{}'.format(item['subclassH'],item['subclassL'])]]*aug_size).squeeze()
        return masked_token_ids_VH, masked_token_ids_VL,\
               input_mask_VH, input_mask_VL,\
               labels_VH, labels_VL,\
               subClassH, subClassL, subClassHLPair, \
               item['entityH'], item['entityL']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_VH, input_ids_VL, input_mask_VH, input_mask_VL,\
        lm_label_ids_VH, lm_label_ids_VL, subClassH, subClassL, subClassHLPair, entityH, entityL = tuple(zip(*batch))

        ## reshape mini-batch of cdrOne masked seqs
        ## e.g. input_ids: tuple -> list, [bs,augment_size,...] -> [bs*augment_size,...]
        if self.mask_stragy == 'cdr_margin' or self.mask_stragy == 'cdr_pair':
            input_ids_VH = list(chain(*input_ids_VH))
            input_ids_VL = list(chain(*input_ids_VL))
            input_mask_VH = list(chain(*input_mask_VH))
            input_mask_VL = list(chain(*input_mask_VL))
            lm_label_ids_VH = list(chain(*lm_label_ids_VH))
            lm_label_ids_VL = list(chain(*lm_label_ids_VL))
            subClassH = list(chain(*subClassH))
            subClassL = list(chain(*subClassL))
            subClassHLPair = list(chain(*subClassHLPair))

        input_ids_VH = torch.from_numpy(pad_sequences(input_ids_VH, 0))
        input_ids_VL = torch.from_numpy(pad_sequences(input_ids_VL, 0))
        input_mask_VH = torch.from_numpy(pad_sequences(input_mask_VH, 0))
        input_mask_VL = torch.from_numpy(pad_sequences(input_mask_VL, 0))
        # ignore_index is -1
        lm_label_ids_VH = torch.from_numpy(pad_sequences(lm_label_ids_VH, -1))
        lm_label_ids_VL = torch.from_numpy(pad_sequences(lm_label_ids_VL, -1))
        subClassH = torch.LongTensor(np.array(subClassH))  # type: ignore
        subClassL = torch.LongTensor(np.array(subClassL))  # type: ignore
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids_VH': input_ids_VH,
                'input_ids_VL': input_ids_VL,
                'input_mask_VH': input_mask_VH,
                'input_mask_VL': input_mask_VL,
                'targets_VH': lm_label_ids_VH,
                'targets_VL': lm_label_ids_VL,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair,
                'entityH': entityH,
                'entityL': entityL}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)
                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token(not special tokens)
                    token = self.tokenizer.convert_id_to_token(
                        random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                else:
                    # 10% chance to keep current token
                    pass
                masked_tokens[i] = token    
        return masked_tokens, labels

    def _apply_bert_mask_noise(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens_aug = []
        labels_aug = []
        for cpy in range(3):
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                # also stop token is used as separation token between H/L
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        #token = self.tokenizer.mask_token
                        pass
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token    
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

    def _apply_bert_mask_cdrVani(self, tokens: List[str],
                                 cdrIdx: List[List]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrIdx_list = list(range(cdrIdx[0][0]+1,cdrIdx[0][1]+2)) + \
                      list(range(cdrIdx[1][0]+1,cdrIdx[1][1]+2)) + \
                      list(range(cdrIdx[2][0]+1,cdrIdx[2][1]+2))
        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            # also stop token is used as separation token between H/L
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                continue
            prob = random.random()
            ## non-cdr region, only change residue
            if i not in cdrIdx_list:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        #token = self.tokenizer.mask_token
                        pass
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token 
            else:
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)
                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token(not special tokens)
                        token = self.tokenizer.convert_id_to_token(
                            random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                    else:
                        # 10% chance to keep current token
                        pass
                    masked_tokens[i] = token
        return masked_tokens, labels
    def _apply_bert_mask_cdrOne(self, tokens: List[str],
                                cdrIdx: List[List]) -> Tuple[List[List], List[List]]:
        masked_tokens_aug = []
        labels_aug = []
        ## rescale index of cdr region
        ## <cls>,VH,<sep>,<VL>,<sep>
        cdrIdx_list = [list(range(cdrIdx[0][0]+1,cdrIdx[0][1]+2)),list(range(cdrIdx[1][0]+1,cdrIdx[1][1]+2)),list(range(cdrIdx[2][0]+1,cdrIdx[2][1]+2))]
        for cdr in range(len(cdrIdx_list)):
            cdr_range = cdrIdx_list[cdr]
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                # also stop token is used as separation token between H/L
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                    continue
                ## mask cdr region
                if i in cdr_range:
                    labels[i] = self.tokenizer.convert_token_to_id(token) 
                    token = self.tokenizer.mask_token
                    masked_tokens[i] = token
                else:
                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15
                        labels[i] = self.tokenizer.convert_token_to_id(token)
                        if prob < 0.8:
                            # 80% random change to mask token
                            #token = self.tokenizer.mask_token
                            pass
                        elif prob < 0.9:
                            # 10% chance to change to random token(not special tokens)
                            token = self.tokenizer.convert_id_to_token(
                                random.sample(self.tokenizer.get_normal_token_ids(),1)[0])
                        else:
                            # 10% chance to keep current token
                            pass
                        masked_tokens[i] = token
            masked_tokens_aug.append(masked_tokens)
            labels_aug.append(labels)
        return masked_tokens_aug, labels_aug

@registry.register_task('antibody_embed_seqIndiv')
class ABSeqIndivEmbedLanguageModelingDataset(Dataset):
    """Creates the Masked Language Modeling for antibody Dataset
       Heavy and light chain seqs are encoded individually
       Self-attention for intra-seq; across-attention for inter-seq
    Args:
        data_path (Union[str, Path]): Path to tape data root.
        split (str): One of ['train', 'valid', 'holdout', ...], specifies which data file to load.
        in_memory (bool, optional): Whether to load the full dataset into memory. Default: False.
        file_format (str): format of data file (Default: 'lmdb')
    """
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        super().__init__()
        '''
        if split not in ('train', 'valid', 'holdout'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'valid', 'holdout']")
        '''
        self.mask_stragy = kwargs.get('mlm_mask_stragy') # if not exist, return None
        #print('**mask_stragy:{}**'.format(self.mask_stragy))
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        data_path = Path(data_path)
        data_file = f"{split}.{file_format}" # 'HL_pair_{split}'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        # for i-th example, add start/end token, mask seq, convert to ids, make input_mask
        item = self.data[index]
        tokens_VH = self.tokenizer.tokenize(item['seqVH'])
        tokens_VL = self.tokenizer.tokenize(item['seqVL'])
        tokensSE_VH = self.tokenizer.add_special_tokens(tokens_VH)
        tokensSE_VL = self.tokenizer.add_special_tokens(tokens_VL)
        token_ids_VH = np.array(self.tokenizer.convert_tokens_to_ids(tokensSE_VH), np.int64)
        token_ids_VL = np.array(self.tokenizer.convert_tokens_to_ids(tokensSE_VL), np.int64)
        input_mask_VH = np.ones_like(token_ids_VH)
        input_mask_VL = np.ones_like(token_ids_VL)
        subClassH_str = item['subclassH'] if 'subclassH' in item.keys() else 'unknown'
        subClassL_str = item['subclassL'] if 'subclassL' in item.keys() else 'unknown'
        subClassH = ab_H_subclass[subClassH_str]
        subClassL = ab_L_subclass[subClassL_str]
        subClassHLPair = ab_HL_subclass['{}-{}'.format(subClassH_str,subClassL_str)]
        return token_ids_VH, token_ids_VL,\
               input_mask_VH, input_mask_VL,\
               subClassH, subClassL, subClassHLPair,\
               item['entityH'], item['entityL']

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids_VH, input_ids_VL, input_mask_VH, input_mask_VL, subClassH, subClassL, subClassHLPair, entityH, entityL = tuple(zip(*batch))

        input_ids_VH = torch.from_numpy(pad_sequences(input_ids_VH, 0))
        input_ids_VL = torch.from_numpy(pad_sequences(input_ids_VL, 0))
        input_mask_VH = torch.from_numpy(pad_sequences(input_mask_VH, 0))
        input_mask_VL = torch.from_numpy(pad_sequences(input_mask_VL, 0))
        
        subClassH = torch.LongTensor(np.array(subClassH))  # type: ignore
        subClassL = torch.LongTensor(np.array(subClassL))  # type: ignore
        subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
        return {'input_ids_VH': input_ids_VH,
                'input_ids_VL': input_ids_VL,
                'input_mask_VH': input_mask_VH,
                'input_mask_VL': input_mask_VL,
                'subClassH': subClassH,
                'subClassL': subClassL,
                'subClassHLPair': subClassHLPair,
                'entityH': entityH,
                'entityL': entityL}

@registry.register_task('antibody_mutation_MLM')
class ABMutLikelihoodDataset(Dataset):
    def __init__(self,
                 data_path: Union[str, Path]=None,
                 split: str=None,
                 tokenizer: Union[str, BaseTokenizer] = 'pfam',
                 in_memory: bool = True,
                 file_format: str = 'lmdb',
                 **kwargs):
        if isinstance(tokenizer, str):
            tokenizer = BaseTokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.concate_stragy = kwargs.get('concate_stragy',None)
        df_object = kwargs.get('df_object',None)
        assert self.concate_stragy is not None
        assert df_object is not None
        self.data = dataset_factory(None,in_memr_type='dataframe',df_object=df_object)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        
        tokens_wt_VH = self.tokenizer.tokenize(item['stateSeqVH'])
        tokens_wt_VL = self.tokenizer.tokenize(item['stateSeqVL'])
        tokens_mut_VH = self.tokenizer.tokenize(item['mutSeqVH'])
        tokens_mut_VL = self.tokenizer.tokenize(item['mutSeqVL'])
        mut_relative_idxs_VH = np.sort(item['mutRelaIdxVH'])
        mut_relative_idxs_VL = np.sort(item['mutRelaIdxVL'])

        # extract mutation name string
        mut_VH, mut_VL = [], []
        for mut_idx in mut_relative_idxs_VH:
            mut_VH.append(f'{tokens_wt_VH[mut_idx]}{mut_idx}{tokens_mut_VH[mut_idx]}')
        for mut_idx in mut_relative_idxs_VL:
            mut_VL.append(f'{tokens_wt_VL[mut_idx]}{mut_idx}{tokens_mut_VL[mut_idx]}')
        mut_VH_str, mut_VL_str = ':'.join(mut_VH), ':'.join(mut_VL)
        mut_name_str = f'H-{mut_VH_str}&L-{mut_VL_str}'

        masked_tokens_VH, labels_wt_VH, labels_mut_VH = self._apply_mut_mask(tokens_wt_VH,tokens_mut_VH,mut_relative_idxs_VH)
        masked_tokens_VL, labels_wt_VL, labels_mut_VL = self._apply_mut_mask(tokens_wt_VL,tokens_mut_VL,mut_relative_idxs_VL)

        if self.concate_stragy == 'seqConcate':
            masked_tokens = masked_tokens_VH + ['<sep>'] + masked_tokens_VL
            masked_tokens = self.tokenizer.add_special_tokens(masked_tokens)
            labels_wt = np.concatenate(([-1],labels_wt_VH,[-1],labels_wt_VL,[-1]))
            labels_mut = np.concatenate(([-1],labels_mut_VH,[-1],labels_mut_VL,[-1]))
            masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64) 
            input_mask = np.ones_like(masked_token_ids)
            
            ## token_type_ids(segment_id)
            ## <cls> VH <sep> VL <sep> <pad> ...
            ##   0  {0}   0  {1}   1     1 ...
            aug_size = 1 if len(masked_token_ids.shape) == 1 else masked_token_ids.shape[0]
            token_type_ids = np.array([[0] + [0]*len(masked_tokens_VH) + [0] + [1]*len(masked_tokens_VL) + [1]]*aug_size, np.int64).squeeze()
            subClassH = np.array([ab_H_subclass[item['subclassH']]]*aug_size).squeeze() if item['subclassH'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
            subClassL = np.array([ab_L_subclass[item['subclassL']]]*aug_size).squeeze() if item['subclassL'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
            if item['subclassH'].lower() != 'unknown' and item['subclassL'].lower() != 'unknown':
                subClassHLPair = np.array([ab_HL_subclass['{}-{}'.format(item['subclassH'],item['subclassL'])]]*aug_size).squeeze()
            else:
                subClassHLPair = np.array([-1]*aug_size).squeeze()
            return masked_token_ids, input_mask, labels_wt, labels_mut, token_type_ids, subClassH, subClassL, subClassHLPair, mut_name_str, item['seq_id']
        elif self.concate_stragy == 'seqIndiv':
            masked_tokens_VH = self.tokenizer.add_special_tokens(masked_tokens_VH)
            masked_tokens_VL = self.tokenizer.add_special_tokens(masked_tokens_VL)
            labels_wt_VH = np.concatenate(([-1],labels_wt_VH,[-1]))
            labels_wt_VL = np.concatenate(([-1],labels_wt_VL,[-1]))
            labels_mut_VH = np.concatenate(([-1],labels_mut_VH,[-1]))
            labels_mut_VL = np.concatenate(([-1],labels_mut_VL,[-1]))
            masked_token_ids_VH = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens_VH), np.int64)
            masked_token_ids_VL = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens_VL), np.int64)
            input_mask_VH = np.ones_like(masked_token_ids_VH)
            input_mask_VL = np.ones_like(masked_token_ids_VL) 
            ## token_type_ids(segment_id)
            ## <cls> VH <sep> VL <sep> <pad> ...
            ##   0  {0}   0  {1}   1     1 ...
            aug_size = 1 if len(masked_token_ids_VH.shape) == 1 else masked_token_ids_VH.shape[0]
            subClassH = np.array([ab_H_subclass[item['subclassH']]]*aug_size).squeeze() if item['subclassH'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
            subClassL = np.array([ab_L_subclass[item['subclassL']]]*aug_size).squeeze() if item['subclassL'].lower() != 'unknown' else np.array([-1]*aug_size).squeeze()
            if item['subclassH'].lower() != 'unknown' and item['subclassL'].lower() != 'unknown':
                subClassHLPair = np.array([ab_HL_subclass['{}-{}'.format(item['subclassH'],item['subclassL'])]]*aug_size).squeeze()
            else:
                subClassHLPair = np.array([-1]*aug_size).squeeze()
            return masked_token_ids_VH, masked_token_ids_VL, input_mask_VH, input_mask_VL, labels_wt_VH, labels_wt_VL, labels_mut_VH, labels_mut_VL, subClassH, subClassL, subClassHLPair, mut_name_str, item['seq_id']
        else:
            raise ValueError(f'invalid concate_stragy: {self.concate_stragy}')

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Dict[str, torch.Tensor]:
        if self.concate_stragy == 'seqConcate':
            masked_token_ids, input_mask, labels_wt, labels_mut, token_type_ids, subClassH, subClassL, subClassHLPair, mut_name_str, seq_id = tuple(zip(*batch))
            input_ids = torch.from_numpy(pad_sequences(masked_token_ids, 0))
            input_mask = torch.from_numpy(pad_sequences(input_mask, 0))
            token_type_ids = torch.from_numpy(pad_sequences(token_type_ids, 1))
            # ignore_index is -1
            labels_wt = torch.from_numpy(pad_sequences(labels_wt, -1))
            labels_mut = torch.from_numpy(pad_sequences(labels_mut, -1))
            subClassH = torch.LongTensor(np.array(subClassH))
            subClassL = torch.LongTensor(np.array(subClassL))
            subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'targets': labels_wt,
                    'targets_mut': labels_mut,
                    'token_type_ids': token_type_ids,
                    'subClassH': subClassH,
                    'subClassL': subClassL,
                    'subClassHLPair': subClassHLPair,
                    'mut_names': mut_name_str,
                    'seq_id': seq_id}
        elif self.concate_stragy == 'seqIndiv':
            masked_token_ids_VH, masked_token_ids_VL, input_mask_VH, input_mask_VL, labels_wt_VH, labels_wt_VL, labels_mut_VH, labels_mut_VL, subClassH, subClassL, subClassHLPair, mut_name_str, seq_id = tuple(zip(*batch))
            input_ids_VH = torch.from_numpy(pad_sequences(masked_token_ids_VH, 0))
            input_ids_VL = torch.from_numpy(pad_sequences(masked_token_ids_VL, 0))
            input_mask_VH = torch.from_numpy(pad_sequences(input_mask_VH, 0))
            input_mask_VL = torch.from_numpy(pad_sequences(input_mask_VL, 0))
            # ignore_index is -1
            labels_wt_VH = torch.from_numpy(pad_sequences(labels_wt_VH, -1))
            labels_wt_VL = torch.from_numpy(pad_sequences(labels_wt_VL, -1))
            labels_mut_VH = torch.from_numpy(pad_sequences(labels_mut_VH, -1))
            labels_mut_VL = torch.from_numpy(pad_sequences(labels_mut_VL, -1))
            subClassH = torch.LongTensor(np.array(subClassH))  # type: ignore
            subClassL = torch.LongTensor(np.array(subClassL))  # type: ignore
            subClassHLPair = torch.LongTensor(np.array(subClassHLPair))
            return {'input_ids_VH': input_ids_VH,
                    'input_ids_VL': input_ids_VL,
                    'input_mask_VH': input_mask_VH,
                    'input_mask_VL': input_mask_VL,
                    'targets_VH': labels_wt_VH,
                    'targets_VL': labels_wt_VL,
                    'targets_mut_VH': labels_mut_VH,
                    'targets_mut_VL': labels_mut_VL,
                    'subClassH': subClassH,
                    'subClassL': subClassL,
                    'subClassHLPair': subClassHLPair,
                    'mut_names': mut_name_str,
                    'seq_ids': seq_id}
        else:
            raise ValueError(f'invalid concate_stragy: {self.concate_stragy}')

    def _apply_mut_mask(self, 
                        tokens_wt: List[str], 
                        tokens_mut: List[str],
                        rela_pos_list: List[int]):
        masked_tokens = copy(tokens_wt)
        assert len(tokens_wt) == len(tokens_mut)
        labels_wt = np.zeros([len(tokens_wt)], np.int64) - 1
        labels_mut = np.zeros([len(tokens_mut)], np.int64) - 1
        ## pos_list, wtSeq_startIdx start from 1
        for i in rela_pos_list:
            labels_wt[i] = self.tokenizer.convert_token_to_id(tokens_wt[i])
            labels_mut[i] = self.tokenizer.convert_token_to_id(tokens_mut[i])
            masked_tokens[i] = self.tokenizer.mask_token     
        return masked_tokens, labels_wt, labels_mut
