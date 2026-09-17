from __future__ import annotations

import os
from functools import wraps

from beartype import beartype

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


# helper functions

def exists(val):
    return val is not None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def is_unique(arr):
    return len(set(arr)) == len(arr)

# dataset functions
def read_raw_mat(filename,col,format='f4',end='l'):
	"""read_raw_mat(filename,col,format='float',end='l')
	   Read the binary data from filename
	   Return data, which is a (N, col) array
	   
	   filename: the name of the file, take care about '\\'
	   col:	  the number of column of the data
	   format:   please use the Python protocal to write format
				 default: 'f4', float32
				 see for more format:
	   end:	  little endian 'l' or big endian 'b'?
				 default: 'l'
	   
	   dependency: numpy
	   Note: to read the raw binary data in python, the question
			 is how to interprete the binary data. We can use
			 struct.unpack('f',read_data) to interprete the data
			 as float, however, it is slow.
	"""
	f = open(filename,'rb')
	if end=='l':
		format = '<'+format
	elif end=='b':
		format = '>'+format
	else:
		format = '='+format
	datatype = np.dtype((format,(col,)))
	data = np.fromfile(f,dtype=datatype)
	f.close()
	if data.ndim == 2 and data.shape[1] == 1:
		return data[:,0]
	else:
		return data
     
class CustomDataset(Dataset):
    @beartype
    def __init__(
        self,
        data_path,
        max_length: int | None = None,
    ):
        super().__init__()
        self.data_path = data_path
        assert os.path.exists(data_path), f'data path "{data_path}" does not exist'
        # list_samples = os.listdir(os.path.join(data_path, "semantics"))
        list_samples = os.listdir(os.path.join(data_path, "artics"))
        assert len(list_samples) > 0, 'no samples found in dataset'

        self.files = list_samples

        self.max_length = max_length
        # self.whisper_max = int(self.max_length/1.25)+1
        self.whisper_max = self.max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        full_path_whisper = os.path.join(self.data_path, "Whisper_ppg", file.replace(".npy", ".ppg"))
        full_path_f0 = os.path.join(self.data_path, "F0", file.replace(".npy", ".f0"))
        # full_path_artics = os.path.join(self.data_path, "semantics", file)
        full_path_artics = os.path.join(self.data_path, "artics", file)
        artic_features = np.load(full_path_artics) # (t, 64)
        whisper_codecs = read_raw_mat(full_path_whisper, 8) # (t, 8)
        f0_codecs = read_raw_mat(full_path_f0, 1) # (t, 1)
        # breakpoint()
        if artic_features.shape[0] > self.max_length:
            #  start_idx = random.randint(0, artic_features.shape[0]-self.max_length)
             start_idx = 0
             artic_features = artic_features[start_idx:(start_idx+self.max_length),...]
            #  f0_max = int(self.whisper_max/whisper_codecs.shape[0]*f0_codecs.shape[0])
             f0_max = int(self.whisper_max/whisper_codecs.shape[0]*f0_codecs.shape[0])
             f0_start = int(start_idx/whisper_codecs.shape[0]*f0_codecs.shape[0])
             f0_codecs = f0_codecs[f0_start:(f0_start + f0_max),...]
             whisper_codecs = whisper_codecs[start_idx:(start_idx + self.whisper_max),...]
        # breakpoint()
        return artic_features, whisper_codecs, f0_codecs
    

# dataloader functions

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        # breakpoint()
        semantic_tokens = fn([torch.from_numpy(x[0]) for x in data])
        whisper_tokens = fn([torch.from_numpy(x[1]) for x in data])
        f0_tokens = fn([torch.from_numpy(x[2]) for x in data])
        return (semantic_tokens.float(), whisper_tokens.long(), f0_tokens)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    # return pad_sequence(data, batch_first = True, padding_value=-1)
    return pad_sequence(data, batch_first = True)

def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
