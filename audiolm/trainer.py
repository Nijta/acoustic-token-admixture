from __future__ import annotations

import re
import copy
from math import sqrt
from datetime import timedelta
from random import choice
from pathlib import Path
from shutil import rmtree
from functools import partial
from collections import Counter
from contextlib import contextmanager, nullcontext

from beartype.typing import Type
import numpy as np
from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

import torch
import torchaudio
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

import pytorch_warmup as warmup

from einops import rearrange

from optimizer import get_optimizer

import wandb
from ema_pytorch import EMA

from audiolm_pytorch import (
    CoarseTransformer,
    CoarseTransformerWrapper,
)

from data import CustomDataset, get_dataloader

from version import __version__
from packaging import version

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.tracking import WandBTracker

import soundfile as sf

# constants

DEFAULT_SAMPLE_RATE = 16000

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

# make sure only one trainer is instantiated

ONE_TRAINER_INSTANTIATED = False

def check_one_trainer():
    global ONE_TRAINER_INSTANTIATED
    assert not ONE_TRAINER_INSTANTIATED, 'only one Trainer can be instantiated at a time for training'
    ONE_TRAINER_INSTANTIATED = True

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters = True)

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
    raw_wave = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
    ],
    text = list[str],
    text_embeds = Annotated[
        torch.Tensor,
        Is[lambda t: t.dtype == torch.float and t.ndim == 3]
    ],
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    return True
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def dict_values_to_device(d: dict, device):
    out = {}
    for k, v in d.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict(Counter(tup))
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/semantic.transformer.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])

# optimizer with scheduler + warmup

class OptimizerWithWarmupSchedule(nn.Module):
    @beartype
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        warmup_steps: int = 0
    ):
        super().__init__()
        self.warmup = warmup.LinearWarmup(optimizer, warmup_period = warmup_steps)

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = ConstantLRScheduler(optimizer)

        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator

    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            warmup = self.warmup.state_dict()
        )

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.scheduler.load_state_dict(pkg['scheduler'])
        self.warmup.load_state_dict(pkg['warmup'])

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            with self.warmup.dampening():
                self.scheduler.step()


def count_tail(row):
    # Find the index of the last non-zero element
    non_zero_idx = (row != 0).nonzero(as_tuple=True)[0]
    
    if non_zero_idx.size(0) == 0:
        # No non-zero elements in the row, return 0
        return 0
    
    # Get the last non-zero element's index
    first_zero_index = non_zero_idx[-1]+1
    
    return first_zero_index

# main trainer class

# semantic transformer trainer


class CoarseTransformerTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        transformer: CoarseTransformer,
        codec,
        speaker,
        # wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None,
        *,
        num_train_steps,
        batch_size,
        audio_conditioner = None,
        dataset: Dataset | None = None,
        valid_dataset: Dataset | None = None,
        ds_fields: tuple[str, ...] = ('artics_features', 'whisper_tokens', 'f0_tokens'),
        data_max_length = None,
        data_max_length_seconds = None,
        data_path = None,
        lr = 3e-4,
        grad_accum_every = 1,
        wd = 0.,
        max_grad_norm = 0.5,
        valid_frac = 0.05,
        random_split_seed = 42,
        save_results_every = 100,
        save_model_every = 1000,
        results_folder = './results',
        accelerate_kwargs: dict = dict(),
        init_process_group_timeout_seconds = 1800,
        split_batches = False,
        drop_last = False,
        force_clear_prev_results = None,
        use_wandb_tracking = False,
        average_valid_loss_over_grad_accum_every: bool = True,  # if False, valid loss on a single batch
    ):
        super().__init__()
        check_one_trainer()
        self.use_wandb_tracking = use_wandb_tracking
        if use_wandb_tracking:
            accelerate_kwargs.update(log_with = 'wandb')
        init_process_kwargs = InitProcessGroupKwargs(timeout = timedelta(seconds = init_process_group_timeout_seconds))

        self.accelerator = Accelerator(
            kwargs_handlers = [DEFAULT_DDP_KWARGS, init_process_kwargs],
            split_batches = split_batches,
            **accelerate_kwargs
        )

        self.transformer = transformer
        self.codec = codec
        self.speaker = speaker
        # self.wav2vec = wav2vec
        self.audio_conditioner = audio_conditioner

        self.train_wrapper = CoarseTransformerWrapper(
            codec = codec,
            # wav2vec = wav2vec,
            transformer = transformer,
            audio_conditioner = audio_conditioner
        )

        self.register_buffer('steps', torch.tensor(0))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every

        # optimizers

        self.optim = get_optimizer(transformer.parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm

        # create dataset

        self.ds = dataset

        if not exists(self.ds):
            assert exists(data_path), 'folder must be passed in, if not passing in a custom dataset for text conditioned audio synthesis training'

            assert not (exists(data_max_length) and exists(data_max_length_seconds))

            self.ds = CustomDataset(
                data_path,
                max_length = data_max_length,
            )

        self.ds_fields = ds_fields

        # split for validation

        self.valid_ds = valid_dataset

        if not exists(self.valid_ds):
            if valid_frac > 0:
                train_size = int((1 - valid_frac) * len(self.ds))
                valid_size = len(self.ds) - train_size
                self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
                self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
            else:
                self.valid_ds = self.ds
                self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        assert len(self.ds) >= batch_size, 'dataset must have sufficient samples for training'
        assert len(self.valid_ds) >= batch_size, f'validation dataset must have sufficient number of samples (currently {len(self.valid_ds)}) for training'

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, shuffle = True, drop_last = drop_last)

        # prepare with accelerator

        (
            self.train_wrapper,
            self.optim,
            self.dl
        ) = self.accelerator.prepare(
            self.train_wrapper,
            self.optim,
            self.dl
        )

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        # if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
        #     rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
        self.tracker_hps = hps

        self.accelerator.init_trackers("coarse", config=hps)

        self.train_wrapper.to(self.device)
        self.average_valid_loss_over_grad_accum_every = average_valid_loss_over_grad_accum_every

        self.f0_quantize = torch.from_numpy(np.asarray([10.0, 100.89050217, 115.1907547 , 127.34055481, 142.26394958, 161.6648941 , 180.98721313, 199.32779541, 221.38771667, 252.21411438, 803.56750488])).to("cuda")

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.transformer),
            optim = self.optim.state_dict(),
            version = __version__
        )
        torch.save(pkg, path)

    def load(self, path):
        transformer = self.accelerator.unwrap_model(self.transformer)
        pkg = transformer.load(path)
        # trainer-specific things
        self.optim.load_state_dict(pkg['optim'])

        # + 1 to start from the next step and avoid overwriting the last checkpoint
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.device)

    def print(self, msg):
        self.accelerator.print(msg)

    def generate(self, *args, **kwargs):
        return self.train_wrapper.generate(*args, **kwargs)

    @contextmanager
    def wandb_tracker(self, project, run = None, hps = None):
        assert self.use_wandb_tracking, '`use_wandb_tracking` must be set to True on CoarseTransformerTrainer'

        hps = default(hps, self.tracker_hps)

        self.accelerator.init_trackers(project, config = None)

        if exists(run):
            wandb_tracker = find_first(lambda el: isinstance(el, WandBTracker), self.accelerator.trackers)
            assert exists(wandb_tracker)

            wandb_tracker.run.name = run

        yield

        self.accelerator.end_training()  

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.transformer.train()

        # logs

        logs = {}

        # update transformer
        x = None
        for i in range(self.grad_accum_every):
            is_last = i == (self.grad_accum_every - 1)
            context = partial(self.accelerator.no_sync, self.train_wrapper) if not is_last else nullcontext
            # try:
            x = next(self.dl_iter)
            # except:
                # pass
            # data_kwargs = dict(zip(self.ds_fields, next(self.dl_iter)))
            data_kwargs = dict(zip(self.ds_fields, x))
            # data_kwargs['f0_tokens'] = F.interpolate(data_kwargs['f0_tokens'].unsqueeze(1), size=data_kwargs['artics_features'].size(1), mode='linear', align_corners=True).squeeze(1)
            # data_kwargs['f0_tokens'] = torch.bucketize(data_kwargs['f0_tokens'], self.f0_quantize, right=True)
            del data_kwargs['f0_tokens']
            # breakpoint()
            with self.accelerator.autocast(), context():
                loss = self.train_wrapper(
                    **data_kwargs,
                    return_loss = True
                )

                self.accelerator.backward(loss / self.grad_accum_every)

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # log

        self.print(f"{steps}: loss: {logs['loss']}")
        self.accelerator.log({"train_loss": logs['loss']}, step=steps)

        # sample results every so often

        self.accelerator.wait_for_everyone()


        if self.is_main and steps > 5 and not (steps % self.save_results_every):
            data_kwargs = dict(zip(self.ds_fields, next(self.valid_dl_iter)))
            gen_wavs = self.train_wrapper.generate(artics_features = data_kwargs['artics_features'], reconstruct_wave = True, f0_tokens=data_kwargs['f0_tokens'])
            real_wavs = []
            codes = rearrange(data_kwargs['whisper_tokens'].long(), 'b t q -> q b t')
            codes_sum = torch.sum(codes, dim=0)
            tail_lengths = torch.tensor([count_tail(row) for row in codes_sum])
            # breakpoint()
            embs = [torch.tensor(0.0, device=codes.device) for _ in range(data_kwargs['f0_tokens'].shape[0])]
            for i, indices in enumerate(codes):
                layer = self.codec.rq.layers[i]
                for j in range(indices.shape[0]):
                    this_indices = indices[j][indices[j] >= 0][:tail_lengths[j]]
                    this_quantized = layer.get_codes_from_indices(this_indices)
                    embs[j] = embs[j] + this_quantized
            for ind in range(data_kwargs['f0_tokens'].shape[0]):
                new_pitch = self.speaker.convert_pitch(data_kwargs['f0_tokens'][data_kwargs['f0_tokens'] >= 0].detach().cpu().numpy())
                # breakpoint()
                wav = self.codec.BW.synthesize(embs[ind].detach().cpu().numpy(), self.speaker.xvector, new_pitch, chunk_size = 100)
                # wav = self.BW.synthesize(emb[ind].detach().cpu().numpy(), self.xvector, np.asarray([x for x in f0_tokens[ind].detach().cpu().numpy() if x >= 0]), chunk_size = 50)
                real_wavs.append(wav[0])
            
            for ind in range(data_kwargs['f0_tokens'].shape[0]):
                sf.write(f'training_samples/step_{steps}_sample_{ind}_gen.wav', gen_wavs[ind], 16000)
                sf.write(f'training_samples/step_{steps}_sample_{ind}_real.wav', real_wavs[ind], 16000)
            # breakpoint()

        if self.is_main and not (steps % self.save_results_every):
            valid_loss = 0
            unwrapped_model = self.accelerator.unwrap_model(self.train_wrapper)

            for i in range(self.average_valid_loss_over_grad_accum_every):
                data_kwargs = dict(zip(self.ds_fields, next(self.valid_dl_iter)))
                data_kwargs = dict_values_to_device(data_kwargs, unwrapped_model.device)
                del data_kwargs['f0_tokens']
                # breakpoint()
                with torch.no_grad():
                    unwrapped_model.eval()

                    valid_loss += unwrapped_model(
                        **data_kwargs,
                        return_loss = True
                    )

            valid_loss = valid_loss.clone() # avoid inference mode to non-inference mode error
            valid_loss /= self.average_valid_loss_over_grad_accum_every

            self.print(f'{steps}: valid loss {valid_loss}')
            self.accelerator.log({"valid_loss": valid_loss}, step=steps)

        # save model every so often

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'coarse.transformer.{steps}.pt')
            self.save(model_path)
            if self.use_wandb_tracking:
                wandb.save(model_path)
            self.print(f'{steps}: saving model to {str(self.results_folder)}')

        self.accelerator.wait_for_everyone()

        self.steps.add_(1)
        return logs

    def train(self, log_fn = noop):

        while self.steps < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

        self.print('training complete')

