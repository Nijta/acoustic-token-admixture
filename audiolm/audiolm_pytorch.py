from __future__ import annotations

import math
from functools import partial, wraps

from beartype import beartype

import torch
from torch import nn, einsum, Tensor
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import torchaudio

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# from vq_wav2vec import FairseqVQWav2Vec
# from hubert_kmeans import HubertWithKmeans
from fairseq_spe import SinusoidalPositionalEmbedding


# from t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

from torchaudio.functional import resample

from utils import AudioConditionerBase
from attend import Attend
from tqdm import tqdm
from pathlib import Path
from version import __version__
from packaging import version

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def maybe(fn):
    if not exists(fn):
        return always(None)

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def ceil_div(numer, denom):
    return (numer + denom - 1) // denom

def remainder_needed_until_multiple(n, mult):
    return (ceil_div(n, mult) * mult) - n

def round_down_nearest_multiple(val, mult):
    return (val // mult) * mult

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# tensor helpers

def generate_mask_with_prob(shape, mask_prob, device):
    seq = shape[-1]
    rand = torch.randn(shape, device = device)
    rand[:, 0] = -torch.finfo(rand.dtype).max
    num_mask = min(int(seq * mask_prob), seq - 1)
    indices = rand.topk(num_mask, dim = -1).indices
    mask = ~torch.zeros(shape, device = device).scatter(1, indices, 1.).bool()
    return mask

# attention related utils

def grad_shrink(t, alpha = 0.1):
    return t * alpha + t.detach() * (1 - alpha)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    return F.normalize(t, dim = -1)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def mask_out_after_eos_id(t, eos_id, mask_value = -1, keep_eos = True):
    eos_mask = (t == eos_id).float()

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim = -1) > 0
    return t.masked_fill(after_eos_mask, mask_value)

def all_rows_have_eos_id(t, eos_id):
    eos_mask = (t == eos_id)
    return torch.any(eos_mask, dim = -1).all()

def safe_cat(*tensors, dim = -2):
    args = [*filter(exists, tensors)]

    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        return torch.cat(args, dim = dim)

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# removing unique consecutives in the semantic token ids
# important detail noted by @eonglints

def count_tail(row):
    # Find the index of the last non-zero element
    non_zero_idx = (row != 0).nonzero(as_tuple=True)[0]
    
    if non_zero_idx.size(0) == 0:
        # No non-zero elements in the row, return 0
        return 0
    
    # Get the last non-zero element's index
    first_zero_index = non_zero_idx[-1]+1
    
    return first_zero_index

def append_eos_id(ids, eos_id):
    b, device = ids.shape[0], ids.device
    if type(eos_id) == int:
        eos_ids = torch.ones(1, device = device).long() * eos_id
        zero_ids = torch.zeros(1, device = device).long()
        zero_ids = repeat(zero_ids, 'k -> b k', b = b)
        sum_ids = ids.clone()
    else:
        eos_ids = torch.ones_like(eos_id, device = device).long() * eos_id
        zero_ids = torch.zeros_like(eos_id, device = device).long()
        zero_ids = repeat(zero_ids, 'k -> b 1 k', b = b)
        sum_ids = torch.sum(ids, dim=2)

    ids = torch.cat((ids, zero_ids), dim = 1)

    tail_lengths = torch.tensor([count_tail(row) for row in sum_ids])
    for idx, ll in enumerate(tail_lengths):
        ids[idx, ll] = eos_ids

    # breakpoint()

    return ids

def batch_unique_consecutive(t, pad_value = 0.):
    unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
    return pad_sequence(unique_arr, batch_first = True, padding_value = pad_value)

# function for getting embeds from nn.Embedding but with padding as some designated value (-1) outside the range of the embed table

@beartype
def get_embeds(
    embeddings: nn.Embedding,
    codes: torch.Tensor,
    pad_id = -1,
    return_mask = False,
    mask_pad_pos_to = 0
):
    pad_mask = codes == pad_id
    codes_without_pad = codes.masked_fill(pad_mask, 0) # just retrieve first code as dummy
    embeds = embeddings(codes_without_pad)

    if exists(mask_pad_pos_to):
        embeds = embeds.masked_fill(rearrange(pad_mask, '... -> ... 1'), mask_pad_pos_to)

    if return_mask:
        return embeds, ~pad_mask

    return embeds

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# relative positional bias

class RelativePositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        layers = 3
    ):
        super().__init__()
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(1, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert j >= i
        device = self.device

        i_pos = torch.arange(i, device = device) + (j - i)
        j_pos = torch.arange(j, device = device)

        rel_pos = (rearrange(i_pos, 'i -> i 1') - rearrange(j_pos, 'j -> 1 j'))
        rel_pos += (j - 1)

        x = torch.arange(-j + 1, j, device = device).float()
        x = rearrange(x, '... -> ... 1')

        for layer in self.net:
            x = layer(x)

        x = x[rel_pos]
        return rearrange(x, 'i j h -> h i j')

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.1):
    inner_dim = int(dim * 2 * mult / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        num_null_kv = 0,
        dropout = 0.1,
        scale = 8,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(2, num_null_kv, dim_head)) if num_null_kv > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias = False)

        self.attend = Attend(
            flash = flash,
            dropout = dropout,
            causal = causal
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        prefix_context = None,
        prefix_context_mask = None,
        return_kv_cache = False,
        return_values = False,
        value_residual: Tensor | None = None,
        kv_cache = None
    ):
        b, n, _, device = *x.shape, x.device

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        # take care of prefix-based self attention conditioning
        # make sure to either concat the to the self attention mask or lengthen it accordingly

        if exists(prefix_context):
            kv_input = torch.cat((prefix_context, kv_input), dim = -2)
            prefix_seq_len = prefix_context.shape[-2]

            if not exists(mask):
                mask = torch.ones((b, n), device = device, dtype = torch.bool)

            if exists(prefix_context_mask):
                mask = torch.cat((prefix_context_mask, mask), dim = -1)
            else:
                mask = F.pad(mask, (prefix_seq_len, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (prefix_seq_len, 0), value = 0.)

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        # for value residual learning

        orig_v = v

        if exists(value_residual):
            v = 0.5 * (v + value_residual)

        # kv cache

        if exists(kv_cache):
            ck, cv = kv_cache
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # store kv cache

        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # null key / values

        if self.num_null_kv > 0:
            null_k, null_v = repeat(self.null_kv, 'kv n d -> kv b n d', b = b).unbind(dim = 0)
            k = torch.cat((null_k, k), dim = -2)
            v = torch.cat((null_v, v), dim = -2)

        # split for multi-headed attention

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # handle mask and null key / value

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)

        # attention

        out = self.attend(q, k, v, attn_bias = attn_bias, mask = mask)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_kv_cache and not return_values:
            return out

        if return_kv_cache and not return_values:
            return out, kv_cache

        if return_values and not return_kv_cache:
            return out, orig_v

        return out, (kv_cache, orig_v)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_context = None,
        cross_attend = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        grad_shrink_alpha = 0.1,
        cond_as_self_attn_prefix = False,
        rel_pos_bias = True,
        flash_attn = False,
        add_value_residual = True,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        assert not (cross_attend and cond_as_self_attn_prefix)

        self.dim_context = default(dim_context, dim)

        self.cond_as_self_attn_prefix = cond_as_self_attn_prefix

        self.grad_shrink = partial(grad_shrink, alpha = grad_shrink_alpha)

        self.layers = nn.ModuleList([])

        self.rel_pos_bias = RelativePositionBias(dim = dim // 2, heads = heads) if rel_pos_bias else None

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dropout = attn_dropout, flash = flash_attn, causal = True, **kwargs),
                Attention(dim = dim, heads = heads, dropout = attn_dropout, dim_context = dim_context, flash = flash_attn, num_null_kv = 1, norm_context = True, **kwargs) if cross_attend else None,
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

        self.norm = LayerNorm(dim)

        self.add_value_residual = add_value_residual

    def forward(
        self,
        x,
        self_attn_mask = None,
        context = None,
        context_mask = None,
        attn_bias = None,
        return_kv_cache = False,
        kv_cache = None
    ):
        assert not (self.cond_as_self_attn_prefix and not exists(context))
        assert not (exists(context) and context.shape[-1] != self.dim_context), f'you had specified a conditioning dimension of {self.dim_context}, yet what was received by the transformer has dimension of {context.shape[-1]}'

        n, device = x.shape[1], x.device

        # from cogview paper, adopted by GLM 130B LLM, decreases likelihood of attention net instability

        x = self.grad_shrink(x)

        # turn off kv cache if using conditioning as self attention (as in valle), for now

        if self.cond_as_self_attn_prefix:
            kv_cache = None

        # handle kv cache

        new_kv_cache = []

        if exists(kv_cache):
            cache_len = kv_cache.shape[-2]
            kv_cache = iter(kv_cache)
        else:
            cache_len = 0
            kv_cache = iter([])

        x = x[:, cache_len:]
        # relative positional bias

        if exists(attn_bias):
            rel_pos_bias = attn_bias
        else:
            rel_pos_bias = maybe(self.rel_pos_bias)(n, n)

        if exists(rel_pos_bias):
            rel_pos_bias = rel_pos_bias[..., cache_len:, :]

        # self attention kwargs

        self_attn_kwargs = dict()
        if self.cond_as_self_attn_prefix:
            self_attn_kwargs = dict(
                prefix_context = context,
                prefix_context_mask = context_mask
            )

        # value residuals

        self_attn_value_residual = None
        cross_attn_value_residual = None

        # transformer layers
        for attn, cross_attn, ff in self.layers:

            residual = x

            x, (layer_kv_cache, values) = attn(x, attn_bias = rel_pos_bias, mask = self_attn_mask, kv_cache = next(kv_cache, None), return_kv_cache = True, return_values = True, value_residual = self_attn_value_residual, **self_attn_kwargs)

            if self.add_value_residual:
                self_attn_value_residual = default(self_attn_value_residual, values)

            new_kv_cache.append(layer_kv_cache)

            x = x + residual

            if exists(cross_attn):
                assert exists(context)

                cross_attend_out, values = cross_attn(x, context = context, mask = context_mask, return_values = True, value_residual = cross_attn_value_residual)
                x = cross_attend_out + x

                if self.add_value_residual:
                    cross_attn_value_residual = default(cross_attn_value_residual, values)

            x = ff(x) + x

        x = self.norm(x)

        if not return_kv_cache:
            return x

        return x, torch.stack(new_kv_cache)

# the three hierarchical transformers

def apply_positional_embeddings(semantic_tokens, coarse_tokens, SPE, num_q = 8, scaling_factor = 1.0):
    # breakpoint()
    semantic_pos_embeddings = SPE(torch.ones(size=[semantic_tokens.shape[0], 1024]).to("cuda"))
    # coarse_pos_embeddings = SPE(torch.ones(size=[semantic_tokens.shape[0], 514]).to("cuda"))
    # coarse_seq_len = int((1024)/1.25)+3
    if scaling_factor != 1.0:
        coarse_seq_len = int((1024)/scaling_factor)
        coarse_pos_embeddings = F.interpolate(semantic_pos_embeddings.permute(0,2,1), size=coarse_seq_len, mode='linear', align_corners=False)
        coarse_pos_embeddings = coarse_pos_embeddings.permute(0,2,1)
    else:
        coarse_pos_embeddings = SPE(torch.ones(size=[semantic_tokens.shape[0], 1024]).to("cuda"))

    coarse_pos_embeddings = torch.repeat_interleave(coarse_pos_embeddings, repeats=num_q, dim=1)
    # Combine the tokens and their respective positional embeddings
    try:
        semantic_tokens_with_pos = semantic_tokens + semantic_pos_embeddings[:, :semantic_tokens.shape[1], :]
        coarse_start_tokens_with_pos = coarse_tokens + coarse_pos_embeddings[:, (num_q-1):coarse_tokens.shape[1]+(num_q-1), :]
    except:
        breakpoint()

    # Concatenate both tensors
    tokens_with_pos = torch.cat((semantic_tokens_with_pos, coarse_start_tokens_with_pos), dim=1)

    return tokens_with_pos

class CoarseTransformer(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        codebook_size,
        num_coarse_quantizers,
        dim,
        depth,
        num_semantic_tokens,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        # t5_name = DEFAULT_T5_NAME,
        has_condition = False,
        cond_dim = None,
        audio_text_condition = False,
        cond_as_self_attn_prefix = False,
        cond_drop_prob = 0.5,
        grad_shrink_alpha = 0.1,
        project_semantic_logits = True,
        rel_pos_bias = True,
        flash_attn = False,
        **kwargs
    ):
        super().__init__()
        rel_pos_bias = rel_pos_bias and not flash_attn

        self.num_semantic_tokens = num_semantic_tokens

        if audio_text_condition:
            has_condition = True
            cond_dim = default(cond_dim, dim)

        self.has_condition = has_condition
        # self.embed_text = partial(t5_encode_text, name = t5_name)
        self.cond_drop_prob = cond_drop_prob

        self.semantic_start_token = nn.Parameter(torch.randn(dim))
        self.coarse_start_token = nn.Parameter(torch.randn(dim))

        # self.semantic_eos_id = num_semantic_tokens
        self.semantic_eos_id = nn.Parameter(torch.randn(64))
        
        self.semantic_embedding = nn.Embedding(num_semantic_tokens + 1, dim)
        self.proj_semantics = nn.Linear(64, dim, bias = False)

        self.coarse_eos_id = codebook_size
        codebook_size_with_eos = codebook_size + 1

        self.coarse_embedding = nn.Embedding(num_coarse_quantizers * codebook_size_with_eos, dim)
        self.coarse_quantize_embedding = nn.Embedding(num_coarse_quantizers, dim)

        self.SPE = SinusoidalPositionalEmbedding(dim, padding_idx=None)

        # text_dim = default(cond_dim, get_encoded_dim(t5_name))
        # self.proj_text_embed = nn.Linear(text_dim, dim, bias = False) if text_dim != dim else nn.Identity()

        self.cross_attn_bias = nn.Parameter(torch.zeros(heads, 1, 1)) if rel_pos_bias else None

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            cross_attend = has_condition and not cond_as_self_attn_prefix,
            cond_as_self_attn_prefix = cond_as_self_attn_prefix,
            grad_shrink_alpha = grad_shrink_alpha,
            rel_pos_bias = rel_pos_bias,
            flash_attn = flash_attn,
            **kwargs
        )

        self.codebook_size = codebook_size
        self.num_coarse_quantizers = num_coarse_quantizers

        self.to_semantic_logits = nn.Linear(dim, num_semantic_tokens + 1) if project_semantic_logits else None
        self.coarse_logit_weights = nn.Parameter(torch.randn(num_coarse_quantizers, codebook_size_with_eos, dim))

    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, path):
        # Return pkg so that if this function gets called from within a Trainer function call,
        # the trainer can also access the package loaded from the checkpoint.
        device = self.device
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = device)
        # check version
        if 'version' in pkg and version.parse(pkg['version']) < version.parse(__version__):
            print(f'model was trained on older version {pkg["version"]} of audiolm-pytorch')
        
        model_state = self.state_dict()
        filtered_state = {
            k: v for k, v in pkg['model'].items()
            if k in model_state and model_state[k].shape == v.shape
        }
        self.load_state_dict(filtered_state, strict = False)
        return pkg

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3,
        return_kv_cache = False,
        kv_cache = None,
        embed_cache = None,
        **kwargs
    ):
        iter_kv_cache = iter(default(kv_cache, []))
        iter_embed_cache = iter(default(embed_cache, []))
        new_kv_caches = []
        new_embed_caches = []

        (semantic_logits, coarse_logits), (new_kv_cache, new_embed_cache) = self.forward(*args, cond_drop_prob = 0., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(new_kv_cache)
        new_embed_caches.append(new_embed_cache)

        if cond_scale == 1 or not self.has_condition:
            if not return_kv_cache:
                return semantic_logits, coarse_logits

            return (semantic_logits, coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

        (null_semantic_logits, null_coarse_logits), (null_new_kv_cache, null_new_embed_cache) = self.forward(*args, cond_drop_prob = 1., return_cache = True, kv_cache = next(iter_kv_cache, None), embed_cache = next(iter_embed_cache, None), **kwargs)
        new_kv_caches.append(null_new_kv_cache)
        new_embed_caches.append(null_new_embed_cache)

        scaled_semantic_logits = None
        if exists(null_semantic_logits):
            scaled_semantic_logits = null_semantic_logits + (semantic_logits - null_semantic_logits) * cond_scale

        scaled_coarse_logits = null_coarse_logits + (coarse_logits - null_coarse_logits) * cond_scale

        if not return_kv_cache:
            return scaled_semantic_logits, scaled_coarse_logits

        return (scaled_semantic_logits, scaled_coarse_logits), (torch.stack(new_kv_caches), torch.stack(new_embed_caches))

    @beartype
    def forward(
        self,
        *,
        artics_features,
        coarse_token_ids,
        self_attn_mask = None,
        text: list[str] | None = None,
        text_embeds = None,
        cond_drop_prob = None,
        return_only_coarse_logits = False,
        return_cache = False,
        kv_cache = None,
        embed_cache = None,
        f0_tokens = None
    ):
        b, device = artics_features.shape[0], artics_features.device
        arange = partial(torch.arange, device = device)

        has_text = exists(text) or exists(text_embeds)
        assert not (self.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.embed_text(text, output_device = device)

        text_mask = None
        if exists(text_embeds):
            text_mask = torch.any(text_embeds != 0, dim = -1)

            text_embeds = self.proj_text_embed(text_embeds)

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        if exists(text_mask) and cond_drop_prob > 0:
            keep_mask = prob_mask_like((b,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        # coarse_token_ids, artics_features = map(lambda t: rearrange(t, 'b ... -> b (...)'), (coarse_token_ids, artics_features))

        offsets = self.codebook_size * arange(self.num_coarse_quantizers)
        offsets = repeat(offsets, 'q -> 1 (n q)', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        offsets = offsets[:, :coarse_token_ids.shape[-1]]
        coarse_token_ids = coarse_token_ids + offsets

        # semantic_tokens = get_embeds(self.semantic_embedding, artics_features)
        semantic_tokens = self.proj_semantics(artics_features)
        
        coarse_tokens = self.coarse_embedding(coarse_token_ids)

        coarse_quantize_tokens = repeat(self.coarse_quantize_embedding.weight, 'q d -> (n q) d', n = ceil_div(coarse_token_ids.shape[-1], self.num_coarse_quantizers))
        coarse_quantize_tokens = coarse_quantize_tokens[:coarse_token_ids.shape[-1], ...]
        coarse_tokens = coarse_tokens + coarse_quantize_tokens

        semantic_seq_len = semantic_tokens.shape[1]

        semantic_start_tokens = repeat(self.semantic_start_token, 'd -> b 1 d', b = b)
        coarse_start_tokens = repeat(self.coarse_start_token, 'd -> b 1 d', b = b)
        
        tokens_semantic = torch.cat((
            semantic_start_tokens,
            semantic_tokens), dim=1)
        tokens_coarse = torch.cat((
            coarse_start_tokens,
            coarse_tokens
        ), dim = 1)

        tokens = apply_positional_embeddings(tokens_semantic, tokens_coarse, self.SPE, num_q=self.num_coarse_quantizers)

        # engineer the attention bias so that cross attention is not dominated by relative positions

        seq_len = tokens.shape[-2]

        attn_bias = None

        if exists(self.transformer.rel_pos_bias):
            attn_bias = self.transformer.rel_pos_bias(seq_len, seq_len)

            is_semantic = arange(seq_len) < (semantic_seq_len + 1) # semantic seq len + start token
            is_cross_attn = rearrange(is_semantic, 'i -> i 1') ^ rearrange(is_semantic, 'j -> 1 j')

            attn_bias = torch.where(
                is_cross_attn,
                self.cross_attn_bias,
                attn_bias
            )

        # attend
        
        tokens, new_kv_cache = self.transformer(
            tokens,
            context = text_embeds,
            attn_bias = attn_bias,
            self_attn_mask = self_attn_mask,
            context_mask = text_mask,
            kv_cache = kv_cache,
            return_kv_cache = True
        )

        if exists(embed_cache):
            tokens = torch.cat((embed_cache, tokens), dim = -2)

        new_embed_cache = tokens

        # segment into semantic and coarse acoustic tokens

        pred_semantic_tokens, pred_coarse_tokens = tokens[:, :semantic_seq_len], tokens[:, (semantic_seq_len + 1):]

        # semantic logits

        semantic_logits = self.to_semantic_logits(pred_semantic_tokens) if not return_only_coarse_logits and exists(self.to_semantic_logits) else None

        # get coarse logits

        n = pred_coarse_tokens.shape[1]
        nq = round_down_nearest_multiple(n, self.num_coarse_quantizers)

        pred_coarse_tokens_groupable, pred_coarse_tokens_remainder = pred_coarse_tokens[:, :nq], pred_coarse_tokens[:, nq:]

        pred_coarse_tokens_groupable = rearrange(pred_coarse_tokens_groupable, 'b (n q) d -> b n q d', q = self.num_coarse_quantizers)

        coarse_logits_groupable = einsum('q c d, b n q d -> b n q c', self.coarse_logit_weights, pred_coarse_tokens_groupable)

        coarse_logits_groupable = rearrange(coarse_logits_groupable, 'b n q c -> b (n q) c')

        remainder_num_quantizers = pred_coarse_tokens_remainder.shape[1]

        if remainder_num_quantizers > 0:
            coarse_logits_remainder = einsum('q c d, b q d -> b q c', self.coarse_logit_weights[:remainder_num_quantizers], pred_coarse_tokens_remainder)

            coarse_logits = torch.cat((coarse_logits_groupable, coarse_logits_remainder), dim = 1)
        else:
            coarse_logits = coarse_logits_groupable

        logits = (semantic_logits, coarse_logits)

        if not return_cache:
            return logits

        return logits, (new_kv_cache, new_embed_cache)

class CoarseTransformerWrapper(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        transformer: CoarseTransformer,
        codec = None,
        # wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None = None,
        audio_conditioner = None,
        pad_id = -1,
        unique_consecutive = False,
        semantic_cross_entropy_loss_weight = 0.,
        mask_prob = 0.15
    ):
        super().__init__()
        self.codec = codec
        # self.wav2vec = wav2vec

        self.transformer = transformer
        self.to(transformer.device)
        self.audio_conditioner = audio_conditioner

        assert not (exists(audio_conditioner) and not transformer.has_condition), 'if conditioning on audio embeddings from mulan, transformer has_condition must be set to True'

        self.unique_consecutive = unique_consecutive
        self.pad_id = pad_id

        self.semantic_cross_entropy_loss_weight = semantic_cross_entropy_loss_weight

        if codec:
            rq_groups = codec.rq_groups
        else:
            rq_groups = 1

        self.num_coarse_quantizers = transformer.num_coarse_quantizers * rq_groups
        self.semantic_eos_id = transformer.semantic_eos_id
        self.coarse_eos_id = transformer.coarse_eos_id

        self.mask_prob = mask_prob

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_chunk(self, artics_chunk,  
                       attn_chunk,
                       sampled_coarse_token_ids,
                       init_coarse_time_step=0,
                       text_embeds = None,
                       cond_scale = 3.0,
                       use_kv_cache = True,
                       filter_thres = 0.9,
                       temperature = 1.,
                       **kwargs):
        
        batch_size = artics_chunk.shape[0]
        kv_cache = None
        embed_cache = None
        # for time_step in range(init_coarse_time_step, round(artics_chunk.shape[1]/1.25)):
        for time_step in range(init_coarse_time_step, round(artics_chunk.shape[1])):
            for ind in range(self.num_coarse_quantizers):
                # print("XXXX", time_step)
                just_finished_quantizer_step = (ind == 0 and time_step > 0)
                # if time_step > 0 or shifts > 0:
                # print(artics_chunk.shape)
                # print(sampled_coarse_token_ids.shape)
                # print(attn_chunk.shape)
                # if kv_cache is not None:
                #     print(kv_cache.shape)
                # if embed_cache is not None:
                #     print(embed_cache.shape)
                # print("----------------------------------")
                (_, coarse_logits), (next_kv_cache, next_embed_cache) = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = sampled_coarse_token_ids,
                    artics_features = artics_chunk,
                    self_attn_mask = attn_chunk, # attend to semantic bos and all coarse tokens,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_kv_cache = True,
                    kv_cache = kv_cache,
                    embed_cache = embed_cache,
                    return_only_coarse_logits = True,
                    **kwargs
                )
                true_vector = torch.ones((batch_size, 1), dtype=torch.bool, device="cuda")
                # breakpoint()
                attn_chunk = torch.cat([attn_chunk, true_vector], dim=1)

                if use_kv_cache:
                    kv_cache = next_kv_cache
                    embed_cache = next_embed_cache

                last_coarse_logits = coarse_logits[:, -1]

                if not just_finished_quantizer_step:
                    last_coarse_logits[:, -1] = float('-inf') # prevent from eos in the middle of a time step

                filtered_logits = top_k(last_coarse_logits, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_coarse_token_ids = torch.cat((sampled_coarse_token_ids, sampled), dim = -1)
                # total_sampled_coarse_token_ids = torch.cat((total_sampled_coarse_token_ids, sampled), dim = -1)

        return sampled_coarse_token_ids


    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generate(
        self,
        *,
        artics_features,
        f0_tokens=None,
        prime_wave: Tensor | None = None,
        prime_wave_input_sample_hz=None,
        prime_coarse_token_ids: Tensor | None = None,
        text: list[str] | None = None,
        text_embeds=None,
        temperature=1.0,
        reconstruct_wave=False,
        use_kv_cache=True,
        **kwargs
    ):
        batch, device = artics_features.shape[0], self.device
        artics_features = artics_features.to(device)

        # Initialize coarse token ids.
        assert not (exists(prime_wave) and exists(prime_coarse_token_ids)), (
            'you can either pass in the prime as a raw wave (codec required) or as preprocessed acoustic token ids'
        )
        if exists(prime_coarse_token_ids):
            coarse_token_ids = prime_coarse_token_ids
        elif exists(prime_wave):
            assert exists(self.codec)
            with torch.inference_mode():
                self.codec.eval()
                _, indices, _ = self.codec(
                    prime_wave,
                    return_encoded=True,
                    input_sample_hz=prime_wave_input_sample_hz
                )
                coarse_token_ids = indices[..., :self.num_coarse_quantizers]
                coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        else:
            coarse_token_ids = torch.empty((batch, 0), device=device, dtype=torch.long)

        # Derive text embeddings if needed.
        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)
        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device=device)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value=self.pad_id)

        # Prepare attention mask.
        artics_sum = torch.sum(artics_features, dim=-1)
        self_attn_mask = (artics_sum != 0.0)
        self_attn_mask_expanded = self_attn_mask.unsqueeze(-1).repeat(1, 1, 64)
        coarse_token_len = coarse_token_ids.shape[-1]
        artics_features = artics_features.masked_fill(~self_attn_mask_expanded, 0)
        # Pad to attend to the semantic BOS and all coarse tokens.
        self_attn_mask = F.pad(self_attn_mask, (1, coarse_token_len + 1), value=True)

        # Chunking parameters.
        mA = 128  # tokens per chunk from artics_features
        mB = 128  # tokens per chunk for attention slicing offset
        # Concurrent grouping: number of chunks to process concurrently.
        concurrent_chunk_batch = 32

        # (Optional) If a special edge-case is detected, adjust the features.
        if artics_features.shape[1] % mA in [63, 127, 191, 255]:
            artics_features = artics_features[:, :-1, ...]
            self_attn_mask = self_attn_mask[..., :-1]

        # Compute sequence length and the number of chunks (use math.ceil).
        seq_len = artics_features.shape[1]
        num_chunks = math.ceil(seq_len / mB)
        
        artics_chunks = []
        attn_chunks = []
        sampled_coarses = []
        for ii in range(num_chunks):
            start_idx = ii * mB
            end_idx = min(ii * mB + mA, seq_len)
            # Skip if the slice is empty.
            if start_idx >= seq_len:
                continue
            artics_chunk = artics_features[:, start_idx:end_idx, ...]
            # Pad the time dimension if needed so that each chunk has length mA.
            if artics_chunk.shape[1] < mA:
                xos_tensor = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]).unsqueeze(0) 
                pad_len = mA - artics_chunk.shape[1]
                pad = xos_tensor.repeat(pad_len, 1)
                pad = pad.unsqueeze(0).to(self.device)
                artics_chunk = torch.cat([artics_chunk, pad], dim=1)
            
            # Adjust attention slice with the same logic.
            attn_start = 1 + ii * mB
            attn_end = 1 + min(ii * mB + mA, seq_len)
            attn_chunk = torch.cat([
                self_attn_mask[..., :1],
                self_attn_mask[..., attn_start:attn_end],
                self_attn_mask[..., -1:]
            ], dim=1)
            # For consistency pad the attention chunk if needed.
            expected_attn_len = 1 + mA + 1  # bos + mA + eos
            if attn_chunk.shape[1] < expected_attn_len:
                attn_chunk = F.pad(attn_chunk, (0, expected_attn_len - attn_chunk.shape[1]), mode="constant", value=False)
            
            # Clone the coarse token ids for this chunk.
            coarse_tokens_clone = coarse_token_ids.clone()
            artics_chunks.append(artics_chunk)
            attn_chunks.append(attn_chunk)
            sampled_coarses.append(coarse_tokens_clone)

        # Process chunks concurrently in groups.
        chunk_results = [None] * len(artics_chunks)
        for start in range(0, len(artics_chunks), concurrent_chunk_batch):
            end = min(start + concurrent_chunk_batch, len(artics_chunks))
            group_size = end - start

            # Concatenate along the new batch axis. Each chunk has shape [batch, mA, ...],
            # so concatenating them gives shape [group_size * batch, mA, ...].
            artics_group = torch.cat(artics_chunks[start:end], dim=0)
            attn_group = torch.cat(attn_chunks[start:end], dim=0)
            coarse_group = torch.cat(sampled_coarses[start:end], dim=0)
            # breakpoint()
            group_out = self.generate_chunk(
                artics_group, attn_group, coarse_group, temperature=temperature
            )
            # breakpoint()
            # group_out should be of shape [group_size * batch, tokens_generated_for_chunk].
            # Split the output back into individual chunks.
            group_splits = torch.split(group_out, batch, dim=0)
            for idx, out_chunk in enumerate(group_splits):
                chunk_results[start + idx] = out_chunk

        # Reassemble the final outputs per original batch sample by concatenating along the token dimension.
        total_sampled_coarse_token_ids = torch.cat(chunk_results, dim=1)

        # Post-process: mask tokens beyond EOS, and reshape to [batch, n_tokens, num_coarse_quantizers].
        sampled_coarse_token_ids = mask_out_after_eos_id(
            total_sampled_coarse_token_ids, self.coarse_eos_id, keep_eos=True
        )
        sampled_coarse_token_ids = rearrange(
            sampled_coarse_token_ids, 'b (n q) -> b n q', q=self.num_coarse_quantizers
        )

        if not reconstruct_wave:
            return sampled_coarse_token_ids.squeeze().detach().cpu().numpy()

        # Decode waveform if required.
        assert exists(self.codec)
        coarse_tokens_are_variable_lengthed = (sampled_coarse_token_ids == -1).any()
        if not coarse_tokens_are_variable_lengthed:
            wav = self.codec.decode_from_codebook_indices(sampled_coarse_token_ids, f0_tokens)
            return rearrange(wav, 'b 1 n -> b n')

        # Decode variable-length tokens per sample.
        wavs = []
        for coarse_sample in sampled_coarse_token_ids:
            has_padding = reduce(coarse_sample == -1, 'n q -> n', 'any')
            coarse_sample_without_padding = coarse_sample[~has_padding]
            if has_padding.all():
                wavs.append(None)
                continue
            coarse_sample_without_padding = rearrange(coarse_sample_without_padding, '... -> 1 ...')
            wav = self.codec.decode_from_codebook_indices(
                coarse_sample_without_padding, rearrange(f0_tokens, '... -> 1 ...')
            )
            wav = rearrange(wav, '1 1 n -> n')
            wavs.append(wav)

        return wavs




    @eval_decorator
    @torch.inference_mode()
    @beartype
    def generateOLD(
        self,
        *,
        artics_features,
        f0_tokens = None,
        prime_wave: Tensor | None = None,
        prime_wave_input_sample_hz = None,
        prime_coarse_token_ids: Tensor | None = None,
        text: list[str] | None = None,
        text_embeds = None,
        max_time_steps = 512,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        reconstruct_wave = False,
        use_kv_cache = True,
        **kwargs
    ):
        batch, device = artics_features.shape[0], self.device

        artics_features = artics_features.to(device)

        # initialize coarse token ids
        # if a prime audio wave was supplied, then start off with appropriate acoustic tokens

        assert not (exists(prime_wave) and exists(prime_coarse_token_ids)), 'you can either pass in the prime as a raw wave (codec required) or as preprocessed acoustic token ids'

        if exists(prime_coarse_token_ids):
            coarse_token_ids = prime_coarse_token_ids
        elif exists(prime_wave):
            assert exists(self.codec)
            with torch.inference_mode():
                self.codec.eval()

                _, indices, _ = self.codec(
                    prime_wave,
                    return_encoded = True,
                    input_sample_hz = prime_wave_input_sample_hz
                )

                coarse_token_ids = indices[..., :self.num_coarse_quantizers]
                coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')
        else:
            coarse_token_ids = torch.empty((batch, 0), device = device, dtype = torch.long)

        # derive text embeddings if needed

        has_text = exists(text) or exists(text_embeds)
        assert not (self.transformer.has_condition ^ has_text)

        if not exists(text_embeds) and exists(text):
            with torch.inference_mode():
                text_embeds = self.transformer.embed_text(text, output_device = device)

        if self.unique_consecutive:
            semantic_token_ids = batch_unique_consecutive(semantic_token_ids, pad_value=self.pad_id)

        # initialize

        artics_sum = torch.sum(artics_features, dim=-1)
        self_attn_mask = (artics_sum != 0.0)# & (artics_sum != torch.sum(self.semantic_eos_id))
        self_attn_mask_expanded = self_attn_mask.unsqueeze(-1).repeat(1, 1, 64)
        coarse_token_len = coarse_token_ids.shape[-1]
        artics_features = artics_features.masked_fill(~self_attn_mask_expanded, 0)
        self_attn_mask = F.pad(self_attn_mask, (1, coarse_token_len + 1), value = True) # attend to semantic bos and all coarse tokens

        init_coarse_time_step = 0
        sampled_coarse_token_ids = coarse_token_ids.clone()
        total_sampled_coarse_token_ids = sampled_coarse_token_ids.clone()

        # kv cache

        mA = 256
        mC = 205
        artics_chunk = artics_features[:,:mA,...]
        attn_chunk = torch.cat([self_attn_mask[...,:1], self_attn_mask[...,1:mA+1], self_attn_mask[...,-1:]], dim = 1)
        ct_chunk = 0
        shifts = 0

        kv_cache = None
        embed_cache = None
        for time_step in tqdm(range(init_coarse_time_step, int(artics_features.shape[1]/1.25)-40), desc = 'generating coarse'):
            for ind in range(self.num_coarse_quantizers):
                just_finished_quantizer_step = (ind == 0 and time_step > 0)
                # if time_step > 0 or shifts > 0:
                #     print(artics_chunk.shape)
                #     print(sampled_coarse_token_ids.shape)
                #     print(attn_chunk.shape)
                #     print(kv_cache.shape)
                #     print(embed_cache.shape)
                #     print("----------------------------------")
                (_, coarse_logits), (next_kv_cache, next_embed_cache) = self.transformer.forward_with_cond_scale(
                    coarse_token_ids = sampled_coarse_token_ids,
                    artics_features = artics_chunk,
                    self_attn_mask = attn_chunk, # attend to semantic bos and all coarse tokens,
                    text_embeds = text_embeds,
                    cond_scale = cond_scale,
                    return_kv_cache = True,
                    kv_cache = kv_cache,
                    embed_cache = embed_cache,
                    return_only_coarse_logits = True,
                    **kwargs
                )
                # breakpoint()
                true_vector = torch.ones((self_attn_mask.shape[0], 1), dtype=torch.bool, device=self_attn_mask.device)
                attn_chunk = torch.cat([attn_chunk, true_vector], dim=1)

                if use_kv_cache:
                    kv_cache = next_kv_cache
                    embed_cache = next_embed_cache

                last_coarse_logits = coarse_logits[:, -1]

                if not just_finished_quantizer_step:
                    last_coarse_logits[:, -1] = float('-inf') # prevent from eos in the middle of a time step

                filtered_logits = top_k(last_coarse_logits, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

                sampled = rearrange(sampled, 'b -> b 1')
                sampled_coarse_token_ids = torch.cat((sampled_coarse_token_ids, sampled), dim = -1)
                total_sampled_coarse_token_ids = torch.cat((total_sampled_coarse_token_ids, sampled), dim = -1)

            ct_chunk += 1

            if ct_chunk == mC:
                shifts += 1
                artics_chunk = artics_features[:,4*shifts:mA+4*shifts]
                sampled_coarse_token_ids = sampled_coarse_token_ids[:, 9:]
                attn_chunk = torch.cat([self_attn_mask[...,:1], self_attn_mask[...,shifts*4+1:mA+1+shifts*4], self_attn_mask[...,-sampled_coarse_token_ids.shape[1]-1:]], dim = 1)
                kv_cache = kv_cache[...,9:,:]
                embed_cache = embed_cache[...,9:,:]
                # breakpoint()
                ct_chunk -= 3



        sampled_coarse_token_ids = mask_out_after_eos_id(total_sampled_coarse_token_ids, self.coarse_eos_id, keep_eos = False)
        sampled_coarse_token_ids = rearrange(sampled_coarse_token_ids, 'b (n q) -> b n q', q = self.num_coarse_quantizers)

        if not reconstruct_wave:
            return sampled_coarse_token_ids

        assert exists(self.codec)

        coarse_tokens_are_variable_lengthed = (sampled_coarse_token_ids == -1).any()

        if not coarse_tokens_are_variable_lengthed:
            wav = self.codec.decode_from_codebook_indices(sampled_coarse_token_ids, f0_tokens)
            return rearrange(wav, 'b 1 n -> b n')

        # handle variable lengthed coarse tokens

        wavs = []
        for coarse_sample in sampled_coarse_token_ids:
            has_padding = reduce(coarse_sample == -1, 'n q -> n', 'any')
            coarse_sample_without_padding = coarse_sample[~has_padding]

            if has_padding.all():
                wavs.append(None)
                continue
            
            coarse_sample_without_padding = rearrange(coarse_sample_without_padding, '... -> 1 ...')

            wav = self.codec.decode_from_codebook_indices(coarse_sample_without_padding, rearrange(f0_tokens, '... -> 1 ...'))
            wav = rearrange(wav, '1 1 n -> n')

            wavs.append(wav)

        return wavs

    def forward(
        self,
        *,
        artics_features = None,
        whisper_tokens = None,
        text = None,
        text_embeds = None,
        coarse_token_ids = None,
        return_loss = False,
        **kwargs
    ):
        assert exists(artics_features), 'either raw waveform (raw_wave) is given or semantic token ids are given (artics_features)'

        # raw_wave_for_codec = default(raw_wave_for_codec, raw_wave)
        # assert exists(raw_wave_for_codec) or exists(coarse_token_ids), 'either raw waveform (raw_wav) is given, or coarse and fine token ids (coarse_token_ids, fine_token_ids)'

        # assert not all(map(exists, (raw_wave, raw_wave_for_codec, artics_features, coarse_token_ids)))

        # if exists(self.audio_conditioner):
            # assert exists(raw_wave)
            # assert not exists(text) and not exists(text_embeds)
            # text_embeds = self.audio_conditioner(wavs = raw_wave, namespace = 'coarse') # technically audio embeds, but shared text-audio joint embedding space for mulan

        # if not exists(artics_features):
            # assert exists(self.wav2vec), 'VQWav2Vec must be be provided if given raw wave for training'
            # artics_features = self.wav2vec(raw_wave, flatten = False)

        if not exists(coarse_token_ids):
            assert exists(whisper_tokens), 'Codec must be provided if given raw wave for training'
            coarse_token_ids = whisper_tokens[..., :self.num_coarse_quantizers]
            

        # artics_features = rearrange(artics_features, 'b ... -> b (...)')
        coarse_token_ids = rearrange(coarse_token_ids, 'b ... -> b (...)')

        if return_loss:
            # artics_features = append_eos_id(artics_features, self.transformer.semantic_eos_id)
            coarse_token_ids = append_eos_id(coarse_token_ids, self.transformer.coarse_eos_id)

        if self.unique_consecutive:
            artics_features = batch_unique_consecutive(artics_features, pad_value = self.pad_id)

        if return_loss:
            semantic_labels, coarse_labels = artics_features, coarse_token_ids.clone()
            coarse_token_ids = coarse_token_ids[:, :-1]

        # self attention mask would omit any padding and eos tokens in the semantic prime

        artics_sum = torch.sum(artics_features, dim=-1)
        # self_attn_mask = (artics_sum != 0.0) & (artics_sum != torch.sum(self.semantic_eos_id))
        self_attn_mask = (artics_sum != 0.0)# & (artics_sum != torch.sum(self.semantic_eos_id))
        self_attn_mask_expanded = self_attn_mask.unsqueeze(-1).repeat(1, 1, 64)
        artics_features = artics_features.masked_fill(~self_attn_mask_expanded, 0)
        
        # self_attn_mask = (artics_features != self.pad_id) & (artics_features != self.semantic_eos_id)
        # artics_features = artics_features.masked_fill(~self_attn_mask, 0)

        coarse_token_len = coarse_token_ids.shape[-1]
        self_attn_mask = F.pad(self_attn_mask, (1, coarse_token_len + 1), value = True) # attend to semantic bos and all coarse tokens

        # forgetful causal mask - structured dropout

        if self.mask_prob > 0 and self.training:
            self_attn_mask &= generate_mask_with_prob(self_attn_mask.shape, self.mask_prob, device = self_attn_mask.device)

        # breakpoint()
        semantic_logits, coarse_logits = self.transformer(
            artics_features = artics_features,
            coarse_token_ids = coarse_token_ids,
            self_attn_mask = self_attn_mask,
            text = text,
            text_embeds = text_embeds,
            **kwargs
        )

        # whether to early return the logits

        if not return_loss:
            return semantic_logits, coarse_logits

        coarse_logits, semantic_logits = map(lambda t: maybe(rearrange)(t, 'b n c -> b c n'), (coarse_logits, semantic_logits))

        if self.unique_consecutive:
            num_coarse_logits, _num_semantic_logits = coarse_labels.numel(), (semantic_labels != self.pad_id).sum()
        else:
            num_coarse_logits, _num_semantic_logits = coarse_logits.shape[-1], semantic_logits.shape[-1]

        semantic_loss = 0.
        num_semantic_logits = 0

        if self.semantic_cross_entropy_loss_weight > 0 and exists(semantic_logits):
            num_semantic_logits = _num_semantic_logits

            semantic_loss = F.cross_entropy(
                semantic_logits,
                semantic_labels,
                ignore_index = self.pad_id
            )

        coarse_loss = F.cross_entropy(
            coarse_logits,
            coarse_labels,
            ignore_index = self.pad_id
        )

        return (
            semantic_loss * num_semantic_logits * self.semantic_cross_entropy_loss_weight +
            coarse_loss * num_coarse_logits
        ) / (num_semantic_logits + num_coarse_logits)

# audio LM


class AudioLM(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        # wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None, 
        codec = None,
        # semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        audio_conditioner  = None,
        unique_consecutive = False
    ):
        super().__init__()

        self.audio_conditioner = audio_conditioner
        self.codec = codec

        # self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        # self.needs_text = any([self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])
        self.needs_text = False

        # self.semantic = SemanticTransformerWrapper(
        #     wav2vec = wav2vec,
        #     transformer = semantic_transformer,
        #     audio_conditioner = audio_conditioner,
        #     unique_consecutive = unique_consecutive
        # )

        self.coarse = CoarseTransformerWrapper(
            # wav2vec = wav2vec,
            codec = codec,
            transformer = coarse_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    def forward(
        self,
        artics_features,
        f0_tokens,
        *,
        text: list[str] | None = None,
        text_embeds: Tensor | None = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_wave_path = None,
        return_coarse_generated_wave = False,
        mask_out_generated_fine_tokens = False,
        reconstruct_wave = False,
        temperature = 1.0,
        gt_coarse = None
    ):
        assert not (self.needs_text and (not exists(text) and not exists(text_embeds))), 'text needs to be passed in if one of the transformer requires conditioning'

        if self.needs_text:
            if exists(text):
                text_embeds = self.semantic.embed_text(text)

        assert not (exists(prime_wave) and exists(prime_wave_path)), 'prompt audio must be given as either `prime_wave: Tensor` or `prime_wave_path: str`'

        if exists(prime_wave):
            assert exists(prime_wave_input_sample_hz), 'the input sample frequency for the prompt audio must be given as `prime_wave_input_sample_hz: int`'
            prime_wave = prime_wave.to(self.device)
        elif exists(prime_wave_path):
            prime_wave_path = Path(prime_wave_path)
            assert exists(prime_wave_path), f'file does not exist at {str(prime_wave_path)}'

            prime_wave, prime_wave_input_sample_hz = torchaudio.load(str(prime_wave_path))
            prime_wave = prime_wave.to(self.device)

        # semantic_token_ids = self.semantic.generate(
        #     text_embeds = text_embeds if self.semantic_has_condition else None,
        #     batch_size = batch_size,
        #     prime_wave = prime_wave,
        #     prime_wave_input_sample_hz = prime_wave_input_sample_hz,
        #     max_length = max_length
        # )
        if gt_coarse is None:
            coarse_token_ids_or_recon_wave = self.coarse.generate(
                text_embeds = text_embeds if self.coarse_has_condition else None,
                artics_features = artics_features,
                f0_tokens = f0_tokens,
                prime_wave = prime_wave,
                prime_wave_input_sample_hz = prime_wave_input_sample_hz,
                reconstruct_wave = return_coarse_generated_wave,
                temperature=temperature
            )

            if return_coarse_generated_wave:
                return coarse_token_ids_or_recon_wave
        else:
            coarse_token_ids_or_recon_wave = gt_coarse

        # wav = self.codec.decode_from_codebook_indices(coarse_token_ids_or_recon_wave, f0_tokens)
        # breakpoint()
        # from scipy.io.wavfile import write
        # write("output_audio.wav", 16000, coarse_token_ids_or_recon_wave[0].T)

        # generated_wave = self.fine.generate(
        #     text_embeds = text_embeds if self.fine_has_condition else None,
        #     coarse_token_ids = coarse_token_ids_or_recon_wave,
        #     f0_tokens = f0_tokens,
        #     prime_wave = prime_wave,
        #     prime_wave_input_sample_hz = prime_wave_input_sample_hz,
        #     reconstruct_wave = reconstruct_wave,
        #     mask_out_generated_fine_tokens = mask_out_generated_fine_tokens,
        #     temperature=temperature
        # )

        return coarse_token_ids_or_recon_wave



class AudioLMOLD(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        # wav2vec: FairseqVQWav2Vec | HubertWithKmeans | None, 
        codec = None,
        # semantic_transformer: SemanticTransformer,
        coarse_transformer: CoarseTransformer,
        fine_transformer: FineTransformer,
        audio_conditioner: AudioConditionerBase | None = None,
        unique_consecutive = False
    ):
        super().__init__()

        self.audio_conditioner = audio_conditioner

        # assert semantic_transformer.num_semantic_tokens == coarse_transformer.num_semantic_tokens
        assert coarse_transformer.codebook_size == fine_transformer.codebook_size
        assert coarse_transformer.num_coarse_quantizers == fine_transformer.num_coarse_quantizers
        self.codec = codec
        if codec:
            assert (fine_transformer.num_coarse_quantizers + fine_transformer.num_fine_quantizers) == codec.num_quantizers

        # self.semantic_has_condition = semantic_transformer.has_condition
        self.coarse_has_condition = coarse_transformer.has_condition
        self.fine_has_condition = fine_transformer.has_condition
        # self.needs_text = any([self.semantic_has_condition, self.coarse_has_condition, self.fine_has_condition])
        self.needs_text = False

        # self.semantic = SemanticTransformerWrapper(
        #     wav2vec = wav2vec,
        #     transformer = semantic_transformer,
        #     audio_conditioner = audio_conditioner,
        #     unique_consecutive = unique_consecutive
        # )

        self.coarse = CoarseTransformerWrapper(
            # wav2vec = wav2vec,
            codec = codec,
            transformer = coarse_transformer,
            audio_conditioner = audio_conditioner,
            unique_consecutive = unique_consecutive
        )

        self.fine = FineTransformerWrapper(
            codec= codec,
            transformer = fine_transformer,
            audio_conditioner = audio_conditioner
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @eval_decorator
    @torch.inference_mode()
    def forward(
        self,
        artics_features,
        f0_tokens,
        *,
        text: list[str] | None = None,
        text_embeds: Tensor | None = None,
        prime_wave = None,
        prime_wave_input_sample_hz = None,
        prime_wave_path = None,
        return_coarse_generated_wave = False,
        mask_out_generated_fine_tokens = False,
        reconstruct_wave = False,
        temperature = 1.0,
        gt_coarse = None
    ):
        assert not (self.needs_text and (not exists(text) and not exists(text_embeds))), 'text needs to be passed in if one of the transformer requires conditioning'

        if self.needs_text:
            if exists(text):
                text_embeds = self.semantic.embed_text(text)

        assert not (exists(prime_wave) and exists(prime_wave_path)), 'prompt audio must be given as either `prime_wave: Tensor` or `prime_wave_path: str`'

        if exists(prime_wave):
            assert exists(prime_wave_input_sample_hz), 'the input sample frequency for the prompt audio must be given as `prime_wave_input_sample_hz: int`'
            prime_wave = prime_wave.to(self.device)
        elif exists(prime_wave_path):
            prime_wave_path = Path(prime_wave_path)
            assert exists(prime_wave_path), f'file does not exist at {str(prime_wave_path)}'

            prime_wave, prime_wave_input_sample_hz = torchaudio.load(str(prime_wave_path))
            prime_wave = prime_wave.to(self.device)

        # semantic_token_ids = self.semantic.generate(
        #     text_embeds = text_embeds if self.semantic_has_condition else None,
        #     batch_size = batch_size,
        #     prime_wave = prime_wave,
        #     prime_wave_input_sample_hz = prime_wave_input_sample_hz,
        #     max_length = max_length
        # )
        if gt_coarse is None:
            coarse_token_ids_or_recon_wave = self.coarse.generate(
                text_embeds = text_embeds if self.coarse_has_condition else None,
                artics_features = artics_features,
                f0_tokens = f0_tokens,
                prime_wave = prime_wave,
                prime_wave_input_sample_hz = prime_wave_input_sample_hz,
                reconstruct_wave = return_coarse_generated_wave,
                temperature=temperature
            )

            if return_coarse_generated_wave:
                return coarse_token_ids_or_recon_wave
        else:
            coarse_token_ids_or_recon_wave = gt_coarse

        # wav = self.codec.decode_from_codebook_indices(coarse_token_ids_or_recon_wave, f0_tokens)
        # from scipy.io.wavfile import write
        # write("output_audio.wav", 16000, wav[0].T)
        # breakpoint()

        generated_wave = self.fine.generate(
            text_embeds = text_embeds if self.fine_has_condition else None,
            coarse_token_ids = coarse_token_ids_or_recon_wave,
            f0_tokens = f0_tokens,
            prime_wave = prime_wave,
            prime_wave_input_sample_hz = prime_wave_input_sample_hz,
            reconstruct_wave = reconstruct_wave,
            mask_out_generated_fine_tokens = mask_out_generated_fine_tokens,
            temperature=temperature
        )

        return generated_wave