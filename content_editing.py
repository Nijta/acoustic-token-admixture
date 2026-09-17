"""NER-triggered content replacement with fixed-frame graft splicing (C=8).

The replacement tokens are generated independently (phonetic prefix/postfix from the
surrounding transcript), then joined back with a fixed overlap of C=8 encoder frames
(~160 ms at 50 fps) crossfaded onto the real audio at each seam. This improves
continuity without regenerating neighbour words or acoustic priming.
"""
from __future__ import annotations

import numpy as np

FPS = 50
GRAFT_K = 8  # optimum fixed context / overlap frames (C in the paper)


def words_to_prefix_suffix(word_timestamps, group):
    """Phonetic context strings from words before/after the selected span."""
    start_i, end_i = group[0], group[-1]
    prefix = ' '.join(word_timestamps[i][0] for i in range(start_i))
    suffix = ' '.join(word_timestamps[i][0] for i in range(end_i + 1, len(word_timestamps)))
    if prefix:
        prefix += ' '
    if suffix:
        suffix = ' ' + suffix
    return prefix, suffix


def generate_replacement(ALW, AW, rep_text, prefix='', suffix=''):
    """Generate coarse tokens + pitch for the replacement phrase."""
    artics_pp, prlen, polen = ALW.predict_artics(rep_text, prefix=prefix, postfix=suffix)
    pitch_pp = ALW.predict_f0(artics_pp)
    tokens = AW.generate([artics_pp])[0][:artics_pp.shape[0]]
    end_slice = None if polen == 0 else -polen
    target_tok = tokens[prlen:end_slice]
    target_pitch = pitch_pp[prlen:end_slice]
    if target_tok.shape[0] and 1024 in target_tok[:, 0]:
        limit = list(target_tok[:, 0]).index(1024)
        target_tok, target_pitch = target_tok[:limit], target_pitch[:limit]
    return artics_pp, prlen, polen, tokens, pitch_pp, target_tok, target_pitch


def extract_graft_chunk(full_tok, pitch_pp, prlen, polen, n_frames, graft_k=GRAFT_K):
    """Extend the generated chunk by `graft_k` frames on each side before stripping."""
    ext = int(graft_k)
    a = max(0, prlen - ext)
    b = min(n_frames, (n_frames - polen) + ext)
    tok = full_tok[a:b]
    pit = pitch_pp[a:b]
    if tok.shape[0] and 1024 in tok[:, 0]:
        limit = list(tok[:, 0]).index(1024)
        tok, pit = tok[:limit], pit[:limit]
    return tok, pit


def _crossfade_append(base_bn, base_pi, ext_bn, ext_pi, width):
    """Append ext onto base with a linear crossfade of `width` frames at the join."""
    w = int(min(width, base_bn.shape[0], ext_bn.shape[0]))
    if w <= 0:
        return np.concatenate([base_bn, ext_bn], axis=0), np.concatenate([base_pi, ext_pi], axis=0)
    ramp = np.linspace(0.0, 1.0, w + 2)[1:-1][:, None]
    blended = base_bn[-w:] * (1.0 - ramp) + ext_bn[:w] * ramp
    blended_p = base_pi[-w:] * 0.5 + ext_pi[:w] * 0.5
    out_bn = np.concatenate([base_bn[:-w], blended, ext_bn[w:]], axis=0)
    out_pi = np.concatenate([base_pi[:-w], blended_p, ext_pi[w:]], axis=0)
    return out_bn, out_pi


def graft_splice(
    final_bottleneck,
    pitch,
    start_idx,
    end_idx,
    repl_bottleneck,
    repl_pitch,
    graft_k=GRAFT_K,
):
    """Splice replacement bottleneck/pitch into [start_idx, end_idx] with C-frame crossfades."""
    k = int(graft_k)
    pre_bn, pre_pi = final_bottleneck[:start_idx], pitch[:start_idx]
    post_bn, post_pi = final_bottleneck[end_idx:], pitch[end_idx:]
    merged, merged_p = _crossfade_append(pre_bn, pre_pi, repl_bottleneck, repl_pitch, k)
    return _crossfade_append(merged, merged_p, post_bn, post_pi, k)


def apply_graft_replacement(
    final_bottleneck,
    pitch,
    start_idx,
    end_idx,
    ALW,
    AW,
    BW,
    rep_text,
    prefix,
    suffix,
    convert_pitch_fn,
    graft_k=GRAFT_K,
):
    """Full replacement: generate tokens, decode, graft-splice at fixed frame boundaries."""
    artics_pp, prlen, polen, full_tok, pitch_pp, _, _ = generate_replacement(
        ALW, AW, rep_text, prefix, suffix
    )
    graft_tok, graft_pitch = extract_graft_chunk(
        full_tok, pitch_pp, prlen, polen, artics_pp.shape[0], graft_k=graft_k
    )
    graft_pitch = convert_pitch_fn(graft_pitch)
    repl_bn = BW.decode_from_codebook_indices(graft_tok[None, ...])[0]
    n = min(repl_bn.shape[0], graft_pitch.shape[0])
    return graft_splice(
        final_bottleneck, pitch, start_idx, end_idx,
        repl_bn[:n], graft_pitch[:n], graft_k=graft_k,
    )
