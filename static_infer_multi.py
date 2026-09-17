#!/usr/bin/env python3
"""
Batch voice anonymization pipeline (no PNGs, multi-format input).

UPDATED VERSION (2-level parallelism):
- Runs N model processes (each process loads ONE ModelState instance)
- Inside each model process, runs M threads (ThreadPoolExecutor) that share that model
=> "2 models running in parallel with each 4 threads" == --model-procs 2 --threads-per-model 4

Why this fixes OOM vs pure multiprocessing:
- You control how many full model copies exist: exactly --model-procs

Notes:
- Uses spawn start method (safer with CUDA than fork).
- Optional model lock in case wrappers are not thread-safe (set USE_MODEL_LOCK=False to disable).
- No try/except added; exceptions will bubble and be reported by futures.
"""

import os

# Root of the released checkpoints and speaker pool (see README, "Model weights").
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
import glob
import time
import pickle
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import random
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import soundfile as sf
import torchaudio
import torch

# -------------------------------
# Your model wrappers and deps
# -------------------------------
from aligner.src import AlignerWrapper
from rvqwhisper.src.rvqwhisper import RVQFasterWhisperWrapper
from bigvgan.src.bigvgan import BigVGANWrapper

import sys
sys.path.insert(0, "./audiolm")
from audiolm import AudioLMWrapper

import pspi.pseudospeaker as nps
from pspi.pitch import extract as pitch_extract

# -------------------------------
# Thread-safety knobs
# -------------------------------
USE_MODEL_LOCK = True
_MODEL_LOCK = threading.Lock()

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

def _maybe_lock():
    return _MODEL_LOCK if USE_MODEL_LOCK else _NullContext()

# -------------------------------
# Helpers
# -------------------------------
import tempfile

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif", ".caf"}

def _is_wav(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".wav"

def decode_to_temp_wav_16k_mono(in_path: str) -> str:
    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)  # shape [T, C]
    audio = audio.mean(axis=1)  # mono [T]
    wav = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        sr = 16000

    fd, tmp_path = tempfile.mkstemp(prefix="tmp_decode_", suffix=".wav")
    os.close(fd)

    torchaudio.save(tmp_path, wav, sr, encoding="PCM_S", bits_per_sample=16)
    return tmp_path

def safe_basename(path: str) -> str:
    base = os.path.basename(path)
    return base.replace(" ", "_")

def merge_segments(segments, min_words=5, min_chars=70):
    from types import SimpleNamespace
    merged_segments = []
    i = 0
    while i < len(segments):
        current = segments[i]
        merged = SimpleNamespace(
            start=current.start,
            end=current.end,
            text=current.text,
            words=current.words
        )
        while (len(merged.words) < min_words or len(merged.text) < min_chars) and (i + 1 < len(segments)):
            i += 1
            next_seg = segments[i]
            merged.text = f"{merged.text.strip()} {next_seg.text.strip()}"
            merged.words = merged.words + next_seg.words
            merged.end = next_seg.end
        merged_segments.append(merged)
        i += 1
    return merged_segments

def parse_seeds(seeds_str: str) -> List[int]:
    seeds_str = (seeds_str or "").strip()
    if seeds_str == "":
        return []
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip() != ""]

def parse_admixture_ratios(ar_str: str) -> List[float]:
    ar_str = (ar_str or "").strip()
    if ar_str == "":
        return [0.5]
    return [float(x.strip()) for x in ar_str.split(",") if x.strip() != ""]

def read_input_list(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("#"):
                continue
            items.append(ln)
    return items

def collect_inputs(
    inputs: Optional[List[str]],
    input_list: Optional[str],
    input_dir: Optional[str],
    pattern: str,
    recursive: bool
) -> List[str]:
    collected: List[str] = []

    if inputs:
        collected += list(inputs)

    if input_list:
        collected += read_input_list(input_list)

    if input_dir:
        pat = os.path.join(input_dir, "**", pattern) if recursive else os.path.join(input_dir, pattern)
        collected += glob.glob(pat, recursive=recursive)

    # de-dupe while preserving order
    seen = set()
    out = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def _split_round_robin(items: List[str], k: int) -> List[List[str]]:
    buckets = [[] for _ in range(k)]
    for i, x in enumerate(items):
        buckets[i % k].append(x)
    return buckets

# -------------------------------
# Model state (lazy + cached) - one per process
# -------------------------------
class ModelState:
    def __init__(self, pool_path=None, rvq_cfg=None, rvq_en=None, rvq_fr=None,
                 bw_cfg=None, bw_model=None, bw_rvq=None,
                 aligner_model_path=None, f0_predictor_model_path=None,
                 duration_predictor_model_path=None, aw_cfg=None, aw_model=None,
                 lang="eng"):
        pool_path = pool_path or os.path.join(MODELS_DIR, "POOL/english")
        self.pool = nps.Pool.load(pool_path)

        self.WW = RVQFasterWhisperWrapper(
            config_path=rvq_cfg or os.path.join(MODELS_DIR, "RVQWhisper/config.yaml"),
            rvq_model_path_en=rvq_en or os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_en.pth"),
            rvq_model_path_fr=rvq_fr or os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_fr.pth")
        )

        self.BW = BigVGANWrapper(
            config_path=bw_cfg or os.path.join(MODELS_DIR, "BigVGAN/config.json"),
            model_path=bw_model or os.path.join(MODELS_DIR, "BigVGAN/generator_english"),
            rvq_config_path=rvq_cfg or os.path.join(MODELS_DIR, "RVQWhisper/config.yaml"),
            rvq_model_path=bw_rvq or os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_en.pth")
        )

        self.ALW = AlignerWrapper(
            aligner_model_path=aligner_model_path or os.path.join(MODELS_DIR, "Aligner/aligner_english.pth"),
            rvq_config_path=rvq_cfg or os.path.join(MODELS_DIR, "RVQWhisper/config.yaml"),
            rvq_model_path=rvq_en or os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_en.pth"),
            f0_predictor_model_path=f0_predictor_model_path or "aligner/src/f0_predictor.pth",
            duration_predictor_model_path=duration_predictor_model_path or "aligner/src/dur_predictor.pt",
            lang=lang
        )

        self.AW = AudioLMWrapper(
            config_file=aw_cfg or os.path.join(MODELS_DIR, "AudioLM/config.yaml"),
            model_path=aw_model or os.path.join(MODELS_DIR, "AudioLM/audiolm_english.pt")
        )

@lru_cache(maxsize=1)
def get_model_state() -> ModelState:
    return ModelState()

# Speaker extractor lazy load (used for original-speaker branch)
@lru_cache(maxsize=1)
def get_original_speaker_extractor():
    from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def extract_original_xvector(input_wav_path: str) -> np.ndarray:
    signal, fs = torchaudio.load(input_wav_path)
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
    with _maybe_lock():
        xvec = get_original_speaker_extractor().encode_batch(signal).squeeze().detach().cpu().numpy()
    return xvec

# -------------------------------
# Serializable session metadata
# -------------------------------
@dataclass
class SpeakerInfo:
    kind: str  # "original" or "p2"
    xvector: Optional[np.ndarray] = None
    seed: Optional[int] = None
    gender: Optional[str] = None
    spk_cluster: Optional[str] = None
    n_speaker: Optional[int] = None

@dataclass
class ResultEntry:
    wav_path: str
    sample_rate: int
    word_timestamps: List[Tuple[str, float, float]]
    npz_path: Optional[str]
    speaker: SpeakerInfo

class PipelineSession:
    def __init__(self):
        self.results: Dict[str, ResultEntry] = {}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PipelineSession":
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_results(self) -> List[str]:
        return list(self.results.keys())

# -------------------------------
# Core pipeline functions
# -------------------------------
def compute_common_features(input_filepath: str, predict_pitch: bool):
    ms = get_model_state()
    timing_common: Dict[str, float] = {}

    # Pitch extraction
    t = time.perf_counter()
    pitch = np.squeeze(pitch_extract(input_filepath))
    timing_common["pitch extraction"] = time.perf_counter() - t

    # Bottleneck computation
    t = time.perf_counter()
    with _maybe_lock():
        tokens = ms.WW.compute_bottleneck(input_filepath, vectorize=False)
    timing_common["bottleneck computation"] = time.perf_counter() - t

    # Transcription
    t = time.perf_counter()
    with _maybe_lock():
        segments_init, info = ms.WW.get_transcription(input_filepath, word_timestamps=True)
    segments = merge_segments(segments_init, 30, 70)
    timing_common["transcription"] = time.perf_counter() - t

    # Interpolate pitch to match tokens length
    x_old = np.linspace(0, 1, pitch.shape[0]) if pitch.shape[0] > 1 else np.array([0.0, 1.0])
    x_new = np.linspace(0, 1, tokens.shape[0])
    pitch = np.interp(x_new, x_old, pitch)

    if predict_pitch:
        pass

    if isinstance(tokens, list):
        tokens = tokens[0]

    # Articulatory features extraction
    t = time.perf_counter()
    with _maybe_lock():
        artics_features_segments, segments_alignment, word_timestamps = ms.ALW.get_articulatory_features(
            tokens, segments, info, return_word_timestamps=True
        )
    timing_common["articulatory features extraction"] = time.perf_counter() - t

    # Segment token generation
    t = time.perf_counter()
    with _maybe_lock():
        segment_tokens_syn = ms.AW.generate(artics_features_segments)
    timing_common["segment token generation"] = time.perf_counter() - t

    tokens_syn = tokens.copy()
    for i in range(len(segment_tokens_syn)):
        this_syn_len = min(segment_tokens_syn[i].shape[0], segments_alignment[i][1] - segments_alignment[i][0])
        tokens_syn[segments_alignment[i][0]:(segments_alignment[i][0] + this_syn_len)] = segment_tokens_syn[i][:this_syn_len]

    common = {
        "pitch": pitch,
        "tokens": tokens,
        "tokens_syn": tokens_syn,
        "word_timestamps": word_timestamps
    }
    return common, timing_common

def seed_specific_anonymization(
    common_data: Dict[str, Any],
    predict_pitch: bool,
    admixture_ratio: float,
    gender: Optional[str],
    spk_cluster: str,
    n_speaker: int,
    seed: Optional[int],
    db: float,
    pitch_f: float
):
    ms = get_model_state()
    timing_seed: Dict[str, float] = {}

    t = time.perf_counter()
    seed_used = seed if seed is not None else random.randint(1, 2**31 - 1)
    gender_arg = gender if gender not in ("", None) else None

    with _maybe_lock():
        pseudospeaker = nps.generate_pseudospeaker(
            ms.pool,
            n_speakers=n_speaker,
            gender=gender_arg,
            criterion=spk_cluster if spk_cluster != "None" else None,
            seed=seed_used
        )
    timing_seed["pseudospeaker generation"] = time.perf_counter() - t

    t = time.perf_counter()
    with _maybe_lock():
        final_bottleneck, _corr = ms.BW.admixture(
            common_data["tokens"], common_data["tokens_syn"], admixture_ratio, block_hallucination=False
        )
    timing_seed["admixture"] = time.perf_counter() - t

    t = time.perf_counter()
    pitch_seed = pseudospeaker.convert_pitch(common_data["pitch"])
    timing_seed["pitch conversion"] = time.perf_counter() - t

    if not predict_pitch:
        t = time.perf_counter()
        with _maybe_lock():
            pitch_seed = ms.BW.f0_transformation(pitch_seed, a=pitch_f, dB=db)
        timing_seed["f0 transformation"] = time.perf_counter() - t

    t = time.perf_counter()
    with _maybe_lock():
        array, sample_rate = ms.BW.synthesize(final_bottleneck, pseudospeaker.xvector, pitch_seed, chunk_size=100)
    timing_seed["synthesis"] = time.perf_counter() - t

    return array, sample_rate, timing_seed, final_bottleneck, pitch_seed, seed_used, pseudospeaker.xvector

def write_replacement_npz(npz_path: str, final_bottleneck: np.ndarray, pitch: np.ndarray):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, final_bottleneck=final_bottleneck, pitch=pitch)

def load_replacement_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["final_bottleneck"], data["pitch"]

# -------------------------------
# Worker - single file
# -------------------------------
def _format_adm_suffix(adm: float) -> str:
    return f"adm{adm:.3f}".replace(".", "p")

def _safe_outname(base: str, adm: float, seed_label: str, gender_label: str) -> str:
    adm_s = _format_adm_suffix(adm)
    return f"{base}.wav"

def process_file_worker(
    in_path: str,
    output_dir: str,
    predict_pitch: bool,
    admixture_ratios: List[float],
    gender: Optional[str],
    spk_cluster: str,
    n_speaker: int,
    seeds: List[int],
    db: float,
    pitch_f: float,
    store_replacement_data: bool,
    skip_existing: bool,
    include_original: bool
) -> Tuple[List[str], List[Dict[str, Any]]]:
    tmp_wav = None
    wav_input = in_path

    if not _is_wav(in_path):
        tmp_wav = decode_to_temp_wav_16k_mono(in_path)
        wav_input = tmp_wav
        print(f"  decoded -> {tmp_wav}")

    logs: List[str] = []
    print(f"START: {in_path}")
    results: List[Dict[str, Any]] = []

    base = safe_basename(in_path)
    name_base = os.path.splitext(base)[0]

    print(f"PROCESSING: {in_path}")

    if not os.path.exists(in_path):
        print(f"  !! missing file, skip")
        return logs, results

    common_data, _timing_common = compute_common_features(wav_input, predict_pitch)

    if include_original:
        original_xvector = extract_original_xvector(wav_input)
        for adm in admixture_ratios:
            result_name = f"anonymized_{name_base}_original_{_format_adm_suffix(adm)}"
            wav_out = os.path.join(output_dir, f"{result_name}.wav")
            npz_out = os.path.join(output_dir, "replacement_npz", f"{result_name}.npz") if store_replacement_data else None

            if skip_existing and os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                logs.append(f"  -> SKIP (exists): {wav_out}")
                results.append({
                    "wav_path": wav_out,
                    "sample_rate": 0,
                    "word_timestamps": common_data["word_timestamps"],
                    "npz_path": npz_out,
                    "speaker": {"kind": "original", "xvector": None}
                })
                continue

            with _maybe_lock():
                final_bottleneck, _corr = get_model_state().BW.admixture(
                    common_data["tokens"], common_data["tokens_syn"], adm, block_hallucination=False
                )
                array, sample_rate = get_model_state().BW.synthesize(
                    final_bottleneck, original_xvector, common_data["pitch"], chunk_size=100
                )
            sf.write(wav_out, array, sample_rate)

            if store_replacement_data and npz_out is not None:
                write_replacement_npz(npz_out, final_bottleneck, common_data["pitch"])

            results.append({
                "wav_path": wav_out,
                "sample_rate": sample_rate,
                "word_timestamps": common_data["word_timestamps"],
                "npz_path": npz_out,
                "speaker": {"kind": "original", "xvector": original_xvector}
            })
            logs.append(f"  -> WROTE: {wav_out}")

            del array, final_bottleneck

    seeds_to_iterate = seeds if len(seeds) > 0 else [None]

    for this_seed in seeds_to_iterate:
        seed = this_seed if this_seed is not None else random.randint(0, 999999)
        for adm in admixture_ratios:
            seed_label = f"seed{seed}" if seed is not None else f"rand{random.randint(0,999999)}"
            gender_label = gender if gender not in ("", None) else "rand"
            result_fname = _safe_outname(name_base, adm, seed_label, gender_label)
            wav_out = os.path.join(output_dir, result_fname)
            npz_out = os.path.join(output_dir, "replacement_npz", f"{os.path.splitext(result_fname)[0]}.npz") if store_replacement_data else None

            if skip_existing and os.path.exists(wav_out) and os.path.getsize(wav_out) > 0:
                logs.append(f"  -> SKIP (exists): {wav_out}")
                results.append({
                    "wav_path": wav_out,
                    "sample_rate": 0,
                    "word_timestamps": common_data["word_timestamps"],
                    "npz_path": npz_out,
                    "speaker": {"kind": "p2", "seed": seed, "gender": gender, "spk_cluster": spk_cluster, "n_speaker": n_speaker}
                })
                continue

            array, sample_rate, _timing_seed, final_bottleneck, pitch_seed, seed_used, xvector = seed_specific_anonymization(
                common_data=common_data,
                predict_pitch=predict_pitch,
                admixture_ratio=adm,
                gender=gender,
                spk_cluster=spk_cluster,
                n_speaker=n_speaker,
                seed=seed,
                db=db,
                pitch_f=pitch_f
            )

            sf.write(wav_out, array, sample_rate)

            if store_replacement_data and npz_out is not None:
                write_replacement_npz(npz_out, final_bottleneck, pitch_seed)

            results.append({
                "wav_path": wav_out,
                "sample_rate": sample_rate,
                "word_timestamps": common_data["word_timestamps"],
                "npz_path": npz_out,
                "speaker": {"kind": "p2", "seed": seed_used, "gender": gender or None, "spk_cluster": spk_cluster, "n_speaker": n_speaker, "xvector": xvector}
            })
            logs.append(f"  -> WROTE: {wav_out} (seed_used={seed_used})")

            del array, final_bottleneck, pitch_seed

    del common_data
    if tmp_wav is not None:
        os.remove(tmp_wav)
    return logs, results

# -------------------------------
# Chunk runner: ONE model process + many threads
# -------------------------------
def _process_chunk_in_one_model_process(
    chunk_paths: List[str],
    output_dir: str,
    predict_pitch: bool,
    admixture_ratios: List[float],
    gender: Optional[str],
    spk_cluster: str,
    n_speaker: int,
    seeds: List[int],
    db: float,
    pitch_f: float,
    store_replacement_data: bool,
    skip_existing: bool,
    include_original: bool,
    threads_per_model: int,
    verbose: bool
) -> Tuple[List[str], List[Dict[str, Any]], int]:
    # Ensure model loads once in this process

    print(f"[chunk] starting {len(chunk_paths)} files with threads={threads_per_model}", flush=True)

    _ = get_model_state()

    logs: List[str] = []
    flat_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=threads_per_model) as tpool:
        tfuts = {
            tpool.submit(
                process_file_worker,
                p,
                output_dir,
                predict_pitch,
                admixture_ratios,
                gender,
                spk_cluster,
                n_speaker,
                seeds,
                db,
                pitch_f,
                store_replacement_data,
                skip_existing,
                include_original
            ): p
            for p in chunk_paths
        }

        completed = 0
        total = len(chunk_paths)
        start = time.perf_counter()

        for tf in as_completed(tfuts):
            p = tfuts[tf]
            completed += 1
            elapsed = time.perf_counter() - start
            avg = elapsed / completed
            eta = (total - completed) * avg
            print(f"[chunk] {completed}/{total} {os.path.basename(p)} ETA:{eta:.1f}s", flush=True)

            file_logs, results = tf.result()
            logs.extend(file_logs)
            flat_results.extend(results)
            print("got the results")
            if verbose:
                for l in file_logs:
                    print("  " + l, flush=True)

    return logs, flat_results, len(chunk_paths)

# -------------------------------
# Orchestrator: N model processes in parallel
# -------------------------------
def process_batch_models_x_threads(
    input_paths: List[str],
    output_dir: str,
    predict_pitch: bool,
    admixture_ratios: List[float],
    gender: Optional[str],
    spk_cluster: str,
    n_speaker: int,
    seeds: List[int],
    db: float,
    pitch_f: float,
    store_replacement_data: bool,
    skip_existing: bool,
    include_original: bool,
    model_procs: int,
    threads_per_model: int,
    session: PipelineSession,
    verbose: bool
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    logs: List[str] = []

    total_files = len(input_paths)
    start_time = time.perf_counter()

    chunks = _split_round_robin(input_paths, model_procs)

    with ProcessPoolExecutor(max_workers=model_procs) as ppool:
        pfuts = []
        for chunk in chunks:
            if not chunk:
                continue
            pfuts.append(ppool.submit(
                _process_chunk_in_one_model_process,
                chunk,
                output_dir,
                predict_pitch,
                admixture_ratios,
                gender,
                spk_cluster,
                n_speaker,
                seeds,
                db,
                pitch_f,
                store_replacement_data,
                skip_existing,
                include_original,
                threads_per_model,
                verbose
            ))

        chunks_done = 0
        for pf in as_completed(pfuts):
            chunk_logs, flat_results, n_files = pf.result()
            logs.extend(chunk_logs)

            for r in flat_results:
                wav_path = r["wav_path"]
                npz_path = r["npz_path"]
                sample_rate = r["sample_rate"]
                wt = r["word_timestamps"]
                spk = r["speaker"]
                key = os.path.splitext(os.path.basename(wav_path))[0]

                session.results[key] = ResultEntry(
                    wav_path=wav_path,
                    sample_rate=sample_rate,
                    word_timestamps=wt,
                    npz_path=npz_path,
                    speaker=SpeakerInfo(
                        kind=spk.get("kind"),
                        xvector=spk.get("xvector"),
                        seed=spk.get("seed"),
                        gender=spk.get("gender"),
                        spk_cluster=spk.get("spk_cluster"),
                        n_speaker=spk.get("n_speaker")
                    )
                )

            chunks_done += 1
            elapsed = time.perf_counter() - start_time
            print(f"[main] chunks_done={chunks_done}/{len(pfuts)} files_total={total_files} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.perf_counter() - start_time
    logs.append(f"All done. Total elapsed: {total_elapsed:.2f}s")
    print(logs[-1], flush=True)
    return logs

# -------------------------------
# Replacement utilities (same logic, with lock around model calls)
# -------------------------------
def regenerate_pseudospeaker(info: SpeakerInfo):
    if info.kind != "p2":
        raise ValueError("regenerate_pseudospeaker called for non-p2 speaker")
    ms = get_model_state()
    with _maybe_lock():
        return nps.generate_pseudospeaker(
            ms.pool,
            n_speakers=int(info.n_speaker),
            gender=str(info.gender),
            criterion=info.spk_cluster if info.spk_cluster != "None" else None,
            seed=int(info.seed)
        )

def apply_replacements(
    session: PipelineSession,
    result_name: str,
    selected_occurrence_indices: List[int],
    replacement_texts: List[str],
    out_wav_path: str
) -> str:
    if result_name not in session.results:
        raise KeyError(f"Result '{result_name}' not found in session.")

    entry = session.results[result_name]
    if not entry.npz_path or not os.path.exists(entry.npz_path):
        raise RuntimeError(
            "Replacement data not found for this result. "
            "Run anonymization with --store-replacement-data so it writes .npz files."
        )

    ms = get_model_state()
    word_timestamps = entry.word_timestamps
    final_bottleneck, pitch = load_replacement_npz(entry.npz_path)
    speaker_info = entry.speaker

    if not selected_occurrence_indices:
        raise ValueError("No word indices selected.")

    selected_indices = sorted(set(int(i) for i in selected_occurrence_indices))
    groups: List[List[int]] = []
    current_group = [selected_indices[0]]
    for idx in selected_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    if len(replacement_texts) != len(groups):
        raise ValueError(
            f"Number of replacement texts ({len(replacement_texts)}) must match number of groups ({len(groups)})."
        )

    if speaker_info.kind == "original":
        nonzero_idxs = pitch.nonzero()
        src_nonzeros = pitch[nonzero_idxs]
        log_src = np.log(src_nonzeros)
        target_mean = float(np.mean(log_src))
        target_std = float(np.std(log_src))

    replacement_ops = []
    for group, rep_text in zip(groups, replacement_texts):
        first_entry = word_timestamps[group[0]]
        last_entry = word_timestamps[group[-1]]
        t_start = first_entry[1]
        t_end = last_entry[2]

        start_idx = round(t_start * 50)
        end_idx = round(t_end * 50)

        with _maybe_lock():
            artics_features, prlen, polen = ms.ALW.predict_artics(rep_text)
            gspitch = ms.ALW.predict_f0(artics_features)
            tokens_syn = ms.AW.generate([artics_features])[0]

        tokens_syn = tokens_syn[:artics_features.shape[0], :]
        tokens_syn = tokens_syn[prlen:-polen, :]
        gspitch = gspitch[prlen:-polen]

        replacement_ops.append((start_idx, end_idx, tokens_syn, gspitch))

    replacement_ops.sort(key=lambda x: x[0], reverse=True)

    pseudospeaker = None
    if speaker_info.kind == "p2":
        pseudospeaker = regenerate_pseudospeaker(speaker_info)

    for start_idx, end_idx, tokens_syn, gspitch in replacement_ops:
        with _maybe_lock():
            new_bottleneck = ms.BW.decode_from_codebook_indices(tokens_syn[None, ...])[0]

        if speaker_info.kind == "p2":
            new_pitch = pseudospeaker.convert_pitch(gspitch)
        else:
            nonzero_idxs = gspitch.nonzero()
            src_nonzeros = gspitch[nonzero_idxs]
            log_src = np.log(src_nonzeros)
            src_mean = float(np.mean(log_src))
            src_std = float(np.std(log_src))
            new_pitch = np.zeros_like(gspitch)
            new_pitch[nonzero_idxs] = np.exp(
                ((np.log(gspitch[nonzero_idxs]) - src_mean) / max(src_std, 1e-8)) * target_std + target_mean
            )

        final_bottleneck = np.concatenate(
            (final_bottleneck[:start_idx], new_bottleneck, final_bottleneck[end_idx:]),
            axis=0
        )
        pitch = np.concatenate((pitch[:start_idx], new_pitch, pitch[end_idx:]), axis=0)

    xvector = pseudospeaker.xvector if speaker_info.kind == "p2" else speaker_info.xvector

    with _maybe_lock():
        array, sample_rate = ms.BW.synthesize(final_bottleneck, xvector, pitch, chunk_size=100)
    sf.write(out_wav_path, array, sample_rate)

    write_replacement_npz(entry.npz_path, final_bottleneck, pitch)
    entry.wav_path = out_wav_path
    entry.sample_rate = sample_rate

    return f"Replacements applied. Wrote: {out_wav_path}"

# -------------------------------
# CLI
# -------------------------------
import torch.multiprocessing as mp

def main():
    # Safe default for CUDA + multiprocessing
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser("Batch voice anonymization pipeline (models x threads)")
    ap.add_argument("--inputs", nargs="*", default=None, help="Direct list of input paths.")
    ap.add_argument("--input-list", type=str, default=None, help="Text file: one audio path per line.")
    ap.add_argument("--input-dir", type=str, default=None, help="Directory containing audio files.")
    ap.add_argument("--glob", type=str, default="*.wav", help="Glob pattern for --input-dir (default: *.wav).")
    ap.add_argument("--recursive", action="store_true", help="Recurse in --input-dir.")

    ap.add_argument("--output-dir", required=True, help="Output directory (wavs).")
    ap.add_argument("--session-path", default=None, help="Session path (default: <output-dir>/session.pkl).")

    ap.add_argument("--predict-pitch", action="store_true")
    ap.add_argument("--admixture-ratio", type=str, default="0.5",
                    help='Comma-separated admixture ratios, e.g. "0,0.5,1.0"')
    ap.add_argument("--gender", type=str, default="", help='Gender "m", "f", or "" for random per-sample (default "").')
    ap.add_argument("--spk-cluster", choices=["cluster_dense", "cluster_sparse", "None"], default="None")
    ap.add_argument("--n-speaker", type=int, default=2)
    ap.add_argument("--seeds", type=str, default="", help='Comma-separated seeds (empty => random pseudospeaker per sample).')
    ap.add_argument("--db", type=float, default=0.0)
    ap.add_argument("--pitch-f", type=float, default=0.0)

    ap.add_argument("--store-replacement-data", action="store_true",
                    help="Store per-result .npz with final_bottleneck and pitch for later replacements.")
    ap.set_defaults(skip_existing=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                    help="Recompute even if output wav already exists.")
    ap.add_argument("--write-log", type=str, default=None, help="Write log to this file path.")

    ap.add_argument("--include-original", action="store_true",
                    help="Also produce anonymization using the original speaker embedding.")

    # NEW: models x threads
    ap.add_argument("--model-procs", type=int, default=1,
                    help="Number of model instances (processes). Each process loads ModelState once.")
    ap.add_argument("--threads-per-model", type=int, default=1,
                    help="Threads per model process (ThreadPool inside each process).")

    ap.add_argument("--verbose", action="store_true", help="Verbose per-file logging.")

    # Replacement utilities
    ap.add_argument("--print-words", type=str, default=None, help="After processing, print word indices for this result name.")
    ap.add_argument("--replace-result", type=str, default=None)
    ap.add_argument("--replace-indices", type=str, default=None,
                    help="Comma-separated word indices to replace (auto-grouped).")
    ap.add_argument("--replace-texts", type=str, default=None,
                    help="Comma-separated replacement texts (must match number of groups).")
    ap.add_argument("--replace-out", type=str, default=None,
                    help="Output wav path for replaced audio (default: <output-dir>/<result>_replaced.wav).")

    args = ap.parse_args()
    # breakpoint()
    input_paths = collect_inputs(args.inputs, args.input_list, args.input_dir, args.glob, args.recursive)
    if not input_paths:
        raise SystemExit("No inputs found. Use --inputs, --input-list, or --input-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    session_path = args.session_path or os.path.join(args.output_dir, "session.pkl")
    seeds = parse_seeds(args.seeds)
    admixture_ratios = parse_admixture_ratios(args.admixture_ratio)

    session = PipelineSession()

    logs = process_batch_models_x_threads(
        input_paths=input_paths,
        output_dir=args.output_dir,
        predict_pitch=args.predict_pitch,
        admixture_ratios=admixture_ratios,
        gender=args.gender if args.gender != "" else None,
        spk_cluster=args.spk_cluster,
        n_speaker=args.n_speaker,
        seeds=seeds,
        db=args.db,
        pitch_f=args.pitch_f,
        store_replacement_data=args.store_replacement_data,
        skip_existing=args.skip_existing,
        include_original=args.include_original,
        model_procs=args.model_procs,
        threads_per_model=args.threads_per_model,
        session=session,
        verbose=args.verbose
    )

    session.save(session_path)

    print("\nSummary:")
    for name, entry in session.results.items():
        print(f"  - {name}")
        print(f"    wav: {entry.wav_path}")
        if entry.npz_path:
            print(f"    npz: {entry.npz_path}")

    if args.write_log:
        with open(args.write_log, "w", encoding="utf-8") as f:
            f.write("\n".join(logs) + "\n")
        print(f"\nWrote log: {args.write_log}")

    if args.print_words:
        if args.print_words not in session.results:
            print(f"\n[print-words] Unknown result: {args.print_words}")
        else:
            entry = session.results[args.print_words]
            wt = entry.word_timestamps
            print(f"\nWords for result: {args.print_words}")
            print("Index | Word | Start(s) -> End(s)")
            print("-" * 44)
            for i, (w, s, e) in enumerate(wt[:200]):
                print(f"{i:5d} | {w} | {s:.2f} -> {e:.2f}")
            if len(wt) > 200:
                print(f"... (showing first 200 of {len(wt)})")

    if args.replace_result:
        session2 = PipelineSession.load(session_path)

        if args.replace_result not in session2.results:
            available = session2.list_results()
            raise SystemExit(
                f"--replace-result '{args.replace_result}' not found. "
                f"Use one of: {available[:10]}{'...' if len(available) > 10 else ''}"
            )

        if not args.replace_indices or not args.replace_texts:
            raise SystemExit("For replacement, provide both --replace-indices and --replace-texts.")

        indices = [int(x.strip()) for x in args.replace_indices.split(",") if x.strip() != ""]
        repl_texts = [s.strip() for s in args.replace_texts.split(",") if s.strip() != ""]
        out_wav = args.replace_out or os.path.join(args.output_dir, f"{args.replace_result}_replaced.wav")

        msg = apply_replacements(session2, args.replace_result, indices, repl_texts, out_wav)
        print("\n" + msg)

        session2.save(session_path)
        print(f"Updated session saved: {session_path}")

if __name__ == "__main__":
    main()
