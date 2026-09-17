import os

# Root of the released checkpoints and speaker pool (see README, "Model weights").
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
import io
import time
import json
import base64
import random
import pickle
import argparse
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt

# -------------------------------
# Your model wrappers and deps
# -------------------------------
from aligner.src import AlignerWrapper
from rvqwhisper.src.rvqwhisper import RVQFasterWhisperWrapper
from bigvgan import BigVGANWrapper

import sys
sys.path.insert(0, "./audiolm")
from audiolm import AudioLMWrapper

import pspi.pseudospeaker as nps
from pspi.pitch import extract as pitch_extract

from content_editing import apply_graft_replacement, words_to_prefix_suffix, FPS

from speechbrain.inference.speaker import EncoderClassifier


# -------------------------------
# Helpers
# -------------------------------
def merge_segments(segments, min_words=5, min_chars=70):
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


def save_pie_chart_png(times_dict: Dict[str, float], total_time: float, out_png_path: str):
    labels = list(times_dict.keys())
    sizes = list(times_dict.values())
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f"Total Time: {total_time:.2f} sec")
    fig.savefig(out_png_path, format="png", bbox_inches="tight")
    plt.close(fig)


def ensure_wav_path(input_path: str) -> str:
    # You can add format conversion here if needed.
    # For now, we assume input is a readable wav file.
    return input_path


def parse_seeds(seeds_str: str) -> List[int]:
    seeds_str = (seeds_str or "").strip()
    if seeds_str == "":
        return []
    return [int(s.strip()) for s in seeds_str.split(",") if s.strip() != ""]


def safe_basename(path: str) -> str:
    base = os.path.basename(path)
    base = base.replace(" ", "_")
    return base


# -------------------------------
# Model state (loaded once)
# -------------------------------
class ModelState:
    def __init__(self):
        self.pool = nps.Pool.load(os.path.join(MODELS_DIR, "POOL/english"))
        self.WW = RVQFasterWhisperWrapper(
            config_path=os.path.join(MODELS_DIR, "RVQWhisper/config.yaml"),
            rvq_model_path_en=os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_en.pth"),
            rvq_model_path_fr=os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_fr.pth")
        )
        self.BW = BigVGANWrapper(
            config_path=os.path.join(MODELS_DIR, "BigVGAN/config.json"),
            model_path=os.path.join(MODELS_DIR, "BigVGAN/generator_english"),
            rvq_config_path=os.path.join(MODELS_DIR, "RVQWhisper/config.yaml"),
            rvq_model_path=os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_en.pth")
        )
        self.ALW = AlignerWrapper(
            aligner_model_path=os.path.join(MODELS_DIR, "Aligner/aligner_english.pth"),
            rvq_config_path=os.path.join(MODELS_DIR, "RVQWhisper/config.yaml"),
            rvq_model_path=os.path.join(MODELS_DIR, "RVQWhisper/rvq_model_en.pth"),
            f0_predictor_model_path="aligner/src/f0_predictor.pth",
            duration_predictor_model_path="aligner/src/dur_predictor.pt",
            lang="eng"
        )
        self.AW = AudioLMWrapper(
            config_file=os.path.join(MODELS_DIR, "AudioLM/config.yaml"),
            model_path=os.path.join(MODELS_DIR, "AudioLM/audiolm_english.pt")
        )


model_state = ModelState()

# Speaker embedding extractor (ECAPA-TDNN)
original_speaker_extractor = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


# -------------------------------
# Core pipeline functions
# -------------------------------
def compute_common_features(
    input_filepath: str,
    predict_pitch: bool,
    admixture_ratio: float,
    gender: str,
    spk_cluster: str,
    n_speaker: int,
    db: float,
    pitch_f: float
):
    timing_common: Dict[str, float] = {}

    # Pitch extraction
    t = time.perf_counter()
    pitch = np.squeeze(pitch_extract(input_filepath))
    timing_common["pitch extraction"] = time.perf_counter() - t

    # Bottleneck computation
    t = time.perf_counter()
    tokens = model_state.WW.compute_bottleneck(input_filepath, vectorize=False)
    timing_common["bottleneck computation"] = time.perf_counter() - t

    # Transcription
    t = time.perf_counter()
    segments_init, info = model_state.WW.get_transcription(input_filepath, word_timestamps=True)
    segments = merge_segments(segments_init, 30, 70)
    timing_common["transcription"] = time.perf_counter() - t

    # Interpolate pitch to match tokens length
    x_old = np.linspace(0, 1, pitch.shape[0])
    x_new = np.linspace(0, 1, tokens.shape[0])
    pitch = np.interp(x_new, x_old, pitch)

    if predict_pitch:
        # Placeholder for pitch prediction logic (kept same as your original)
        pass

    if isinstance(tokens, list):
        tokens = tokens[0]

    # Articulatory features extraction
    t = time.perf_counter()
    artics_features_segments, segments_alignment, word_timestamps = model_state.ALW.get_articulatory_features(
        tokens, segments, info, return_word_timestamps=True
    )
    timing_common["articulatory features extraction"] = time.perf_counter() - t

    # Segment token generation
    t = time.perf_counter()
    segment_tokens_syn = model_state.AW.generate(artics_features_segments)
    timing_common["segment token generation"] = time.perf_counter() - t

    tokens_syn = tokens.copy()
    for i in range(len(segment_tokens_syn)):
        this_syn_len = min(segment_tokens_syn[i].shape[0], segments_alignment[i][1] - segments_alignment[i][0])
        tokens_syn[segments_alignment[i][0]:(segments_alignment[i][0] + this_syn_len)] = segment_tokens_syn[i][:this_syn_len]

    return {"pitch": pitch, "tokens": tokens, "tokens_syn": tokens_syn, "word_timestamps": word_timestamps}, timing_common


def extract_original_xvector(input_wav_path: str) -> np.ndarray:
    signal, fs = torchaudio.load(input_wav_path)

    # Resample to 16 kHz if needed
    if fs != 16000:
        signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)

    xvec = original_speaker_extractor.encode_batch(signal).squeeze().detach().cpu().numpy()
    return xvec


def seed_specific_anonymization(
    common_data: Dict[str, Any],
    predict_pitch: bool,
    admixture_ratio: float,
    gender: str,
    spk_cluster: str,
    n_speaker: int,
    seed: int,
    db: float,
    pitch_f: float
):
    timing_seed: Dict[str, float] = {}

    t = time.perf_counter()
    pseudospeaker = nps.generate_pseudospeaker(
        model_state.pool,
        n_speakers=n_speaker,
        gender=gender if gender in ("m", "f") else None,
        criterion=spk_cluster if spk_cluster != "None" else None,
        seed=seed
    )
    timing_seed["pseudospeaker generation"] = time.perf_counter() - t

    # Seed the per-frame admixture draw and the F0 noise so a seed gives a repeatable output.
    np.random.seed(seed)

    t = time.perf_counter()
    final_bottleneck, _corr = model_state.BW.admixture(
        common_data["tokens"], common_data["tokens_syn"], admixture_ratio, block_hallucination=True
    )
    timing_seed["admixture"] = time.perf_counter() - t

    t = time.perf_counter()
    pitch_seed = pseudospeaker.convert_pitch(common_data["pitch"])
    timing_seed["pitch conversion"] = time.perf_counter() - t

    if not predict_pitch:
        t = time.perf_counter()
        pitch_seed = model_state.BW.f0_transformation(pitch_seed, a=pitch_f, dB=db)
        timing_seed["f0 transformation"] = time.perf_counter() - t

    t = time.perf_counter()
    array, sample_rate = model_state.BW.synthesize(
        final_bottleneck, pseudospeaker.xvector, pitch_seed, chunk_size=100
    )
    timing_seed["synthesis"] = time.perf_counter() - t

    return array, sample_rate, timing_seed, final_bottleneck, pitch_seed, pseudospeaker


# -------------------------------
# Session storage (for replacements)
# -------------------------------
# We store for each result_name:
#   word_timestamps, final_bottleneck, pitch, speaker_repr, is_p2
# where:
#   speaker_repr is either:
#     - pseudospeaker object (if is_p2=True)
#     - original xvector np.ndarray (if is_p2=False)
@dataclass
class ResultEntry:
    wav_path: str
    chart_path: str
    sample_rate: int
    word_timestamps: List[Tuple[str, float, float]]  # (word, start, end)
    final_bottleneck: np.ndarray
    pitch: np.ndarray
    speaker_repr: Any
    is_p2: bool


class PipelineSession:
    def __init__(self):
        self.results: Dict[str, ResultEntry] = {}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PipelineSession":
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_results(self) -> List[str]:
        return list(self.results.keys())


# -------------------------------
# Replacement logic (same behavior as your Gradio version)
# -------------------------------
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
    word_timestamps = entry.word_timestamps
    final_bottleneck = entry.final_bottleneck
    pitch = entry.pitch
    speaker_repr = entry.speaker_repr
    is_p2 = entry.is_p2

    if not selected_occurrence_indices:
        raise ValueError("No word indices selected.")

    # Group consecutive indices
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

    # Build replacement operations
    replacement_ops = []
    for group, rep_text in zip(groups, replacement_texts):
        first_entry = word_timestamps[group[0]]
        last_entry = word_timestamps[group[-1]]
        start_idx = round(first_entry[1] * FPS)
        end_idx = round(last_entry[2] * FPS)
        prefix, suffix = words_to_prefix_suffix(word_timestamps, group)
        replacement_ops.append((start_idx, end_idx, rep_text, prefix, suffix))

    replacement_ops.sort(key=lambda x: x[0], reverse=True)

    if speaker_repr is not None and not is_p2:
        nonzero_idxs = pitch.nonzero()
        src_nonzeros = pitch[nonzero_idxs]
        log_src = np.log(src_nonzeros)
        target_mean = np.mean(log_src)
        target_std = np.std(log_src)

        def convert_pitch(gspitch):
            out = np.zeros_like(gspitch)
            nz = gspitch.nonzero()
            if nz[0].size == 0:
                return out
            ls = np.log(gspitch[nz])
            sm, ss = np.mean(ls), np.std(ls)
            out[nz] = np.exp(((ls - sm) / max(ss, 1e-8)) * target_std + target_mean)
            return out
    elif speaker_repr is not None and is_p2:
        convert_pitch = speaker_repr.convert_pitch
    else:
        convert_pitch = lambda x: x

    for start_idx, end_idx, rep_text, prefix, suffix in replacement_ops:
        final_bottleneck, pitch = apply_graft_replacement(
            final_bottleneck, pitch, start_idx, end_idx,
            model_state.ALW, model_state.AW, model_state.BW,
            rep_text, prefix, suffix, convert_pitch,
        )

    # Synthesize
    xvector = speaker_repr.xvector if is_p2 else speaker_repr
    array, sample_rate = model_state.BW.synthesize(final_bottleneck, xvector, pitch, chunk_size=100)

    # Write output
    sf.write(out_wav_path, array, sample_rate)

    # Update session entry
    entry.final_bottleneck = final_bottleneck
    entry.pitch = pitch
    entry.sample_rate = sample_rate
    entry.wav_path = out_wav_path

    return f"Replacements applied. Wrote: {out_wav_path}"


# -------------------------------
# Main processing (no UI)
# -------------------------------
def process_inputs(
    input_paths: List[str],
    output_dir: str,
    predict_pitch: bool,
    admixture_ratio: float,
    gender: str,
    spk_cluster: str,
    n_speaker: int,
    seeds: List[int],
    db: float,
    pitch_f: float,
    session: PipelineSession
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    logs: List[str] = []

    for idx, in_path in enumerate(input_paths):
        in_path = ensure_wav_path(in_path)
        base = safe_basename(in_path)

        logs.append(f"[{idx+1}/{len(input_paths)}] Processing: {in_path}")

        common_data, timing_common = compute_common_features(
            in_path, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, db, pitch_f
        )

        if len(seeds) == 0:
            # Original speaker branch
            t0 = time.perf_counter()
            original_xvector = extract_original_xvector(in_path)
            timing_original = {"embedding extraction": time.perf_counter() - t0}

            t0 = time.perf_counter()
            final_bottleneck, _corr = model_state.BW.admixture(
                common_data["tokens"], common_data["tokens_syn"], admixture_ratio, block_hallucination=True
            )
            timing_original["admixture"] = time.perf_counter() - t0

            final_timing = {**timing_common, **timing_original}
            total_time = sum(final_timing.values())

            array, sample_rate = model_state.BW.synthesize(
                final_bottleneck, original_xvector, common_data["pitch"], chunk_size=100
            )

            result_name = f"anonymized_{base}_original"
            wav_out = os.path.join(output_dir, f"{result_name}.wav")
            png_out = os.path.join(output_dir, f"{result_name}_timing.png")

            sf.write(wav_out, array, sample_rate)
            save_pie_chart_png(final_timing, total_time, png_out)

            session.results[result_name] = ResultEntry(
                wav_path=wav_out,
                chart_path=png_out,
                sample_rate=sample_rate,
                word_timestamps=common_data["word_timestamps"],
                final_bottleneck=final_bottleneck,
                pitch=common_data["pitch"],
                speaker_repr=original_xvector,
                is_p2=False
            )

            logs.append(f"  -> {result_name} ({total_time:.2f}s)")
        else:
            # Pseudospeaker branch per seed
            for seed in seeds:
                array, sample_rate, timing_seed, final_bottleneck, pitch_seed, pseudospeaker = seed_specific_anonymization(
                    common_data, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, seed, db, pitch_f
                )

                final_timing = {**timing_common, **timing_seed}
                total_time = sum(final_timing.values())

                result_name = f"anonymized_{base}_seed_{seed}"
                wav_out = os.path.join(output_dir, f"{result_name}.wav")
                png_out = os.path.join(output_dir, f"{result_name}_timing.png")

                sf.write(wav_out, array, sample_rate)
                save_pie_chart_png(final_timing, total_time, png_out)

                session.results[result_name] = ResultEntry(
                    wav_path=wav_out,
                    chart_path=png_out,
                    sample_rate=sample_rate,
                    word_timestamps=common_data["word_timestamps"],
                    final_bottleneck=final_bottleneck,
                    pitch=pitch_seed,
                    speaker_repr=pseudospeaker,
                    is_p2=True
                )

                logs.append(f"  -> {result_name} ({total_time:.2f}s)")

    return logs


def print_words(session: PipelineSession, result_name: str, max_words: int = 200):
    entry = session.results[result_name]
    wt = entry.word_timestamps
    print(f"\nWords for result: {result_name}")
    print("Index | Word | Start(s) -> End(s)")
    print("-" * 40)
    for i, (w, s, e) in enumerate(wt[:max_words]):
        print(f"{i:5d} | {w} | {s:.2f} -> {e:.2f}")
    if len(wt) > max_words:
        print(f"... (showing first {max_words} of {len(wt)})")


def parse_groups_indices(groups_str: str) -> List[int]:
    """
    Accepts:
      - "3,4,5,10,11" -> flat list (will be auto-grouped by consecutiveness)
    """
    groups_str = (groups_str or "").strip()
    if not groups_str:
        return []
    return [int(x.strip()) for x in groups_str.split(",") if x.strip() != ""]


def parse_replacement_texts(repl_str: str) -> List[str]:
    """
    Comma-separated replacement texts: "John Doe, Berlin"
    (Must match number of consecutive-groups formed from indices.)
    """
    repl_str = (repl_str or "").strip()
    if repl_str == "":
        return []
    return [s.strip() for s in repl_str.split(",") if s.strip() != ""]


def main():
    ap = argparse.ArgumentParser("Voice anonymization pipeline (no Gradio)")

    ap.add_argument("--inputs", nargs="+", required=True, help="Input WAV file paths (one or many).")
    ap.add_argument("--output-dir", required=True, help="Directory to write anonymized wavs + timing charts.")
    ap.add_argument("--session-path", default=None, help="Where to save session.pkl (default: <output-dir>/session.pkl).")

    ap.add_argument("--predict-pitch", action="store_true", help="Predict pitch (placeholder behavior same as original).")
    ap.add_argument("--admixture-ratio", type=float, default=0.5)
    ap.add_argument("--gender", choices=["m", "f", "any"], default="m",
                    help="Pseudospeaker pool gender. \"any\" samples independently of source gender (paper setting).")
    ap.add_argument("--spk-cluster", choices=["cluster_dense", "cluster_sparse", "None"], default="None")
    ap.add_argument("--n-speaker", type=int, default=2)
    ap.add_argument("--seeds", type=str, default="52", help="Comma-separated seeds. Empty => original speaker branch.")
    ap.add_argument("--db", type=float, default=0.0)
    ap.add_argument("--pitch-f", type=float, default=0.0)

    # Replacement step (optional)
    ap.add_argument("--replace-result", type=str, default=None, help="Result name to apply replacements to.")
    ap.add_argument("--replace-indices", type=str, default=None, help="Comma-separated word indices to replace.")
    ap.add_argument("--replace-texts", type=str, default=None, help="Comma-separated replacement texts.")
    ap.add_argument("--replace-out", type=str, default=None, help="Output wav path for replaced audio.")

    # Utility
    ap.add_argument("--print-words", type=str, default=None, help="Print word list for a given result name (after processing).")

    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    session = PipelineSession()

    logs = process_inputs(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        predict_pitch=args.predict_pitch,
        admixture_ratio=args.admixture_ratio,
        gender=args.gender,
        spk_cluster=args.spk_cluster,
        n_speaker=args.n_speaker,
        seeds=seeds,
        db=args.db,
        pitch_f=args.pitch_f,
        session=session
    )

    session_path = args.session_path or os.path.join(args.output_dir, "session.pkl")
    session.save(session_path)

    print("\n".join(logs))
    print(f"\nSaved session: {session_path}")
    print("\nResults:")
    for name in session.list_results():
        e = session.results[name]
        print(f"  - {name}")
        print(f"    wav:   {e.wav_path}")
        print(f"    chart: {e.chart_path}")

    if args.print_words:
        if args.print_words not in session.results:
            print(f"\n[print-words] Unknown result: {args.print_words}")
        else:
            print_words(session, args.print_words)

    # Optional replacement step
    if args.replace_result:
        if args.replace_result not in session.results:
            raise SystemExit(f"--replace-result '{args.replace_result}' not found. Use one of: {session.list_results()}")

        if not args.replace_indices or not args.replace_texts:
            raise SystemExit("For replacement, provide both --replace-indices and --replace-texts.")

        indices = parse_groups_indices(args.replace_indices)
        repl_texts = parse_replacement_texts(args.replace_texts)

        out_wav = args.replace_out or os.path.join(args.output_dir, f"{args.replace_result}_replaced.wav")
        msg = apply_replacements(session, args.replace_result, indices, repl_texts, out_wav)
        print("\n" + msg)

        # Persist updated session
        session.save(session_path)
        print(f"Updated session saved: {session_path}")


if __name__ == "__main__":
    main()


"""Usage examples

1) Run anonymization (seeded pseudospeakers)

python static_infer.py \
  --inputs a.wav b.wav \
  --output-dir outputs \
  --seeds "52,123" \
  --gender m \
  --spk-cluster None \
  --n-speaker 2 \
  --admixture-ratio 0.5


2) Use original speaker branch (no seeds)

python static_infer.py \
  --inputs a.wav \
  --output-dir outputs \
  --seeds ""


3) Print word indices for a result

python static_infer.py \
  --inputs a.wav \
  --output-dir outputs \
  --seeds "52" \
  --print-words anonymized_a.wav_seed_52


4) Replace words (indices auto-grouped by consecutiveness)
Example: indices 10,11,12 (group 1) and 30,31 (group 2) ⇒ you must provide two replacement texts:

python static_infer.py \
  --inputs a.wav \
  --output-dir outputs \
  --seeds "52" \
  --replace-result anonymized_a.wav_seed_52 \
  --replace-indices "10,11,12,30,31" \
  --replace-texts "John Doe, Berlin" \
  --replace-out outputs/a_replaced.wav
"""
