#!/usr/bin/env python3
"""End-to-end check of a model release against the code in this repository.

Run from the repository root on a CUDA machine:

    export MODELS_DIR=/path/to/models
    export PYTHONPATH=".:./audiolm:./bigvgan/src"
    python scripts/verify_release.py --inputs-dir testdata --out-dir verify_out

`--inputs-dir` must contain `<name>.wav` files, each with an optional
`<name>.json` holding {"text": ..., "gender": "m"|"f"} for WER checks.

Every check is recorded as PASS, WARN or FAIL in `<out-dir>/report.json` and
`<out-dir>/report.md`. The script exits with status 1 if any check fails.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import soundfile as sf

MODELS_DIR = os.environ.get("MODELS_DIR", "models")

EXPECTED_FILES = [
    "RVQWhisper/config.yaml",
    "RVQWhisper/rvq_model_en.pth",
    "BigVGAN/config.json",
    "BigVGAN/generator_english",
    "Aligner/aligner_english.pth",
    "Aligner/f0_predictor.pth",
    "Aligner/dur_predictor.pt",
    "AudioLM/config.yaml",
    "AudioLM/audiolm_english.pt",
    "POOL/english/spk2gender",
    "POOL/english/xvectors/spk_xvector.ark",
    "POOL/english/cluster/cc_idx.pkl",
    "POOL/english/cluster/labels.pkl",
    "POOL/english/cluster/drank.pkl",
    "POOL/english/yaapt_pitch/pitch.json",
]
FRENCH_FILES = [
    "RVQWhisper/rvq_model_fr.pth",
    "BigVGAN/generator_french",
    "Aligner/aligner_french.pth",
    "AudioLM/audiolm_french.pt",
    "POOL/french/spk2gender",
    "POOL/french/xvectors/spk_xvector.ark",
]

RESULTS = []
RUN_START = time.time()
# One row per timed event: {"group", "stage", "seconds", "audio_seconds", "file", "cold"}.
TIMINGS = []


def gpu_reset():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_gb():
    import torch
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def timed(group, stage, seconds, audio_seconds=None, file=None, cold=False):
    TIMINGS.append({"group": group, "stage": stage, "seconds": float(seconds),
                    "audio_seconds": audio_seconds, "file": file, "cold": cold})


def record(name, status, detail="", **metrics):
    RESULTS.append({"check": name, "status": status, "detail": detail, **metrics})
    extra = " ".join(f"{k}={v}" for k, v in metrics.items())
    print(f"[{status}] {name}: {detail} {extra}".rstrip(), flush=True)


def check(name):
    """Run the decorated function as one check; exceptions become FAIL."""
    def deco(fn):
        def wrapper(*a, **kw):
            t0 = time.time()
            try:
                return fn(*a, **kw)
            except Exception as e:  # noqa: BLE001
                record(name, "FAIL", f"{type(e).__name__}: {e}",
                       seconds=round(time.time() - t0, 1))
                traceback.print_exc()
                return None
        return wrapper
    return deco


def normalize_text(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z' ]", " ", s)
    return s.split()


def wer(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def audio_sanity(path, ref_path=None):
    """Return a list of problems with a synthesized file (empty list means OK)."""
    x, sr = sf.read(path, dtype="float32")
    problems = []
    if sr != 16000:
        problems.append(f"sample rate {sr}")
    if x.ndim != 1:
        problems.append(f"{x.ndim} dims")
    if not np.isfinite(x).all():
        problems.append("non-finite samples")
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    if rms < 1e-3:
        problems.append(f"near silent (rms {rms:.5f})")
    if x.size and np.mean(np.abs(x) > 0.999) > 0.01:
        problems.append("clipping")
    if ref_path:
        ref = sf.info(ref_path)
        ratio = (len(x) / sr) / ref.duration
        if not 0.9 < ratio < 1.1:
            problems.append(f"duration ratio {ratio:.2f}")
    return problems


# ---------------------------------------------------------------- checks

@check("files present")
def check_files(include_french):
    names = EXPECTED_FILES + (FRENCH_FILES if include_french else [])
    missing = [f for f in names if not os.path.isfile(os.path.join(MODELS_DIR, f))]
    if missing:
        record("files present", "FAIL", f"missing: {missing}")
    else:
        record("files present", "PASS", f"{len(names)} files under {MODELS_DIR}")


@check("checksums")
def check_checksums():
    sums = os.path.join(MODELS_DIR, "SHA256SUMS")
    if not os.path.isfile(sums):
        record("checksums", "WARN", "no SHA256SUMS file")
        return
    bad = []
    lines = [l.split(maxsplit=1) for l in open(sums) if l.strip()]
    for digest, name in lines:
        h = hashlib.sha256()
        with open(os.path.join(MODELS_DIR, name.strip()), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            bad.append(name.strip())
    record("checksums", "FAIL" if bad else "PASS",
           f"mismatch: {bad}" if bad else f"{len(lines)} files match")


@check("speaker pool")
def check_pool():
    import pspi.pseudospeaker as nps
    pool = nps.Pool.load(os.path.join(MODELS_DIR, "POOL/english"))
    dims = {v.shape[0] for v in pool.xvectors.values()}
    genders = {g: len(pool.get_speakers_of_gender(g)) for g in ("m", "f")}
    ok = dims == {192} and all(genders.values())
    record("speaker pool", "PASS" if ok else "FAIL",
           f"{len(pool.xvectors)} speakers, dims {dims}, per gender {genders}")
    for crit in ("cluster_sparse", "cluster_dense", None):
        ps = nps.generate_pseudospeaker(pool, n_speakers=2, gender=None, criterion=crit, seed=52)
        if ps.xvector.shape != (192,) or not np.isfinite(ps.xvector).all():
            raise ValueError(f"bad pseudospeaker for criterion {crit}")
    record("pseudospeaker strategies", "PASS", "random, sparse and dense all produce 192-d x-vectors")
    # With top_n = 10 and fewer clusters per gender, sparse and dense draw from the
    # same shortlist. The published samples rely on this, so it is reported, not failed.
    n_clusters = {g: sum(pool.clustering_info["gender"][c] == g for c in pool.clustering_info["density_rank"])
                  for g in ("m", "f")}
    same = 0
    for seed in range(20):
        for g in ("m", "f"):
            a = nps.generate_pseudospeaker(pool, n_speakers=2, gender=g, criterion="cluster_sparse", seed=seed)
            b = nps.generate_pseudospeaker(pool, n_speakers=2, gender=g, criterion="cluster_dense", seed=seed)
            same += bool(np.allclose(a.xvector, b.xvector))
    record("cluster strategies", "PASS",
           f"clusters per gender {n_clusters}; sparse and dense gave the same pseudospeaker "
           f"for {same} of 40 seed/gender pairs (expected 40 while every gender has <= 10 clusters)")


@check("french checkpoints")
def check_french():
    import torch
    t_fr = time.perf_counter()
    from aligner.src import AlignerWrapper
    from audiolm import AudioLMWrapper
    from bigvgan import BigVGANWrapper
    from rvqwhisper.src.rvqwhisper import RVQFasterWhisperWrapper

    m = MODELS_DIR
    ww = RVQFasterWhisperWrapper(
        config_path=f"{m}/RVQWhisper/config.yaml",
        rvq_model_path_en=f"{m}/RVQWhisper/rvq_model_en.pth",
        rvq_model_path_fr=f"{m}/RVQWhisper/rvq_model_fr.pth")
    assert "french" in ww.rvq_model, "French RVQ codebooks not loaded"
    del ww
    bw = BigVGANWrapper(
        config_path=f"{m}/BigVGAN/config.json", model_path=f"{m}/BigVGAN/generator_french",
        rvq_config_path=f"{m}/RVQWhisper/config.yaml", rvq_model_path=f"{m}/RVQWhisper/rvq_model_fr.pth")
    del bw
    aw = AudioLMWrapper(config_file=f"{m}/AudioLM/config.yaml", model_path=f"{m}/AudioLM/audiolm_french.pt")
    del aw
    alw = AlignerWrapper(
        aligner_model_path=f"{m}/Aligner/aligner_french.pth",
        rvq_config_path=f"{m}/RVQWhisper/config.yaml",
        rvq_model_path=f"{m}/RVQWhisper/rvq_model_fr.pth", lang="fra")
    del alw
    import pspi.pseudospeaker as nps
    fr_pool = nps.Pool.load(f"{m}/POOL/french")
    assert {v.shape[0] for v in fr_pool.xvectors.values()} == {192}, "French pool x-vectors are not 192-d"
    torch.cuda.empty_cache()
    timed("startup", "load French checkpoints (all four)", time.perf_counter() - t_fr)
    record("french checkpoints", "PASS", f"RVQ, BigVGAN, AudioLM, aligner and pool ({len(fr_pool.xvectors)} speakers) load")


def ecapa_similarity(si, a, b):
    ea, eb = si.extract_original_xvector(a), si.extract_original_xvector(b)
    return float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb)))


def transcribe(si, path):
    segments, _ = si.model_state.WW.get_transcription(path, word_timestamps=False)
    return " ".join(s.text.strip() for s in segments)


@check("pipeline")
def check_pipeline(si, inputs, out_dir):
    """Paper operating points on every input: sanity, anonymization strength, WER."""
    configs = [
        ("beta0.0", dict(admixture_ratio=0.0, pitch_f=0.0, db=0.0)),
        ("beta0.3", dict(admixture_ratio=0.3, pitch_f=0.0, db=0.0)),
        ("beta0.7", dict(admixture_ratio=0.7, pitch_f=0.75, db=2.0)),
    ]
    summary = {c: {"sim": [], "wer": [], "sec": []} for c, _ in configs}
    ref_wer = []
    gpu_reset()
    for n, (wav, meta) in enumerate(inputs):
        dur = sf.info(wav).duration
        base = os.path.basename(wav)
        t_common = time.perf_counter()
        common, timing_common = si.compute_common_features(wav, False, 0.0, "m", "None", 2, 0.0, 0.0)
        t_common = time.perf_counter() - t_common
        for stage, sec in timing_common.items():
            timed("shared analysis", stage, sec, dur, base, cold=n == 0)
        timed("shared analysis", "TOTAL", t_common, dur, base, cold=n == 0)
        if meta.get("text"):
            ref_wer.append(wer(meta["text"], transcribe(si, wav)))
        for cname, cfg in configs:
            t0 = time.perf_counter()
            arr, sr, timing_seed, _, _, _ = si.seed_specific_anonymization(
                common, False, cfg["admixture_ratio"], None, "None", 2, 52, cfg["db"], cfg["pitch_f"])
            t_seed = time.perf_counter() - t0
            for stage, sec in timing_seed.items():
                timed(f"synthesis {cname}", stage, sec, dur, base, cold=n == 0)
            timed(f"synthesis {cname}", "TOTAL", t_seed, dur, base, cold=n == 0)
            timed(f"end to end {cname}", "TOTAL", t_common + t_seed, dur, base, cold=n == 0)
            out = os.path.join(out_dir, "pipeline", f"{os.path.basename(wav)[:-4]}_{cname}.wav")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            sf.write(out, arr, sr)
            summary[cname]["sec"].append(t_seed)
            problems = audio_sanity(out, wav)
            if problems:
                record(f"output sanity {os.path.basename(out)}", "FAIL", "; ".join(problems))
            summary[cname]["sim"].append(ecapa_similarity(si, wav, out))
            if meta.get("text"):
                summary[cname]["wer"].append(wer(meta["text"], transcribe(si, out)))

    record("GPU memory (pipeline)", "PASS", "peak allocated by PyTorch during the pipeline check",
           peak_gb=gpu_peak_gb())
    if ref_wer:
        record("reference WER (original audio)", "PASS", "large-v2 on the inputs",
               wer=round(100 * float(np.mean(ref_wer)), 2))
    for cname, s in summary.items():
        sim = float(np.mean(s["sim"]))
        w = 100 * float(np.mean(s["wer"])) if s["wer"] else float("nan")
        # Loose bounds: these catch broken weights, not paper-level regressions.
        status = "PASS" if sim < 0.5 and (np.isnan(w) or w < 25) else "FAIL"
        record(f"operating point {cname}", status,
               f"{len(s['sim'])} files; ECAPA similarity to source (lower is better)",
               mean_sim=round(sim, 3), wer=round(w, 2), sec_per_file=round(float(np.mean(s["sec"])), 1))


@check("determinism")
def check_determinism(si, wav):
    common, _ = si.compute_common_features(wav, False, 0.0, "m", "None", 2, 0.0, 0.0)
    outs = [si.seed_specific_anonymization(common, False, 0.7, None, "None", 2, 52, 2.0, 0.75)[0]
            for _ in range(2)]
    same = outs[0].shape == outs[1].shape and np.array_equal(outs[0], outs[1])
    record("determinism (same seed, same features)", "PASS" if same else "WARN",
           "identical output" if same else "outputs differ between runs with the same seed")


@check("reproduce published samples")
def check_published(si, pairs, out_dir):
    """Regenerate published clips with their seed; the voice should match the published one."""
    sims, cross = [], []
    for src, published, seed in pairs:
        common, _ = si.compute_common_features(src, False, 0.0, "m", "None", 2, 0.0, 0.0)
        arr, sr, *_ = si.seed_specific_anonymization(common, False, 0.7, None, "None", 2, seed, 2.0, 0.75)
        out = os.path.join(out_dir, "reproduce", f"{os.path.basename(published)}")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sf.write(out, arr, sr)
        sims.append(ecapa_similarity(si, out, published))
        cross.append(ecapa_similarity(si, out, src))
    s, c = float(np.mean(sims)), float(np.mean(cross))
    status = "PASS" if s > 0.6 and s > c + 0.2 else "WARN"
    record("reproduce published samples", status,
           f"{len(pairs)} clips; similarity of regenerated voice to published voice vs to source",
           to_published=round(s, 3), to_source=round(c, 3))


@check("published sample settings sweep")
def check_published_sweep(si, pairs, pool_dirs, out_dir):
    """Find which pseudospeaker settings reproduce the published voices.

    The voice is set by the pool, the gender filter, the number of averaged
    speakers and the cluster criterion. Each combination is synthesized at
    beta = 0.7 and compared with the published clip.
    """
    import itertools
    import pspi.pseudospeaker as nps
    original_pool = si.model_state.pool
    commons = {}
    for src, _, _, _ in pairs:
        if src not in commons:
            commons[src] = si.compute_common_features(src, False, 0.0, "m", "None", 2, 0.0, 0.0)[0]
    rows = []
    for pool_name, pool_dir in pool_dirs.items():
        try:
            si.model_state.pool = nps.Pool.load(pool_dir)
        except Exception as e:  # noqa: BLE001
            record(f"sweep pool {pool_name}", "WARN", f"could not load: {type(e).__name__}: {e}")
            continue
        for gmode, n_spk, crit in itertools.product(("source", "opposite", "any"), (1, 2), ("None", "cluster_sparse", "cluster_dense")):
            sims = []
            for src, published, seed, src_gender in pairs:
                g = {"source": src_gender, "any": None,
                     "opposite": {"m": "f", "f": "m"}.get(src_gender)}[gmode]
                arr, sr, *_ = si.seed_specific_anonymization(commons[src], False, 0.7, g, crit, n_spk, seed, 2.0, 0.75)
                tmp = os.path.join(out_dir, "sweep", "tmp.wav")
                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                sf.write(tmp, arr, sr)
                sims.append(ecapa_similarity(si, tmp, published))
            rows.append({"pool": pool_name, "gender": gmode, "n_speaker": n_spk, "criterion": crit,
                         "mean_sim": round(float(np.mean(sims)), 3), "min_sim": round(float(np.min(sims)), 3)})
    si.model_state.pool = original_pool
    rows.sort(key=lambda r: -r["mean_sim"])
    with open(os.path.join(out_dir, "sweep", "published_settings_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2)
    best = rows[0]
    status = "PASS" if best["min_sim"] > 0.6 else "WARN"
    record("published sample settings sweep", status,
           f"{len(rows)} settings x {len(pairs)} clips; best: {best}; full table in sweep/published_settings_sweep.json")


@check("speech editing")
def check_editing(si, wav, meta, out_dir):
    session = si.PipelineSession()
    si.process_inputs([wav], os.path.join(out_dir, "editing"), False, 0.7, "any", "None", 2,
                      [52], 2.0, 0.75, session)
    name = session.list_results()[0]
    anon = session.results[name].wav_path  # apply_replacements repoints wav_path to the edit
    words = [w for w, _, _ in session.results[name].word_timestamps]
    # Replace the longest word so the edit is easy to hear and to find in the transcript.
    idx = max(range(len(words)), key=lambda i: len(words[i]))
    replacement = "Jonathan"
    out = os.path.join(out_dir, "editing", "edited.wav")
    t0 = time.perf_counter()
    si.apply_replacements(session, name, [idx], [replacement], out)
    timed("speech editing", "replace one word + resynthesize", time.perf_counter() - t0,
          sf.info(wav).duration, os.path.basename(wav))
    problems = audio_sanity(out)
    hyp = transcribe(si, out)
    consistency = ecapa_similarity(si, anon, out)
    found = "jonathan" in normalize_text(hyp)
    status = "PASS" if not problems and consistency > 0.7 and found else "WARN" if not problems else "FAIL"
    record("speech editing", status,
           f"replaced '{words[idx]}' with '{replacement}'; transcript: '{hyp}'; {problems or 'audio OK'}",
           edit_sim=round(consistency, 3))


def run_cli(name, cmd, out_dir):
    log = os.path.join(out_dir, "cli", f"{name}.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, log, round(time.time() - t0, 1)


@check("CLI")
def check_cli(inputs, out_dir):
    py = sys.executable
    a, b = inputs[0][0], inputs[1][0]
    d = os.path.join(out_dir, "cli")
    cases = {
        "static_infer": [py, "static_infer.py", "--inputs", a, "--output-dir", f"{d}/single",
                         "--seeds", "52", "--gender", "any", "--admixture-ratio", "0.7",
                         "--pitch-f", "0.75", "--db", "2", "--print-words",
                         f"anonymized_{os.path.basename(a)}_seed_52",
                         "--replace-result", f"anonymized_{os.path.basename(a)}_seed_52",
                         "--replace-indices", "1", "--replace-texts", "Jonathan",
                         "--replace-out", f"{d}/single/edited.wav"],
        "static_infer_batch": [py, "static_infer_batch.py", "--inputs", a, b,
                               "--output-dir", f"{d}/batch", "--admixture-ratio", "0,0.7",
                               "--pitch-f", "0.75", "--db", "2", "--workers", "1",
                               "--store-replacement-data"],
        "static_infer_multi": [py, "static_infer_multi.py", "--inputs", a, b,
                               "--output-dir", f"{d}/multi", "--admixture-ratio", "0.7",
                               "--model-procs", "1", "--threads-per-model", "2"],
    }
    expected = {
        "static_infer": [f"{d}/single/anonymized_{os.path.basename(a)}_seed_52.wav", f"{d}/single/edited.wav"],
        "static_infer_batch": None,
        "static_infer_multi": None,
    }
    for name, cmd in cases.items():
        rc, log, sec = run_cli(name, cmd, out_dir)
        timed("CLI wall time (includes model loading)", name, sec)
        wavs = expected[name] or glob.glob(f"{d}/{name.split('_')[-1]}/**/*.wav", recursive=True)
        missing = [w for w in wavs if not os.path.isfile(w)]
        bad = {os.path.basename(w): audio_sanity(w) for w in wavs if w not in missing}
        bad = {k: v for k, v in bad.items() if v}
        ok = rc == 0 and wavs and not missing and not bad
        record(f"CLI {name}", "PASS" if ok else "FAIL",
               f"exit {rc}, {len(wavs)} wavs, missing {missing}, problems {bad}, log {log}", seconds=sec)


def timing_table():
    """Aggregate TIMINGS per (group, stage), separating the first (cold) file."""
    import collections
    import platform
    rows = collections.OrderedDict()
    for t in TIMINGS:
        rows.setdefault((t["group"], t["stage"]), []).append(t)
    out = ["## Timing", ""]
    try:
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        out.append(f"Hardware: {dev}; torch {torch.__version__}; Python {platform.python_version()}.")
    except Exception:  # noqa: BLE001
        pass
    out += ["",
            "Warm statistics exclude the first file, which includes CUDA and cache warm-up. "
            "RTF is processing time divided by audio duration (below 1 is faster than real time).",
            "",
            "| Group | Stage | n | Cold (s) | Warm mean (s) | Warm median (s) | Warm p95 (s) | Warm max (s) | Warm RTF |",
            "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for (group, stage), items in rows.items():
        cold = [i["seconds"] for i in items if i["cold"]]
        warm = [i for i in items if not i["cold"]] or items
        secs = np.array([i["seconds"] for i in warm])
        audio = [i["audio_seconds"] for i in warm if i["audio_seconds"]]
        rtf = f"{secs.sum() / sum(audio):.3f}" if len(audio) == len(warm) else ""
        out.append(f"| {group} | {stage} | {len(items)} | {cold[0]:.2f} |" if cold else f"| {group} | {stage} | {len(items)} |  |")
        out[-1] += (f" {secs.mean():.2f} | {np.median(secs):.2f} | {np.percentile(secs, 95):.2f} |"
                    f" {secs.max():.2f} | {rtf} |")
    audio = sorted({(t["file"], t["audio_seconds"]) for t in TIMINGS if t["file"] and t["audio_seconds"]})
    if audio:
        d = np.array([a for _, a in audio])
        out += ["", f"Input audio: {len(d)} files, {d.min():.1f} to {d.max():.1f} s, mean {d.mean():.1f} s."]
    return "\n".join(out) + "\n"


PRICING = {}


def cost_section(run_seconds):
    hourly = sum(v for k, v in PRICING.items() if k.endswith("_usd_per_hour"))
    out = ["## Cost estimate", ""]
    if hourly <= 0:
        return "\n".join(out + ["No prices given. Pass --vm-usd-per-hour and --gpu-usd-per-hour to enable this section.", ""])
    out += [f"Hourly rate used: **${hourly:.3f}/h** "
            f"(VM ${PRICING['vm_usd_per_hour']:.3f} + GPU ${PRICING['gpu_usd_per_hour']:.3f}"
            f" + platform ${PRICING['platform_usd_per_hour']:.3f}).",
            f"Price sources: {PRICING['source']}",
            "Costs cover compute time only; persistent disks are billed per GB-month whether or not jobs run.",
            "",
            f"- This verification run: {run_seconds / 60:.1f} min wall time, about **${hourly * run_seconds / 3600:.3f}**.",
            ""]
    groups = {}
    for t in TIMINGS:
        if t["stage"] == "TOTAL" and t["audio_seconds"] and not t["cold"] and t["group"].startswith("end to end"):
            g = groups.setdefault(t["group"], [0.0, 0.0])
            g[0] += t["seconds"]
            g[1] += t["audio_seconds"]
    if groups:
        out += ["| Setting | Warm RTF | Audio hours per GPU hour | Cost per audio hour | Cost per 1,000 audio hours |",
                "|---|--:|--:|--:|--:|"]
        for name, (sec, aud) in groups.items():
            rtf = sec / aud
            out.append(f"| {name.replace('end to end ', '')} | {rtf:.3f} | {1 / rtf:.1f} | "
                       f"${hourly * rtf:.3f} | ${1000 * hourly * rtf:,.0f} |")
        out += ["", "Single process, one file at a time, no batching; batch scripts with several workers will be cheaper per audio hour."]
    return "\n".join(out) + "\n"


def write_report(out_dir):
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    with open(os.path.join(out_dir, "timings.json"), "w") as f:
        json.dump(TIMINGS, f, indent=2)
    lines = ["| Check | Status | Detail | Metrics |", "|---|---|---|---|"]
    for r in RESULTS:
        m = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("check", "status", "detail"))
        lines.append(f"| {r['check']} | {r['status']} | {r['detail']} | {m} |")
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("# Release verification\n\n" + "\n".join(lines) + "\n\n" + timing_table()
                + "\n" + cost_section(time.time() - RUN_START))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs-dir", required=True)
    ap.add_argument("--out-dir", default="verify_out")
    ap.add_argument("--max-inputs", type=int, default=20)
    ap.add_argument("--published-dir", default=None,
                    help="Folder with published <input>_seed_<seed>_anon.wav clips to reproduce.")
    ap.add_argument("--alt-pool", default=None,
                    help="Extra pool directory to include in the published-settings sweep.")
    ap.add_argument("--vm-usd-per-hour", type=float, default=0.0)
    ap.add_argument("--gpu-usd-per-hour", type=float, default=0.0)
    ap.add_argument("--platform-usd-per-hour", type=float, default=0.0,
                    help="Managed service fee, e.g. Cloud Workstations.")
    ap.add_argument("--price-source", default="not given")
    ap.add_argument("--skip-french", action="store_true")
    ap.add_argument("--skip-cli", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    PRICING.update(vm_usd_per_hour=args.vm_usd_per_hour, gpu_usd_per_hour=args.gpu_usd_per_hour,
                   platform_usd_per_hour=args.platform_usd_per_hour, source=args.price_source)

    wavs = sorted(glob.glob(os.path.join(args.inputs_dir, "*.wav")))[: args.max_inputs]
    if len(wavs) < 2:
        sys.exit("Need at least two input wavs.")
    inputs = []
    for w in wavs:
        j = w[:-4] + ".json"
        inputs.append((w, json.load(open(j)) if os.path.isfile(j) else {}))

    check_files(not args.skip_french)
    check_checksums()
    check_pool()
    if not args.skip_french:
        check_french()

    gpu_reset()
    t0 = time.perf_counter()
    import static_infer as si  # loads the English models once
    load_s = time.perf_counter() - t0
    timed("startup", "import + load English models", load_s)
    record("load English models", "PASS", "static_infer.ModelState",
           seconds=round(load_s, 1), peak_gb=gpu_peak_gb())

    check_pipeline(si, inputs, args.out_dir)
    check_determinism(si, wavs[0])
    check_editing(si, *inputs[0], args.out_dir)
    if args.published_dir:
        pairs = []
        for w, _ in inputs:
            stem = os.path.basename(w)[:-4].replace("_orig", "")
            for p in sorted(glob.glob(os.path.join(args.published_dir, f"{stem}_seed_*_anon.wav")))[:2]:
                pairs.append((w, p, int(re.search(r"_seed_(\d+)_", p).group(1))))
        if pairs:
            check_published(si, pairs[:10], args.out_dir)
            # two male and two female sources, one published seed each
            genders = {w: m.get("gender") for w, m in inputs}
            by_g = {"m": [], "f": []}
            for src, pub, seed in pairs:
                g = genders.get(src)
                if g in by_g and len(by_g[g]) < 2 and all(p[0] != src for p in by_g[g]):
                    by_g[g].append((src, pub, seed, g))
            pools = {"release": os.path.join(MODELS_DIR, "POOL/english")}
            if args.alt_pool:
                pools["alt"] = args.alt_pool
            check_published_sweep(si, by_g["m"] + by_g["f"], pools, args.out_dir)
    if not args.skip_cli:
        del si.model_state  # free GPU memory for the subprocesses
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        check_cli(inputs, args.out_dir)

    write_report(args.out_dir)
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    n_warn = sum(r["status"] == "WARN" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks, {n_fail} FAIL, {n_warn} WARN. Report: {args.out_dir}/report.md")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
