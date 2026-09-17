import sys
from types import SimpleNamespace
import gradio as gr

# Monkey-patch gr.Audio to remove unsupported keyword arguments.
original_audio_init = gr.Audio.__init__
def new_audio_init(self, *args, **kwargs):
    if "source" in kwargs:
        del kwargs["source"]
    if "type" in kwargs and kwargs["type"] == "binary":
        kwargs["type"] = "numpy"
    original_audio_init(self, *args, **kwargs)
gr.Audio.__init__ = new_audio_init

import tempfile
import os

# Root of the released checkpoints and speaker pool (see README, "Model weights").
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
import random
import io
import base64
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import json

# Import your model wrappers and dependencies
from aligner.src import AlignerWrapper
from rvqwhisper.src.rvqwhisper import RVQFasterWhisperWrapper
from bigvgan import BigVGANWrapper
import sys
sys.path.insert(0, "./audiolm")
from audiolm import AudioLMWrapper
import pspi.pseudospeaker as nps

# Import only the extraction function to avoid circular import issues.
from pspi.pitch import extract as pitch_extract

# -------------------------------
# LOAD MODELS ONCE - GLOBAL STATE
# -------------------------------
def merge_segments(segments, min_words=5, min_chars=70):
    merged_segments = []
    i = 0
    while i < len(segments):
        # Start a new merged segment with the current segment.
        current = segments[i]
        merged = SimpleNamespace(
            start=current.start,
            end=current.end,
            text=current.text,
            words=current.words
        )
        # Merge with subsequent segments until the merged segment meets the threshold.
        while (len(merged.words) < min_words or len(merged.text) < min_chars) and (i + 1 < len(segments)):
            i += 1
            next_seg = segments[i]
            merged.text = f"{merged.text.strip()} {next_seg.text.strip()}"
            merged.words = merged.words + next_seg.words
            merged.end = next_seg.end
        merged_segments.append(merged)
        i += 1
    return merged_segments

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
        # with open("best_speakers.json") as f:
        #     self.best_speakers = json.load(f)

# Initialize the global model state
model_state = ModelState()

# -------------------------------
# SET UP ORIGINAL SPEAKER EMBEDDING EXTRACTOR
# -------------------------------
from speechbrain.inference.speaker import EncoderClassifier
original_speaker_extractor = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# -------------------------------
# GLOBAL STORAGE FOR MASKING/REPLACEMENT DATA
# -------------------------------
# This dictionary will map result_name to a tuple: 
# (audio_bytes, sample_rate, word_timestamps, final_bottleneck, pitch, pseudospeaker)
results_data = {}

# -------------------------------
# HELPER FUNCTION: GENERATE PIE CHART FROM TIMINGS
# -------------------------------
def generate_pie_chart(times_dict, total_time):
    labels = list(times_dict.keys())
    sizes = list(times_dict.values())
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(f"Total Time: {total_time:.2f} sec")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# -------------------------------
# HELPER FUNCTION: COMMON PROCESSING (computed once per file)
# -------------------------------
def compute_common_features(input_filepath, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, db, pitch_f):
    timing_common = {}
    t_total_start = time.perf_counter()
    
    # --- Pitch Extraction ---
    t = time.perf_counter()
    pitch = np.squeeze(pitch_extract(input_filepath))
    timing_common["pitch extraction"] = time.perf_counter() - t

    # --- Bottleneck Computation ---
    t = time.perf_counter()
    tokens = model_state.WW.compute_bottleneck(input_filepath, vectorize=False)
    timing_common["bottleneck computation"] = time.perf_counter() - t

    # --- Transcription ---
    t = time.perf_counter()
    segments_init, info = model_state.WW.get_transcription(input_filepath, word_timestamps=True)
    segments = merge_segments(segments_init, 30, 70)
    timing_common["transcription"] = time.perf_counter() - t

    # Interpolate pitch to match tokens length
    x_old = np.linspace(0, 1, pitch.shape[0])
    x_new = np.linspace(0, 1, tokens.shape[0])
    pitch = np.interp(x_new, x_old, pitch)

    if predict_pitch:
        # Placeholder for pitch prediction logic
        pass

    if isinstance(tokens, list):
        tokens = tokens[0]

    # --- Articulatory Features Extraction ---
    t = time.perf_counter()
    artics_features_segments, segments_alignment, word_timestamps = model_state.ALW.get_articulatory_features(
        tokens, segments, info, return_word_timestamps=True
    )
    timing_common["articulatory features extraction"] = time.perf_counter() - t

    # --- Segment Token Generation ---
    t = time.perf_counter()
    segment_tokens_syn = model_state.AW.generate(artics_features_segments)
    timing_common["segment token generation"] = time.perf_counter() - t

    tokens_syn = tokens.copy()
    for i in range(len(segment_tokens_syn)):
        this_syn_len = min(segment_tokens_syn[i].shape[0], segments_alignment[i][1] - segments_alignment[i][0])
        tokens_syn[segments_alignment[i][0]:(segments_alignment[i][0] + this_syn_len)] = segment_tokens_syn[i][:this_syn_len]

    # Return word_timestamps along with the other data.
    return {"pitch": pitch, "tokens": tokens, "tokens_syn": tokens_syn, "word_timestamps": word_timestamps}, timing_common

# -------------------------------
# HELPER FUNCTION: SEED-SPECIFIC ANONYMIZATION (runs for each seed)
# -------------------------------
def seed_specific_anonymization(common_data, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, seed, db, pitch_f):
    timing_seed = {}
    t = time.perf_counter()
    pseudospeaker = nps.generate_pseudospeaker(
        model_state.pool, n_speakers=n_speaker, gender=gender, criterion=spk_cluster if spk_cluster != "None" else None, seed=seed#model_state.best_speakers[gender][seed%len(model_state.best_speakers[gender])]
    )
    timing_seed["pseudospeaker generation"] = time.perf_counter() - t

    t = time.perf_counter()
    final_bottleneck, corr = model_state.BW.admixture(
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
    array, sample_rate = model_state.BW.synthesize(final_bottleneck, pseudospeaker.xvector, pitch_seed, chunk_size=100)
    timing_seed["synthesis"] = time.perf_counter() - t

    # Return additional data (final_bottleneck, pitch_seed, pseudospeaker) for later replacements.
    return array, sample_rate, timing_seed, final_bottleneck, pitch_seed, pseudospeaker

# -------------------------------
# HELPER: Convert Audio Bytes to HTML Audio Player
# -------------------------------
def audio_to_html(fname, audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"<p>{fname}</p><audio controls src='data:audio/wav;base64,{b64}'></audio>"

# -------------------------------
# HELPER: Convert Pie Chart Bytes to HTML Image
# -------------------------------
def chart_to_html(fname, chart_bytes):
    b64 = base64.b64encode(chart_bytes).decode("utf-8")
    return f"<p>{fname} Processing Times</p><img src='data:image/png;base64,{b64}' style='max-width:400px;'/>"

# -------------------------------
# GRADIO INTERFACE FUNCTION
# -------------------------------
def process_files(files, record_audio, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, seeds_str, db, pitch_f):
    results = []
    messages = []
    
    # Combine uploaded files and recorded audio into one list.
    files_to_process = []
    if files is not None and len(files) > 0:
        files_to_process.extend(files)
    if record_audio is not None:
        if not isinstance(record_audio, list):
            files_to_process.append(record_audio)
        else:
            files_to_process.extend(record_audio)
    
    if len(files_to_process) == 0:
        return [], "No files or recordings provided."
    
    # If seeds_str is empty, we use the original speaker and original pitch.
    seeds = [] if seeds_str.strip() == "" else [int(s.strip()) for s in seeds_str.split(",") if s.strip() != ""]
    
    for idx, file_obj in enumerate(files_to_process):
        # Process uploaded file (dict)
        if isinstance(file_obj, dict):
            file_name = file_obj.get("name", f"file_{idx+1}.wav")
            file_data = file_obj.get("data")
        # Process recorded audio returned as a tuple
        elif isinstance(file_obj, tuple):
            try:
                if len(file_obj) >= 2:
                    audio_array, sample_rate = file_obj[1], file_obj[0]
                else:
                    audio_array = file_obj[0]
                    sample_rate = 44100  # default sample rate
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)
                # Convert stereo to mono if needed
                if audio_array.ndim > 1 and audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1)
                file_name = f"recorded_{idx+1}.wav"
                buf = io.BytesIO()
                sf.write(buf, audio_array, sample_rate, format="WAV")
                buf.seek(0)
                file_data = buf.read()
            except Exception as e:
                messages.append(f"Error processing recorded audio for file {idx+1}: {e}")
                continue
        # Process recorded audio returned directly as a NumPy array
        elif isinstance(file_obj, np.ndarray):
            try:
                audio_array = file_obj
                sample_rate = 44100  # default sample rate
                if audio_array.ndim > 1 and audio_array.shape[1] > 1:
                    audio_array = np.mean(audio_array, axis=1)
                file_name = f"recorded_{idx+1}.wav"
                buf = io.BytesIO()
                sf.write(buf, audio_array, sample_rate, format="WAV")
                buf.seek(0)
                file_data = buf.read()
            except Exception as e:
                messages.append(f"Error processing recorded audio (np.ndarray) for file {idx+1}: {e}")
                continue
        # Process bytes
        elif isinstance(file_obj, bytes):
            file_name = f"file_{idx+1}.wav"
            file_data = file_obj
        # Process file-like objects
        else:
            file_name = getattr(file_obj, "name", f"file_{idx+1}.wav")
            file_data = file_obj.read()

        messages.append(f"Processing file {idx+1}/{len(files_to_process)}: {file_name}")
        
        if not hasattr(file_data, "read"):
            file_data = io.BytesIO(file_data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
            temp_in.write(file_data.read())
            temp_in.flush()
            input_path = temp_in.name

        try:
            common_data, timing_common = compute_common_features(input_path, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, db, pitch_f)
            
            # If no seeds were provided, use original speaker embedding and original pitch.
            if len(seeds) == 0:
                messages.append(f"Processing file {idx+1}/{len(files_to_process)}: {file_name} with original speaker settings.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
                    output_path = temp_out.name
                try:
                    t0 = time.perf_counter()
                    # Extract the original speaker embedding using ECAPATDNNFeatureExtractor.
                    signal, fs = torchaudio.load(input_path)

                    # Resample to 16000 Hz if needed
                    if fs != 16000:
                        signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
                        fs = 16000  # optional, if you use fs later

                    original_xvector = original_speaker_extractor.encode_batch(signal).squeeze().detach().cpu().numpy()
                    t1 = time.perf_counter()
                    timing_original = {"embedding extraction": t1 - t0}
                    
                    t0 = time.perf_counter()
                    final_bottleneck, corr = model_state.BW.admixture(
                        common_data["tokens"], common_data["tokens_syn"], admixture_ratio, block_hallucination=True
                    )
                    t1 = time.perf_counter()
                    timing_original["admixture"] = t1 - t0
                    
                    total_time = sum(timing_common.values()) + sum(timing_original.values())
                    array, sample_rate = model_state.BW.synthesize(final_bottleneck, original_xvector, common_data["pitch"], chunk_size=100)
                    
                    sf.write(output_path, array, sample_rate)
                    with open(output_path, "rb") as f:
                        audio_bytes = f.read()
                    chart_bytes = generate_pie_chart({**timing_common, **timing_original}, total_time)
                    result_name = f"anonymized_{file_name}_original"
                    results.append((result_name, audio_bytes, chart_bytes, common_data["word_timestamps"], sample_rate))
                    messages.append(f"Finished processing: {file_name} with original speaker in {total_time:.2f} sec")
                    
                    # Store data for later replacement. Here pseudospeaker is set to None.
                    results_data[result_name] = (audio_bytes, sample_rate, common_data["word_timestamps"], final_bottleneck, common_data["pitch"], original_xvector, False)
                except Exception as e:
                    messages.append(f"Error processing {file_name} with original speaker: {e}")
                finally:
                    os.remove(output_path)
            else:
                for seed in seeds:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
                        output_path = temp_out.name
                    try:
                        array, sample_rate, timing_seed, final_bottleneck, pitch_seed, pseudospeaker = seed_specific_anonymization(common_data, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, seed, db, pitch_f)
                        final_timing = {**timing_common, **timing_seed}
                        total_time = sum(final_timing.values())
                        
                        sf.write(output_path, array, sample_rate)
                        with open(output_path, "rb") as f:
                            audio_bytes = f.read()
                        chart_bytes = generate_pie_chart(final_timing, total_time)
                        result_name = f"anonymized_{file_name}_seed_{seed}"
                        results.append((result_name, audio_bytes, chart_bytes, common_data["word_timestamps"], sample_rate))
                        messages.append(f"Finished processing: {file_name} with seed {seed} in {total_time:.2f} sec")
                        
                        # Store data for later replacement.
                        results_data[result_name] = (audio_bytes, sample_rate, common_data["word_timestamps"], final_bottleneck, pitch_seed, pseudospeaker, True)
                    except Exception as e:
                        messages.append(f"Error processing {file_name} with seed {seed}: {e}")
                    finally:
                        os.remove(output_path)
        except Exception as e:
            messages.append(f"Error processing {file_name}: {e}")
        finally:
            os.remove(input_path)
    return results, "\n".join(messages)

def run_pipeline(files, record_audio, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, seeds_str, db, pitch_f):
    processed_results, log = process_files(files, record_audio, predict_pitch, admixture_ratio, gender, spk_cluster, n_speaker, seeds_str, db, pitch_f)
    html_outputs = ""
    result_names = []
    for fname, audio_bytes, chart_bytes, word_timestamps, sample_rate in processed_results:
        html_outputs += audio_to_html(fname, audio_bytes)
        html_outputs += chart_to_html(fname, chart_bytes)
        result_names.append(fname)
    # Set the default selection to the first result if available.
    default_result = result_names[0] if result_names else None
    # Return the HTML outputs, log, and update the dropdown choices and selected value.
    return html_outputs, log, gr.update(choices=result_names, value=default_result)

# -------------------------------
# NEW FUNCTION: Update word choices for replacement given an output selection.
# -------------------------------
def get_word_choices(result_name):
    if result_name in results_data:
        # Use each occurrence's index and word for display.
        _, _, word_timestamps, _, _, _, _ = results_data[result_name]
        # Create choices like "0: word", "1: word", etc.
        choices = [f"{i}: {entry[0]}" for i, entry in enumerate(word_timestamps)]
        return gr.update(choices=choices)
    return gr.update(choices=[])

# -------------------------------
# NEW FUNCTION: Replace selected words (or phrases) with provided texts.
# -------------------------------
def apply_replacements(result_name, selected_occurrences, replacement_texts):
    """
    Given a selected result, a list of selected occurrences (each labeled with index and word),
    and a comma-separated string of replacement texts, this function groups consecutive selections
    into one segment, uses the earliest start timestamp and the latest end timestamp for each group,
    and then applies the replacement.
    """
    if result_name not in results_data:
        return None, "Selected result not found."
    
    audio_bytes, sample_rate, word_timestamps, final_bottleneck, pitch, pseudospeaker, is_p2 = results_data[result_name]
    
    # Parse the indices from the selected_occurrences.
    # Expected format is "index: word", so we take the index part.
    selected_indices = []
    for sel in selected_occurrences:
        try:
            idx = int(sel.split(":")[0])
            selected_indices.append(idx)
        except Exception as e:
            print("Error parsing selection:", sel, e)
            continue
    if not selected_indices:
        return None, "No words selected."
    
    # Sort indices and group consecutive indices together.
    selected_indices.sort()
    groups = []
    current_group = [selected_indices[0]]
    for idx in selected_indices[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)
    
    # Parse the replacement texts (comma-separated).
    replacements = [s.strip() for s in replacement_texts.split(",") if s.strip() != ""]
    if len(replacements) != len(groups):
        return None, "Number of replacement texts must match number of groups of selected words."
    
    # Build replacement operations for each group.
    # Each group is replaced with the corresponding replacement text.
    replacement_ops = []
    for group, rep_text in zip(groups, replacements):
        # Get start timestamp from the first occurrence and end timestamp from the last occurrence.
        first_entry = word_timestamps[group[0]]
        last_entry = word_timestamps[group[-1]]
        t_start = first_entry[1]
        t_end = last_entry[2]
        # Calculate indices as used by the model (example: times multiplied by factor 50).
        start_idx = round(t_start * 50)
        end_idx = round(t_end * 50)
        # Obtain synthesis data from the model given the replacement text.
        artics_features, prlen, polen = model_state.ALW.predict_artics(rep_text)
        gspitch = model_state.ALW.predict_f0(artics_features)
        tokens_syn = model_state.AW.generate([artics_features])[0]
        tokens_syn = tokens_syn[:artics_features.shape[0], :]
        tokens_syn = tokens_syn[prlen:-polen, :]
        gspitch = gspitch[prlen:-polen]
        replacement_ops.append((start_idx, end_idx, tokens_syn, gspitch))
    
    # Sort the replacement operations in reverse order (by start index) so that modifying the arrays doesn't
    # shift subsequent indices.
    replacement_ops.sort(key=lambda x: x[0], reverse=True)
    
    # Apply the replacement modifications.
    for start_idx, end_idx, tokens_syn, gspitch in replacement_ops:
        if not is_p2:
            nonzero_idxs = pitch.nonzero()
            src_nonzeros = pitch[nonzero_idxs]
            log_src = np.log(src_nonzeros)
            target_mean = np.mean(log_src)
            target_std = np.std(log_src)
        
        new_bottleneck = model_state.BW.decode_from_codebook_indices(tokens_syn[None,...])[0]
        # Use pseudospeaker to convert pitch if available; if not (original branch), use gspitch directly.
        if pseudospeaker is not None and is_p2:
            new_pitch = pseudospeaker.convert_pitch(gspitch)
        elif pseudospeaker is not None:
            nonzero_idxs = gspitch.nonzero()
            src_nonzeros = gspitch[nonzero_idxs]
            log_src = np.log(src_nonzeros)
            src_mean = np.mean(log_src)
            src_std = np.std(log_src)
            new_pitch = np.zeros_like(gspitch)
            new_pitch[nonzero_idxs] = np.exp(((np.log(gspitch[nonzero_idxs]) - src_mean) / src_std) * target_std + target_mean)
        else:
            new_pitch = gspitch  # Fallback if no pseudospeaker is used.
        
        # Replace the section in final_bottleneck and pitch.
        final_bottleneck = np.concatenate((
            final_bottleneck[:start_idx],
            new_bottleneck,
            final_bottleneck[end_idx:]
        ), axis=0)
        pitch = np.concatenate((
            pitch[:start_idx],
            new_pitch,
            pitch[end_idx:]
        ), axis=0)
    
    # Synthesize new audio with replaced parts.
    # For the xvector, if pseudospeaker is None, we assume the original xvector remains in use.
    xvector = pseudospeaker.xvector if is_p2 else pseudospeaker
    array, sample_rate = model_state.BW.synthesize(final_bottleneck, xvector, pitch, chunk_size=100)
    
    # Update the stored data with the new final_bottleneck and pitch.
    results_data[result_name] = (audio_bytes, sample_rate, word_timestamps, final_bottleneck, pitch, pseudospeaker, is_p2)
    return (sample_rate, array), "Replacements applied successfully."

# -------------------------------
# BUILD THE GRADIO INTERFACE WITH BLOCKS
# -------------------------------
with gr.Blocks(title="Voice Anonymization Pipeline") as demo:
    gr.Markdown("# Voice Anonymization Pipeline")
    gr.Markdown(
        "Upload one or more WAV files or record your voice using the microphone. "
        "Each audio is processed through the anonymization pipeline. "
        "For each file, multiple anonymization outputs are generated, one for each provided seed. "
        "If no seed is provided, the system uses the original pitch and a speaker embedding extracted by ECAPA-TDNN. "
        "A pie chart shows the processing time breakdown for each file/seed combination."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            predict_pitch_input = gr.Checkbox(label="Predict Pitch", value=False)
            admixture_ratio_input = gr.Slider(label="Admixture Ratio", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
            gender_input = gr.Radio(label="Select Gender", choices=["m", "f"], value="m")
            spk_cluster_input = gr.Radio(label="Select Speaker Cluster", choices=["cluster_dense", "cluster_sparse", "None"], value="None")
            n_spk_input = gr.Slider(label="N Speakers", minimum=1, maximum=10, value=2, step=1)
            seeds_input = gr.Textbox(label="Seed(s)", value="52", placeholder="Enter seeds separated by commas (e.g., 52, 123, 456). Leave empty to use original speaker.")
            db_input = gr.Number(label="dB Adjustment", value=0.0)
            pitch_f_input = gr.Slider(label="Pitch Factor", minimum=0.0, maximum=1.0, value=0.0, step=0.05)
        with gr.Column(scale=2):
            file_input = gr.File(label="Upload WAV files", file_count="multiple", type="binary")
            record_input = gr.Audio(type="numpy", label="Record Your Voice")
            run_button = gr.Button("Run Anonymization")
            output_html = gr.HTML(label="Anonymized Outputs")
            log_output = gr.Textbox(label="Processing Log", interactive=False, lines=10)
    
    # These hidden/output components are for the replacement feature.
    result_dropdown = gr.Dropdown(label="Select Anonymized Output for Replacement", choices=[], interactive=True)
    # Instead of a free-form textbox, we now provide a checkbox group displaying each word occurrence.
    word_multiselect = gr.CheckboxGroup(label="Select Words to Replace", choices=[])
    replacement_texts_input = gr.Textbox(label="Enter Replacement Texts", placeholder="Enter comma-separated replacement texts (in order)")
    replace_button = gr.Button("Apply Replacements")
    replaced_audio_output = gr.Audio(label="Replaced Audio Output", type="numpy")
    replace_log = gr.Textbox(label="Replacement Log", interactive=False, lines=2)
    
    # Run anonymization and update the dropdown choices for replacement.
    run_button.click(
        fn=run_pipeline,
        inputs=[file_input, record_input, predict_pitch_input, admixture_ratio_input, gender_input,
                spk_cluster_input, n_spk_input, seeds_input, db_input, pitch_f_input],
        outputs=[output_html, log_output, result_dropdown]
    )
    
    # When an output is selected, update the word multiselect choices.
    result_dropdown.change(
        fn=get_word_choices,
        inputs=[result_dropdown],
        outputs=[word_multiselect]
    )
    
    # When the replace button is clicked, apply the replacements and show the new audio.
    replace_button.click(
        fn=apply_replacements,
        inputs=[result_dropdown, word_multiselect, replacement_texts_input],
        outputs=[replaced_audio_output, replace_log]
    )

demo.launch()
