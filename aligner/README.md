# Aligner

The phoneme front end for Stream B (paper, Sec. 2.1.2).

- `src/__init__.py`: `AlignerWrapper`, the inference entry point used by `static_infer.py`.
  It turns Whisper word segments into IPA with Transphone, force-aligns them to
  RVQ-Whisper frames (50 fps), and returns frame-level articulatory features.
  For content replacement it also predicts durations and F0 for new text
  (`predict_artics`, `predict_f0`).
- `src/training.py`: CTC head (111 classes: phonemes plus blank) on frozen
  Whisper encoder features, Viterbi forced aligner, and the text duration predictor.
- `src/f0predictor.py`: F0 predictor from articulatory features.
- `src/articulatory_features.py`: 64-dimensional articulatory feature inventory (IMS-Toucan).
- `src/f0_predictor.pth`, `src/dur_predictor.pt`: small pretrained English
  F0 and duration predictors. They are loaded by default.

The CTC aligner checkpoint (`Aligner/aligner_english.pth`) is part of the main
model release. See the top-level README.

Training scripts read their paths from the environment:

```bash
export MODELS_DIR=/path/to/models           # RVQWhisper/ and Aligner/ checkpoints
export ARTICS_DIR=/path/to/English/artics   # F0 predictor training inputs
export F0_DIR=/path/to/English/F0
PYTHONPATH=.:./rvqwhisper/src python aligner/src/training.py
```
