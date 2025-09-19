# Lung Sound Diagnosis (CRNN Production Ready)

End-to-end pipeline and inference service for diagnosing lung diseases from electronic stethoscope recordings using a Convolutional Recurrent Neural Network (CRNN).

## Classes
normal, asthma, copd, bronchitis, heart failure, lung fibrosis, pleural effusion, pneumonia

## Pipeline Overview
1. Data Ingestion & Parsing (filenames + annotation sheet)
2. Preprocessing: resample -> denoise (optional) -> log-mel (128) with normalization
3. Data Splits: patient-level stratified folds (GroupKFold) + filter-aware balancing
4. Augmentation (time masking, freq masking, random time shift, noise mix)
5. Model: CRNN (CNN feature extractor + Bi-GRU + classifier)
6. Training: class-weighted cross entropy, early stopping, mixed precision, checkpointing
7. Evaluation: macro F1, per-class precision/recall, confusion matrix
8. Packaging: TorchScript export + FastAPI inference + Docker image

## Quick Start (Development)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

### Preprocess
Place raw WAV files under `data/Audio Files/` and the annotation workbook at `data/Data annotation.xlsx`.
```powershell
python -m audio_diagnosis.data.preprocessing --raw_dir "data/Audio Files" --out_dir data/processed --n_mels 128 --sr 4000
```

### Train
```powershell
python -m audio_diagnosis.training.train --config configs/crnn_baseline.yaml
```

### Run Inference API Locally
```powershell
uvicorn audio_diagnosis.inference.service:app --reload --port 8000
```
Then: http://localhost:8000/docs

### Docker Build
```powershell
docker build -t audio-diagnosis:latest .
docker run -p 8000:8000 -v ${PWD}/models:/app/models audio-diagnosis:latest
```

## Config Example (`configs/crnn_baseline.yaml`)
```yaml
model:
  type: crnn
  n_mels: 128
  cnn_channels: [32,64,128]
  rnn_hidden: 128
  rnn_layers: 2
  dropout: 0.3
train:
  epochs: 60
  batch_size: 32
  lr: 0.0005
  weight_decay: 0.0001
  grad_clip: 5.0
  mixed_precision: true
  patience: 12
  num_workers: 4
  seed: 42
  folds: 5
  optimizer: adamw
  scheduler: cosine
  warmup_epochs: 3
  ema: true
  ema_decay: 0.999
augment:
  time_mask: 2
  freq_mask: 2
  mixup: 0.4
  noise_std: 0.01
preprocess:
  sr: 4000
  n_mels: 128
  hop_length: 256
  n_fft: 1024
  fmin: 20
  fmax: 1800
  top_db: 80
paths:
  raw_dir: data/Audio Files
  processed_dir: data/processed
  models_dir: models
```

## Inference
Upload a WAV to `/predict` (FastAPI) to obtain top-k predictions.

## Export TorchScript
During training, the best model is exported to `models/model_crnn.ts` for portability (add mobile / edge inference later).

## Next Enhancements
- Add calibration (temperature scaling)
- Add SHAP or saliency map endpoint
- On-device quantization (int8) for edge deployment
- CI workflow (pytest + lint + docker build)

MIT License.

## Commands
Build & Train: docker build -t audio-diagnosis:trained .
Run: docker run -d --name audio-diagnosis-model --network mobiclinic-net -p 8002:8000 yousten/audio-diagnosis-model:trained