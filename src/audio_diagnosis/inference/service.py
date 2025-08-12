import io
import os
import librosa
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from .schemas import CLASS_LABELS, PredictResponse, Prediction
from ..models.crnn import CRNN

MODEL_PATH = os.getenv('MODEL_PATH', 'models/model_crnn.pt')
SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '4000'))
N_MELS = int(os.getenv('N_MELS', '128'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'crnn_v1')

app = FastAPI(title="Lung Sound Diagnosis API", description="Upload a WAV (mono) to receive predicted pulmonary condition.")

_model = None
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = CRNN(n_mels=N_MELS, n_classes=len(CLASS_LABELS))
        state = torch.load(MODEL_PATH, map_location=_device)
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model = model
    return _model


def extract_logmel(y, sr):
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=1024, hop_length=256, fmin=20, fmax=1800)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # per-sample normalize
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def assess_severity_and_stage(prob: float, entropy: float, margin: float):
    """Heuristic mapping of confidence probability + distribution characteristics to severity & stage.

    Parameters:
      prob: top-1 probability
      entropy: Shannon entropy over class probabilities (higher = more uncertainty)
      margin: difference between top-1 and top-2 probabilities
    Logic (adjustable thresholds):
      - Severity scaled mainly by prob & margin; uncertainty (entropy) can downgrade.
      - Stage escalates only if both prob and margin high and entropy low.
    """
    # Thresholds (could be externalized / calibrated)
    if prob >= 0.80 and margin >= 0.25 and entropy <= 1.2:
        severity = 'High'
        stage = 'Advanced'
    elif prob >= 0.60 and margin >= 0.15 and entropy <= 1.6:
        severity = 'Medium'
        stage = 'Progressed'
    elif prob >= 0.45 and margin >= 0.08:
        severity = 'Medium'
        stage = 'Early'
    else:
        severity = 'Low'
        stage = 'Early'
    # Downgrade if entropy very high (diffuse distribution)
    if entropy > 1.9:
        if severity == 'High':
            severity = 'Medium'
            stage = 'Progressed'
        elif severity == 'Medium':
            severity = 'Low'
            stage = 'Early'
    return severity, stage


def enable_mc_dropout(model):
    # Enable dropout layers during eval for MC sampling
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_dropout_predict(model, x, samples: int = 20):
    probs_accum = []
    with torch.no_grad():
        for _ in range(samples):
            enable_mc_dropout(model)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            probs_accum.append(probs.cpu().numpy())
    probs_stack = np.stack(probs_accum, axis=0)  # (S, B, C)
    mean_probs = probs_stack.mean(axis=0)[0]
    std_probs = probs_stack.std(axis=0)[0]
    return mean_probs, std_probs


@app.post('/predict', response_model=PredictResponse, summary="Predict diagnosis from a lung sound WAV")
async def predict(
    file: UploadFile = File(..., description="WAV audio file"),
    topk: int = Query(3, ge=1, le=len(CLASS_LABELS))
):
    if topk <= 0:
        raise HTTPException(status_code=400, detail="topk must be > 0")
    try:
        data = await file.read()
        y, sr = librosa.load(io.BytesIO(data), sr=SAMPLE_RATE)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")
    mel = extract_logmel(y, sr)
    tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    try:
        model = load_model()
        tensor = tensor.to(_device)
        mc_samples = int(os.getenv('MC_SAMPLES', '20'))
        confidence_level = float(os.getenv('CONFIDENCE_LEVEL', '0.95'))
        mean_probs, std_probs = mc_dropout_predict(model, tensor, samples=mc_samples)
        probs = mean_probs
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    idx_sorted = np.argsort(-probs)
    # Compute distribution characteristics
    top1_prob = float(probs[idx_sorted[0]])
    top1_std = float(std_probs[idx_sorted[0]])
    top2_prob = float(probs[idx_sorted[1]]) if len(idx_sorted) > 1 else 0.0
    margin = top1_prob - top2_prob
    # Entropy (base e)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())

    severity, stage = assess_severity_and_stage(top1_prob, entropy, margin)
    # Uncertainty-based adjustments: widen stage if high variance
    if top1_std > 0.10 and severity == 'High':
        severity = 'Medium'
        stage = 'Progressed'
    if top1_std > 0.15:
        severity = 'Low'
        stage = 'Early'

    k = min(topk, len(CLASS_LABELS))
    top_preds = []
    from math import sqrt
    import scipy.stats as st
    z = st.norm.ppf(0.5 + confidence_level/2.0)
    for rank, i in enumerate(idx_sorted[:k]):
        # For non-top1 we optionally annotate severity/stage only if probability close to top1
        if rank == 0:
            sev, stg = severity, stage
        else:
            rel_margin = top1_prob - float(probs[i])
            if rel_margin < 0.1:
                sev, stg = 'Medium', 'Early'
            else:
                sev, stg = None, None
        p = float(probs[i])
        std = float(std_probs[i])
        # Approximate CI using mean +/- z*std (bounded [0,1])
        ci_low = max(0.0, p - z*std)
        ci_up = min(1.0, p + z*std)
        top_preds.append(Prediction(label=CLASS_LABELS[i], confidence=p, severity=sev, stage=stg, prob_std=std, ci_lower=ci_low, ci_upper=ci_up, samples=mc_samples))
    all_preds = [Prediction(label=CLASS_LABELS[i], confidence=float(probs[i])) for i in idx_sorted]
    return PredictResponse(top1=top_preds[0], topk=top_preds, all=all_preds, model_version=MODEL_VERSION)


@app.get('/healthz')
async def healthz():
    status = 'ready'
    model_loaded = _model is not None
    return {"status": status, "model_loaded": model_loaded, "model_path": MODEL_PATH, "device": str(_device)}


@app.get('/')
async def root():
    return JSONResponse({
        "message": "Lung Sound Diagnosis API. Use /predict to POST a WAV file.",
        "labels": CLASS_LABELS,
        "model_version": MODEL_VERSION
    })
