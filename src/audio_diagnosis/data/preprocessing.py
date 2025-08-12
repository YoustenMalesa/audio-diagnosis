import argparse
import os
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

META_COLUMNS = [
    'tensor_path','label','patient_id','filter_mode','sound_type','chest_loc','age','gender'
]

LABEL_MAP = {
    'normal':0,
    'asthma':1,
    'copd':2,
    'bronchitis':3,
    'heart failure':4,
    'lung fibrosis':5,
    'pleural effusion':6,
    'pneumonia':7
}

FILTER_CODES = {'B':'bell','D':'diaphragm','E':'extended'}


def parse_filename(fname: str):
    base = os.path.splitext(fname)[0]
    parts = base.split(',')
    try:
        prefix, diag_part = parts[0].split('_',1)
        filter_code = prefix[0]
        patient_id = prefix[prefix.index('P')+1:]
    except Exception:
        return None
    diag = diag_part.lower().strip()
    if diag not in LABEL_MAP:
        # handle variants (copd, heart failure spacing etc.)
        diag_norm = diag.replace(' ', '')
        for k in LABEL_MAP.keys():
            if k.replace(' ','') == diag_norm:
                diag = k
                break
    meta = {
        'diagnosis': diag,
        'filter_mode': FILTER_CODES.get(filter_code,'unknown'),
        'patient_id': patient_id,
    }
    if len(parts) >= 5:
        meta['sound_type'] = parts[1]
        meta['chest_loc'] = parts[2].replace(' ','')
        try:
            meta['age'] = int(parts[3])
        except Exception:
            meta['age'] = None
        meta['gender'] = parts[4]
    return meta


def compute_logmel(y, sr, n_mels=128, n_fft=1024, hop_length=256, fmin=20, fmax=1800):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def process_audio(path, cfg):
    target_sr = cfg['sr']
    y, sr = librosa.load(path, sr=target_sr)
    # optional trim silence
    y, _ = librosa.effects.trim(y, top_db=cfg.get('top_db',80))
    mel_db = compute_logmel(y, sr, n_mels=cfg['n_mels'], n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], fmin=cfg['fmin'], fmax=cfg['fmax'])
    # normalize per-file
    mean = mel_db.mean()
    std = mel_db.std() + 1e-6
    mel_db = (mel_db - mean)/std
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)  # (1, mels, T)
    return tensor


def main(raw_dir: str, out_dir: str, **pre_cfg):
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records = []
    wavs = list(raw_path.glob('*.wav'))
    for fp in tqdm(wavs, desc='Processing audio'):
        meta = parse_filename(fp.name)
        if not meta:
            continue
        diag = meta['diagnosis']
        if diag not in LABEL_MAP:
            continue
        tensor = process_audio(fp, pre_cfg)
        save_name = fp.stem + '.pt'
        torch.save({'tensor': tensor, 'sr': pre_cfg['sr']}, out_path / save_name)
        records.append({
            'tensor_path': str(out_path / save_name),
            'label': LABEL_MAP[diag],
            'patient_id': meta.get('patient_id'),
            'filter_mode': meta.get('filter_mode'),
            'sound_type': meta.get('sound_type'),
            'chest_loc': meta.get('chest_loc'),
            'age': meta.get('age'),
            'gender': meta.get('gender'),
        })
    df = pd.DataFrame(records)
    df.to_csv(out_path / 'metadata.csv', index=False)
    print('Finished. Files:', len(records))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sr', type=int, default=4000)
    ap.add_argument('--n_mels', type=int, default=128)
    ap.add_argument('--hop_length', type=int, default=256)
    ap.add_argument('--n_fft', type=int, default=1024)
    ap.add_argument('--fmin', type=int, default=20)
    ap.add_argument('--fmax', type=int, default=1800)
    ap.add_argument('--top_db', type=int, default=80)
    args = ap.parse_args()
    main(**vars(args))
