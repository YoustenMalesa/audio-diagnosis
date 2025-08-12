import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
import yaml
from tqdm import tqdm
from ..models.crnn import CRNN

CLASS_LABELS = [
    'normal','asthma','copd','bronchitis','heart failure','lung fibrosis','pleural effusion','pneumonia'
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TensorDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(row.tensor_path)
        tensor = data['tensor']  # (1, mels, T)
        return tensor, int(row.label), row.patient_id


def augment(batch, cfg):
    (x, y) = batch
    if cfg.get('time_mask',0)>0:
        for i in range(x.size(0)):
            for _ in range(cfg['time_mask']):
                t = random.randint(0, x.size(-1)-1)
                w = random.randint(1, max(1, x.size(-1)//20))
                x[i,:,t:t+w] = 0
    if cfg.get('freq_mask',0)>0:
        for i in range(x.size(0)):
            for _ in range(cfg['freq_mask']):
                f = random.randint(0, x.size(2)-1)
                w = random.randint(1, max(1, x.size(2)//20))
                x[i,:,f:f+w,:] = 0
    return x, y


def collate(batch):
    tensors, labels, pids = zip(*batch)
    # pad time dimension
    lengths = [t.size(-1) for t in tensors]
    max_len = max(lengths)
    padded = []
    for t in tensors:
        if t.size(-1) < max_len:
            t = F.pad(t, (0, max_len - t.size(-1)))
        padded.append(t)
    x = torch.stack(padded, dim=0)  # (B,1,mels,T)
    y = torch.tensor(labels, dtype=torch.long)
    return x, y, pids


def train_one_epoch(model, loader, optimizer, criterion, device, aug_cfg):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        x, y = augment((x, y), aug_cfg)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), aug_cfg.get('grad_clip',5.0))
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss/total, correct/total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return total_loss/total, correct/total, all_preds, all_labels


def main(config_path: str):
    with open(config_path,'r') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['train']['seed'])
    paths = cfg['paths']
    processed_dir = Path(paths['processed_dir'])
    models_dir = Path(paths['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(processed_dir / 'metadata.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gkf = GroupKFold(n_splits=cfg['train']['folds'])
    groups = df['patient_id']
    first_split = next(gkf.split(df, df['label'], groups))
    train_idx, val_idx = first_split
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_ds = TensorDataset(train_df)
    val_ds = TensorDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, collate_fn=collate, num_workers=cfg['train']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], collate_fn=collate, num_workers=cfg['train']['num_workers'])

    mcfg = cfg['model']
    model = CRNN(n_mels=mcfg['n_mels'], n_classes=len(CLASS_LABELS), cnn_channels=tuple(mcfg['cnn_channels']), rnn_hidden=mcfg['rnn_hidden'], rnn_layers=mcfg['rnn_layers'], dropout=mcfg['dropout'])
    model.to(device)

    # class weights (robust to missing classes in split)
    counts_series = train_df['label'].value_counts()
    counts = [counts_series.get(i, 0) for i in range(len(CLASS_LABELS))]
    counts_tensor = torch.tensor(counts, dtype=torch.float)
    # Avoid division by zero: add 1 to zero counts for weight shaping, but mark them separately
    adjusted = counts_tensor.clone()
    adjusted[adjusted == 0] = counts_tensor.max().clamp(min=1.0)
    raw_weights = counts_tensor.sum() / (adjusted + 1e-6)
    weights = raw_weights / raw_weights.sum() * len(CLASS_LABELS)
    if (counts_tensor == 0).any():
        missing = [CLASS_LABELS[i] for i, c in enumerate(counts) if c == 0]
        print(f"[WARN] Missing classes in training fold: {missing}. Their weights set heuristically; model won't learn them this fold.")
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    opt_name = cfg['train']['optimizer'].lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs']) if cfg['train']['scheduler']=='cosine' else None

    best_acc = 0.0
    patience = cfg['train']['patience']
    no_improve = 0

    for epoch in range(1, cfg['train']['epochs']+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, {**cfg['augment'], **cfg['train']})
        val_loss, val_acc, preds, labels = evaluate(model, val_loader, criterion, device)
        if scheduler: scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), models_dir / 'model_crnn.pt')
        else:
            no_improve += 1
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} (best {best_acc:.3f})")
        if no_improve >= patience:
            print('Early stopping.')
            break

    # Report (robust to missing classes in this fold)
    print('Validation classification report:')
    all_label_ids = list(range(len(CLASS_LABELS)))
    print(classification_report(labels, preds, labels=all_label_ids, target_names=CLASS_LABELS, digits=3, zero_division=0))

    # Export TorchScript
    best_model = CRNN(n_mels=mcfg['n_mels'], n_classes=len(CLASS_LABELS), cnn_channels=tuple(mcfg['cnn_channels']), rnn_hidden=mcfg['rnn_hidden'], rnn_layers=mcfg['rnn_layers'], dropout=mcfg['dropout'])
    best_model.load_state_dict(torch.load(models_dir / 'model_crnn.pt', map_location=device))
    best_model.eval()
    example = torch.randn(1,1,mcfg['n_mels'], 400, device=device)
    traced = torch.jit.trace(best_model, example)
    traced.save(str(models_dir / 'model_crnn.ts'))
    print('Exported TorchScript model to', models_dir / 'model_crnn.ts')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    main(args.config)
