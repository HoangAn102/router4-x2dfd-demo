from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0

def _resolve_final_calibrators(bundle):
    from collections.abc import Mapping
    required = ('blending', 'diffusion', 'frequency', 'texture')
    if not isinstance(bundle, Mapping):
        raise RuntimeError(f'FINAL calibration bundle is not a mapping: {type(bundle).__name__}')
    if 'experts' not in bundle:
        raise RuntimeError(f"FINAL calibration bundle missing root['experts']; keys={list(bundle.keys())}")
    experts = bundle['experts']
    if not isinstance(experts, Mapping):
        raise RuntimeError("FINAL bundle root['experts'] is not a mapping")
    missing = [e for e in required if e not in experts]
    if missing:
        raise RuntimeError(f"FINAL root['experts'] missing: {missing}")
    out = {}
    for e in required:
        obj = experts[e]
        if isinstance(obj, Mapping):
            raise RuntimeError(f'FINAL {e} calibrator resolved to metadata dict')
        if obj is None:
            raise RuntimeError(f'FINAL {e} calibrator is None')
        usable = callable(getattr(obj, 'predict', None)) or callable(getattr(obj, 'transform', None)) or callable(getattr(obj, 'predict_proba', None)) or callable(obj)
        if not usable:
            raise RuntimeError(f'FINAL {e} object is not a usable calibrator: {type(obj).__module__}.{type(obj).__name__}')
        out[e] = obj
    print("FINAL CALIBRATOR SOURCE = root['experts']")
    print('FINAL CALIBRATOR TYPES =', {e: type(out[e]).__module__ + '.' + type(out[e]).__name__ for e in required})
    return out
BASE = Path('/home/aiotlab/hoangan')
ROOT = BASE / '/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/06_crossdata_benchmark/PAPER_FINAL_2CROSS_CDF_DFDCP/smoke/work'
ROUTER_CKPT = BASE / '/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt'
CAL_FILE = BASE / '/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/01_teacher/calibrators_FINAL_CLEAN.joblib'
EXPERTS = ['blending', 'diffusion', 'frequency', 'texture']
ALIASES = {'blending': 'Blending', 'diffusion': 'Diffusion', 'frequency': 'Frequency', 'texture': 'Texture'}
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class DS(Dataset):

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert('RGB')
            x = tf(im)
        return (x, p)
manifest = pd.read_csv(ROOT / 'manifest.csv', low_memory=False)
paths = manifest['image_path'].astype(str).tolist()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = efficientnet_b0(weights=None)
nf = model.classifier[1].in_features
model.classifier[1] = nn.Linear(nf, len(EXPERTS))
ckpt = torch.load(Path('/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/02_router/final_checkpoint/best.pt'), map_location='cpu')
model.load_state_dict(ckpt['model'])
model = model.to(device).eval()
loader = DataLoader(DS(paths), batch_size=128, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
routes = {}
with torch.no_grad():
    for (x, batch_paths) in loader:
        x = x.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            prob = torch.softmax(model(x), dim=1)
        top2 = prob.topk(2, dim=1)
        idx = top2.indices[:, 0].cpu().numpy()
        conf = top2.values[:, 0].cpu().numpy()
        margin = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
        for (p, r, c, m) in zip(batch_paths, idx, conf, margin):
            routes[str(p)] = {'expert': EXPERTS[int(r)], 'confidence': float(c), 'margin': float(m)}
score_maps = {}
for expert in EXPERTS:
    csv_path = ROOT / f'{expert}_scores.csv'
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.drop_duplicates('image_path', keep='last')
    score_maps[expert] = dict(zip(df['image_path'].astype(str), df['score'].astype(float)))
    print(expert, 'scores =', len(score_maps[expert]))
bundle = joblib.load(Path('/home/aiotlab/hoangan/outputs/ROUTER4_FINAL_CLEAN_RETRAIN_20260919/01_teacher/calibrators_FINAL_CLEAN.joblib'))
calibrators = _resolve_final_calibrators(bundle)
rows = []
missing = []
for p in paths:
    r = routes[p]
    expert = r['expert']
    raw = score_maps[expert].get(p)
    if raw is None:
        missing.append((p, expert))
        continue
    calibrated = float(calibrators[expert].predict([float(raw)])[0])
    rows.append({'image_path': p, 'selected_expert': expert, 'selected_alias': ALIASES[expert], 'selected_raw_score': float(raw), 'selected_cal_score': calibrated, 'router_confidence': r['confidence'], 'router_margin': r['margin']})
out = pd.DataFrame(rows)
out.to_csv(ROOT / 'router_moe_cache.csv', index=False)
print()
print('=' * 90)
print('FULL ROUTER-MOE CACHE')
print('=' * 90)
print('Expected =', len(paths))
print('Cache    =', len(out))
print('Missing  =', len(missing))
if missing:
    print('First missing:', missing[0])
    raise RuntimeError('Router cache incomplete')
print()
print('ROUTE DISTRIBUTION')
dist = out['selected_expert'].value_counts(normalize=True).reindex(EXPERTS, fill_value=0)
for (e, frac) in dist.items():
    print(f'{e:<10} {frac * 100:6.2f}%')
print()
print('Mean confidence =', f"{out['router_confidence'].mean():.4f}")
print('Mean margin     =', f"{out['router_margin'].mean():.4f}")
print()
print('ROUTER CACHE READY ✅')
