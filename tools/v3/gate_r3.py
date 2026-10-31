"""ROOT_FIX GATE R3: Router V3 vs Router v2 on val_teacher_V3 (utilities from calibrator V3, both routers scored
with the same targets). Pass: macro val utility over FFPP/GANGen/StyleGAN_FFHQ not lower than v2 by > 0.5 points,
and Router V3 sends >= 80% of DeepFakeFace-val fakes to Diffusion.

    python tools/v3/gate_r3.py --v3 /home/aiotlab/hoangan/outputs/ROUTER4_V3_20261030 --router-v3 BEST.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sft_v3 import route_batch  # noqa: E402

V2_ROUTER = "/home/aiotlab/hoangan/outputs/router_training_v2/run_effb0_3domain/best.pt"
OLD = ["FFPP", "GANGen", "StyleGAN_FFHQ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, required=True)
    ap.add_argument("--router-v3", required=True)
    a = ap.parse_args()
    val = pd.read_csv(a.v3 / "02_teacher/val_teacher_V3.csv")
    rep = {}
    for name, ckpt in (("v2", V2_ROUTER), ("V3", a.router_v3)):
        sel = route_batch(val.image_path.tolist(), ckpt)
        val[f"{name}_sel"] = sel
        u = val.apply(lambda r: r[f"utility_{r[f'{name}_sel']}"], axis=1)
        per = {ds: float(u[val.dataset == ds].mean()) for ds in val.dataset.unique()}
        fk = (val.dataset == "DeepFakeFace") & (val.label == 1)
        rep[name] = dict(utility_by_domain=per, macro_old3=float(np.mean([per[d] for d in OLD])),
                         macro_all=float(np.mean(list(per.values()))),
                         diffusion_on_dff_fake=float((val.loc[fk, f"{name}_sel"] == "diffusion").mean()),
                         top1_vs_teacher={ds: float((val.loc[val.dataset == ds, f"{name}_sel"] ==
                                                     val.loc[val.dataset == ds, "teacher_hard"]).mean())
                                          for ds in val.dataset.unique()})
    d = (rep["V3"]["macro_old3"] - rep["v2"]["macro_old3"]) * 100
    rep["gate"] = dict(delta_macro_old3_pts=d, cond_utility=d >= -0.5,
                       cond_diffusion=rep["V3"]["diffusion_on_dff_fake"] >= 0.80)
    rep["gate"]["pass"] = rep["gate"]["cond_utility"] and rep["gate"]["cond_diffusion"]
    (a.v3 / "03_router/GATE_R3.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
