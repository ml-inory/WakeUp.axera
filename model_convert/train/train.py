"""Fine-tune the WakeNet9 replica so "Hi,爱芯" triggers and other speech stays silent."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "WakeUp.axera/model_convert"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_code import WakeNet9Train, detect_score, model_thresholds
from fbank import wav_to_fbank


ROOT = Path(__file__).resolve().parent
REF_MODEL = Path(__file__).resolve().parent / "ref_model"
DS = ROOT / "data/dataset.npz"
WINDOW = 64
THR = 0.615
RNG = np.random.default_rng(20260813)


def load_dataset():
    d = np.load(DS, allow_pickle=True)
    feats, labels, meta = d["feats"], d["labels"], d["meta"]
    metas = [json.loads(m) for m in meta]
    train, val = [], []
    for f, l, m in zip(feats, labels, metas):
        if m["kind"] == "pos":
            # Wake-core labeling for "你好，爱芯": split the utterance into
            # voiced segments and label ONLY the last one ("爱芯"). The
            # leading "你好" is a prefix, not the wake core.
            e = np.sqrt(np.clip(f, 0, None).mean(1))
            act = e > max(0.15, e.max() * 0.08)
            segs = []
            inseg = False
            for t, a in enumerate(act):
                if a and not inseg:
                    start = t
                    inseg = True
                elif not a and inseg:
                    segs.append((start, t - 1))
                    inseg = False
            if inseg:
                segs.append((start, len(act) - 1))
            l = np.zeros_like(l)
            if segs:
                s, e2 = segs[-1]
                l[s : e2 + 1] = 1
        rec = {"feats": f, "labels": l, "kind": m["kind"], "voice": m["voice"]}
        (val if m["voice"] in ("zh-CN-YunyangNeural",) else train).append(rec)
    return train, val


def sample_window(rec: dict, force_pos_center: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Random 64-frame window; positives biased to contain voice frames."""
    T = rec["feats"].shape[0]
    if T <= WINDOW:
        feats = rec["feats"]
        labels = rec["labels"]
        if T < WINDOW:
            pad = WINDOW - T
            feats = np.pad(feats, ((0, pad), (0, 0)))
            labels = np.pad(labels, (0, pad))
        return feats.astype(np.float32), labels.astype(np.int64)
    if rec["kind"] == "pos":
        pos_idx = np.where(rec["labels"])[0]
        for _ in range(20):
            # wake word should land anywhere in the middle of the window,
            # preceded AND followed by real audio context (silence or speech).
            if len(pos_idx):
                c = int(RNG.choice(pos_idx))
                # place the wake word at a random offset within the window
                offset = int(RNG.integers(10, 44))
                s = c - offset
                s = min(max(s, 0), max(T - WINDOW, 0))
            else:
                s = int(RNG.integers(0, T - WINDOW + 1))
            if (rec["labels"][s : s + WINDOW] > 0).sum() >= 6:
                return rec["feats"][s : s + WINDOW].astype(np.float32), rec["labels"][s : s + WINDOW].astype(np.int64)
        s = int(RNG.integers(0, T - WINDOW + 1))
    else:
        # negatives: the window must cover the spoken phrase (not just silence
        # or the "Hi" prefix), so the model learns the full phrase is NOT the
        # wake word. Random start around the voice segment.
        pos_idx = np.where(rec["labels"])[0]
        if len(pos_idx):
            a, b = int(pos_idx[0]), int(pos_idx[-1])
            lo = max(0, a - 16)
            hi = min(T - WINDOW, b + 8 - WINDOW + 8)
            if hi >= lo:
                s = int(RNG.integers(lo, hi + 1))
            else:
                s = int(RNG.integers(0, T - WINDOW + 1))
        else:
            s = int(RNG.integers(0, T - WINDOW + 1))
    return rec["feats"][s : s + WINDOW].astype(np.float32), rec["labels"][s : s + WINDOW].astype(np.int64)


def augment(feats: np.ndarray, labels: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Fbank-domain online augmentation: gain, gaussian noise, frame shift."""
    gain = float(RNG.uniform(0.85, 1.15))
    feats = feats * gain
    noise_std = float(RNG.uniform(0.0, 0.05))
    if noise_std:
        feats = feats + RNG.normal(0.0, noise_std, size=feats.shape).astype(np.float32)
    x = torch.from_numpy(feats).permute(1, 0).unsqueeze(0)  # (1,26,T)
    y = torch.from_numpy(labels).unsqueeze(0)  # (1,T)
    return x, y


def wake_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits on the wake channel, matching the trigger threshold scale.

    Positive voice frames are pushed toward wake=1, negative frames toward 0.
    Silence frames in positive windows are down-weighted (they are not a
    "non-wake word" example, just absence of speech).
    """
    w = logits[:, 1, :]  # (B, T) wake logits
    pos = labels == 1
    neg = labels == 0
    pos_loss = F.binary_cross_entropy_with_logits(
        w[pos], torch.ones_like(w[pos]), pos_weight=torch.tensor(2.0)
    )
    # 3-frame trigger needs ~0.2/frame average; BCE alone saturates around
    # logit 2-3, far above the trigger. Add a mild margin push so positive
    # frames stay >= 0.5 (sum3 >= 1.5) even after INT8 quantization.
    margin = F.relu(0.5 - w[pos]).mean()
    neg_loss = F.binary_cross_entropy_with_logits(w[neg], torch.zeros_like(w[neg]))
    eps = 1e-6
    return pos_loss + margin * 0.5 + neg_loss * 1.5 + eps * w.mean()


def eval_split(net, split, name: str) -> dict:
    net.eval()
    pos_hit = neg_hit = pos_n = neg_n = 0
    with torch.no_grad():
        for rec in split:
            x = torch.from_numpy(rec["feats"].T).unsqueeze(0)
            logits = net(x)[0].numpy()
            s = detect_score(logits)
            fire = bool((s > THR).any())
            if rec["kind"] == "pos":
                pos_n += 1
                pos_hit += fire
            else:
                neg_n += 1
                neg_hit += fire
    return {
        f"{name}_pos": f"{pos_hit}/{pos_n} ({pos_hit/max(pos_n,1)*100:.0f}%)",
        f"{name}_neg_false": f"{neg_hit}/{neg_n} ({neg_hit/max(neg_n,1)*100:.0f}%)",
        f"{name}_pos_hit": pos_hit / max(pos_n, 1),
        f"{name}_neg_fp": neg_hit / max(neg_n, 1),
    }


def eval_files(net, files: dict) -> dict:
    net.eval()
    out = {}
    with torch.no_grad():
        for name, path in files.items():
            feats, _ = wav_to_fbank(path)
            x = torch.from_numpy(feats.T).unsqueeze(0)
            logits = net(x)[0].numpy()
            s = detect_score(logits)
            out[f"file_{name}"] = {
                "sum3_max": float(s.max()),
                "fire": bool((s > THR).any()),
                "wake_max": float(logits[1].max()),
            }
    return out


def main():
    torch.set_num_threads(24)
    train, val = load_dataset()
    print(f"train {len(train)}  val {len(val)}  "
          f"(val pos {sum(1 for r in val if r['kind']=='pos')}, neg {sum(1 for r in val if r['kind']=='neg')})")

    net = WakeNet9Train(REF_MODEL)
    thr90, thr95 = model_thresholds(REF_MODEL)
    print(f"thresholds DET90={thr90} DET95={thr95}, using {THR}")

    # baseline before fine-tuning
    print("== baseline (before training) ==")
    print(json.dumps(eval_split(net, train, "train"), ensure_ascii=False))
    print(json.dumps(eval_split(net, val, "val"), ensure_ascii=False))
    files = {
        "nihaoxin_pos": ROOT / "data/accept_nihao_aixin.wav",
        "nihaoxother_neg": ROOT / "data/accept_nihao_qita.wav",
    }
    print(json.dumps(eval_files(net, files), ensure_ascii=False))

    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)

    best = {"score": -1.0}
    log = []
    hard_neg: list = []
    for epoch in range(40):
        net.train()
        losses = []
        # build one epoch: sample 128 windows, half pos half neg
        pos_pool = [r for r in train if r["kind"] == "pos"]
        neg_pool = [r for r in train if r["kind"] == "neg"]
        # hard negative mining: re-evaluate every 3 epochs, sample mined
        # negatives more often in the next rounds
        if epoch % 3 == 0 and epoch > 0:
            net.eval()
            mined = []
            with torch.no_grad():
                for rec in neg_pool:
                    x = torch.from_numpy(rec["feats"].T).unsqueeze(0)
                    logits = net(x)[0].numpy()
                    if (detect_score(logits) > THR).any():
                        mined.append(rec)
            hard_neg = mined
            print(f"  [mine] hard negatives: {len(hard_neg)}")
            net.train()
        for _ in range(64):
            batch_x, batch_y = [], []
            for _ in range(16):
                rec = RNG.choice(pos_pool)
                f, l = sample_window(rec, force_pos_center=True)
                batch_x.append(f)
                batch_y.append(l)
            for _ in range(16):
                pool = hard_neg if (hard_neg and RNG.random() < 0.6) else neg_pool
                rec = RNG.choice(pool)
                f, l = sample_window(rec)
                batch_x.append(f)
                batch_y.append(l)
            xs = torch.cat([augment(f, l)[0] for f, l in zip(batch_x, batch_y)])
            ys = torch.cat([augment(f, l)[1] for f, l in zip(batch_x, batch_y)])
            logp = net(xs)  # (B,2,T)
            loss = wake_loss(logp, ys)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        tr = eval_split(net, train, "train")
        va = eval_split(net, val, "val")
        fres = eval_files(net, files)
        score = va["val_pos_hit"] - 0.5 * va["val_neg_fp"]
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": opt.param_groups[0]["lr"]}
        rec.update(tr)
        rec.update(va)
        rec["files"] = {k: v["fire"] for k, v in fres.items()}
        rec["files_sum3"] = {k: round(v["sum3_max"], 2) for k, v in fres.items()}
        rec["score"] = score
        log.append(rec)
        print(f"ep {epoch:2d} loss {rec['loss']:.4f}  {tr['train_pos']}  {tr['train_neg_false']}  "
              f"{va['val_pos']}  {va['val_neg_false']}  score {score:.3f}  "
              f"files {rec['files']}")
        torch.save(net.state_dict(), ROOT / f"checkpoints/epoch_{epoch:02d}.pt")
        if score > best["score"]:
            best = rec
            torch.save(net.state_dict(), ROOT / "checkpoints/best_model.pt")
            best["epoch"] = epoch

    # final model = last epoch (the best-score pick is often an early,
    # over-triggering checkpoint; file acceptance is what matters)
    final_epoch = len(log) - 1
    net.load_state_dict(torch.load(ROOT / f"checkpoints/epoch_{final_epoch:02d}.pt", map_location="cpu", weights_only=False))
    torch.save(net.state_dict(), ROOT / "checkpoints/final_model.pt")
    print("== best model (after training) ==")
    print(json.dumps(eval_split(net, train, "train"), ensure_ascii=False))
    print(json.dumps(eval_split(net, val, "val"), ensure_ascii=False))
    print(json.dumps(eval_files(net, files), ensure_ascii=False))
    with open(ROOT / "logs/training_log.json", "w") as f:
        json.dump({"log": log, "best": best, "threshold": THR}, f, ensure_ascii=False, indent=2)
    print(f"saved checkpoints/epoch_*.pt + final_model.pt + logs/training_log.json")


if __name__ == "__main__":
    main()
