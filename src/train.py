"""Train the binary real/fake detector with a chosen residual operator.

Mirrors the official NPR training recipe where it matters for comparability:
Adam at lr 2e-4, 50 epochs, LR divided by 10 every 10 epochs, batch 32, and a
BCE-with-logits loss on the single output neuron.

Usage
-----
    python -m src.train --op npr_nearest --epochs 50 --out runs/npr_nearest
    python -m src.train --op laplacian --epochs 50 --out runs/laplacian
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from .build import build_train_loader, build_val_loader, load_config
from .eval.engine import evaluate_loader
from .models.npr_resnet import build_model, count_parameters
from .utils import (AverageMeter, ensure_dir, get_device, save_json,
                    seed_everything, timestamp)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="train NPR-family detector")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--op", default="npr_nearest",
                   help="residual operator name (see src/ops/REGISTRY)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--delr-freq", type=int, default=None,
                   help="divide LR by 10 every N epochs")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", default=None, help="output run directory")
    p.add_argument("--seed", type=int, default=70)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit batches per epoch (smoke testing)")
    p.add_argument("--val-max-batches", type=int, default=None)
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    t = cfg.get("train", {})
    epochs = args.epochs if args.epochs is not None else int(t.get("epochs", 50))
    lr = args.lr if args.lr is not None else float(t.get("lr", 2e-4))
    delr = (args.delr_freq if args.delr_freq is not None
            else int(t.get("delr_freq", 10)))
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg.setdefault("data", {})["num_workers"] = args.workers

    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = ensure_dir(args.out or os.path.join(
        "runs", f"{args.op}-{timestamp()}"))

    model = build_model({**cfg, "op": args.op}).to(device)
    n_par = count_parameters(model)
    print(f"op={args.op}  params={n_par:,}  device={device}")
    print(f"out={os.path.abspath(out_dir)}")

    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg)
    print(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    num_classes = int(cfg.get("model", {}).get("num_classes", 1))
    criterion = (nn.BCEWithLogitsLoss() if num_classes == 1
                 else nn.CrossEntropyLoss())
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val = -1.0
    for epoch in range(epochs):
        if epoch and delr > 0 and epoch % delr == 0:
            for g in optimizer.param_groups:
                g["lr"] /= 10.0
            print(f"[lr] epoch {epoch}: lr -> {optimizer.param_groups[0]['lr']:.2e}")

        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for i, (imgs, targets) in enumerate(train_loader):
            if args.max_batches is not None and i >= args.max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            if num_classes == 1:
                loss = criterion(logits.reshape(-1), targets)
            else:
                loss = criterion(logits, targets.long())
            loss.backward()
            optimizer.step()
            loss_m.update(loss.item(), imgs.size(0))

        vm = evaluate_loader(model, val_loader, device=device,
                             max_batches=args.val_max_batches, progress=False)
        rec = {"epoch": epoch, "train_loss": loss_m.avg,
               "val_acc": vm["acc"], "val_auc": vm["auc"], "val_ap": vm["ap"],
               "lr": optimizer.param_groups[0]["lr"],
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[ep {epoch:3d}] loss={rec['train_loss']:.4f} "
              f"val_acc={vm['acc']*100:5.2f} val_auc={vm['auc']*100:5.2f} "
              f"val_ap={vm['ap']*100:5.2f} ({rec['sec']}s)")

        torch.save({"model": model.state_dict(), "op": args.op,
                    "epoch": epoch, "config": cfg},
                   os.path.join(out_dir, "last.pth"))
        if vm["acc"] > best_val:
            best_val = vm["acc"]
            torch.save({"model": model.state_dict(), "op": args.op,
                        "epoch": epoch, "val_acc": vm["acc"], "config": cfg},
                       os.path.join(out_dir, "best.pth"))

    save_json({"op": args.op, "params": n_par, "epochs": epochs,
               "history": history, "best_val_acc": best_val,
               "config": {k: v for k, v in cfg.items()
                          if not k.startswith("__")}},
              os.path.join(out_dir, "train_log.json"))
    print(f"[done] best_val_acc={best_val*100:.2f}  artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
