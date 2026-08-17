"""Client: pull global model, train locally, push the delta. Repeat until done.

Tier and client-id come from the environment (set by docker-compose / the
launcher). Data shard is client_<rank>.parquet under the partition dir. The
client is intentionally dumb about strategy — asynchrony lives entirely on the
server; the client just loops pull/train/push as fast as its cgroup allows,
which is what produces real wall-clock heterogeneity.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import requests
import torch
import torch.nn as nn

from ..common import get_params, param_shapes, set_params, sub
from ..compression import decode, encode
from ..data import make_loader
from ..model import build_model


def train_local(model, loader, steps, lr, device):
    """Do a fixed number of minibatch steps (cycling the shard).

    Bounded local work — not full epochs over the shard — keeps per-round deltas
    moderate and comparable across the wildly different shard sizes a Dirichlet
    split produces, and is the standard local-work model in async FL.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    model.train()
    n_seen = 0
    it = iter(loader)
    for _ in range(steps):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(loader)
            X, y = next(it)
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss = lossf(model(X), y)
        loss.backward()
        opt.step()
        n_seen += len(y)
    return n_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=os.environ.get("TIER", "fast"))
    args = ap.parse_args()

    server = os.environ["SERVER_ADDR"]
    cid = os.environ["CLIENT_ID"]
    rank = int(os.environ["CLIENT_RANK"])
    part_dir = os.environ["PARTITION_DIR"]
    in_dim = int(os.environ["IN_DIM"])
    n_classes = int(os.environ["N_CLASSES"])
    steps = int(os.environ.get("LOCAL_STEPS", "30"))
    lr = float(os.environ.get("LOCAL_LR", "0.05"))
    batch = int(os.environ.get("BATCH", "256"))
    scheme = os.environ.get("COMPRESSION", "none")
    topk = float(os.environ.get("TOPK_FRAC", "0.10"))
    model_size = os.environ.get("MODEL_SIZE", "small")

    # Pin torch to the tier's CPU budget: an edge device does not see 96 cores,
    # and without this 12 containers each spawn 96 threads and thrash the host.
    threads = max(1, int(float(os.environ.get("CPU_BUDGET", "1"))))
    torch.set_num_threads(threads)

    device = "cpu"  # clients emulate edge hardware; CPU-only by design
    model = build_model(in_dim, n_classes, model_size).to(device)
    shapes = param_shapes(model)
    loader = make_loader(f"{part_dir}/client_{rank:02d}.parquet", batch, shuffle=True)
    base = f"http://{server}"

    print(f"[client {cid}] tier={args.tier} rank={rank} shard={len(loader.dataset)}",
          flush=True)

    last_version = -1
    while True:
        try:
            r = requests.get(f"{base}/pull", params={"after_version": last_version},
                             timeout=180)
        except requests.RequestException:
            time.sleep(1.0); continue
        if r.headers.get("X-Done") == "1":
            print(f"[client {cid}] server done, exiting", flush=True)
            return
        version = int(r.headers["X-Version"])
        global_params = decode(r.content, shapes)
        set_params(model, global_params)

        train_local(model, loader, steps, lr, device)
        delta = sub(get_params(model), global_params)
        blob = encode(delta, scheme, topk)

        try:
            resp = requests.post(f"{base}/push", data=blob, headers={
                "x-client-id": cid, "x-base-version": str(version),
                "x-n-samples": str(len(loader.dataset)),
                "content-type": "application/octet-stream"}, timeout=180)
        except requests.RequestException:
            time.sleep(1.0); continue
        last_version = version
        if resp.json().get("done"):
            print(f"[client {cid}] done", flush=True)
            return


if __name__ == "__main__":
    main()
