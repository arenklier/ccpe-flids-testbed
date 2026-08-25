"""Asynchronous parameter server implementing four aggregation strategies.

The server owns the global model and a monotonically increasing ``version``.
Clients PULL (version, model), train locally, and PUSH a delta tagged with the
base version they trained on. Staleness = current_version - base_version at
apply time. This makes wall-clock asynchrony and staleness first-class, which
round-based frameworks cannot express.

Strategies
----------
sync      FedAvg with a full-participation barrier: buffer every client's delta
          for the round, average (data-weighted), apply, bump version. PULL
          long-polls until the round advances, so all clients train on the same
          version each round; round time is set by the slowest client.
fedasync  Apply each delta on arrival: g <- g + eta * s(tau) * delta, with
          polynomial staleness discount s(tau)=(1+tau)^(-a) (Hu et al., 2019).
fedbuff   Buffer K deltas from any clients, average, apply, bump (Nguyen 2022).
staleness ours: buffered async like FedBuff but each buffered delta is weighted
          by BOTH data size and a polynomial staleness discount before
          averaging: w_i = n_i * (1+tau_i)^(-a). See paper Sec. 4.

All mutation happens under one lock; run uvicorn with a single worker.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from ..common import add, get_params, param_shapes, set_params, sub
from ..compression import decode, encode
from ..data import make_loader
from ..model import build_model


class ParameterServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.strategy = cfg["strategy"]
        self.eta = cfg.get("server_lr", 1.0)
        self.stale_a = cfg.get("staleness_a", 0.5)
        # exponent on the data-size term; 0 = FedBuff, 1 = the plain
        # data-size rule. "clip" caps each client at the median shard.
        self.size_beta = float(cfg.get("size_beta", 1.0))
        self.size_clip = bool(cfg.get("size_clip", False))
        self.buffer_k = cfg.get("buffer_k", 4)
        self.n_clients = cfg["n_clients"]
        self.scheme = cfg.get("compression", "none")
        self.topk_frac = cfg.get("topk_frac", 0.10)
        self.target_f1 = cfg.get("target_macro_f1", 0.80)
        self.eval_every = cfg.get("eval_every", 5)
        self.max_version = cfg.get("max_version", 2000)
        self.max_seconds = cfg.get("max_seconds", 7200)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = build_model(cfg["in_dim"], cfg["n_classes"],
                                 cfg.get("model_size", "small")).to(device)
        self.shapes = param_shapes(self.model)
        self.global_params = get_params(self.model)
        self.version = 0

        # test set for server-side evaluation. The cadence checkpoints use a
        # class-stratified subsample so that a checkpoint is cheap enough to
        # take often; the full split is scored once when the run finishes.
        self.test_loader = make_loader(cfg["test_path"], 4096, shuffle=False)
        self.eval_cap = int(cfg.get("eval_cap", 0))
        self.fast_loader = self._make_stratified_loader(self.eval_cap)

        # async/buffered state
        # time spent queueing for, and inside, the aggregator lock
        self.lock_wait_s = 0.0
        self.lock_hold_s = 0.0
        self._seen_sizes: list[int] = []
        self.buffer: list[tuple[list[np.ndarray], float, int]] = []  # (delta, weight, n)
        self.round_pushes: dict[str, tuple[list[np.ndarray], int]] = {}

        # accounting
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.t0: float | None = None
        self.downlink_bytes = 0
        self.uplink_bytes = 0
        self.history: list[dict] = []
        self.done = False
        # per-push trace: (wall_s, client_id, staleness). Lets us reconstruct
        # the arrival pattern and staleness distribution after the fact, which
        # is what distinguishes the aggregation strategies operationally.
        self.push_log: list[tuple[float, str, int]] = []

        # Evaluation runs on its own thread so it never blocks aggregation;
        # it needs its own model instance so it cannot race the one used
        # elsewhere. See _bump_and_eval for why this matters for fairness.
        self.eval_model = build_model(cfg["in_dim"], cfg["n_classes"],
                                      cfg.get("model_size", "small")).to(device)
        self._eval_q: "queue.Queue" = queue.Queue()
        self._eval_busy = False
        threading.Thread(target=self._eval_worker, daemon=True).start()

        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- wire ----------------------------------------------------------
    def pull_blob(self) -> bytes:
        # downlink is always lossless; compression is studied on the uplink
        return encode(self.global_params, "none")

    # ---- apply strategies ---------------------------------------------
    def _bump_and_eval(self):
        """Advance the version and, on a checkpoint, hand evaluation off.

        Evaluation used to run inline here, holding the aggregator lock for
        the duration of a full pass over the test split (~3 s). That charged
        every strategy an overhead proportional to how often it bumped the
        version, which is 3x more often for the asynchronous rules than for
        the synchronous barrier -- a systematic bias against exactly the
        strategies under test. We now copy the parameters under the lock (a
        memcpy) and let a worker thread do the inference off the critical
        path. The timestamp recorded is the snapshot time, not the time the
        evaluation finishes, so wall-clock stays honest.

        At most one evaluation is in flight: if the worker is still busy at
        the next checkpoint we skip it rather than queue snapshots, which
        bounds memory and keeps the aggregator from ever waiting on evaluation.
        """
        self.version += 1
        if self.version % self.eval_every == 0 and not self._eval_busy:
            self._eval_busy = True
            self._eval_q.put((self.version,
                              (time.monotonic() - self.t0) if self.t0 else 0.0,
                              [p.copy() for p in self.global_params],
                              self.downlink_bytes, self.uplink_bytes))
        if self.version >= self.max_version:
            self.done = True

    def _eval_worker(self):
        while True:
            item = self._eval_q.get()
            if item is None:
                return
            version, elapsed, snap, dn_bytes, up_bytes = item
            rec = self._evaluate_snapshot(version, elapsed, snap, dn_bytes, up_bytes)
            with self.cond:
                self.history.append(rec)
                self._eval_busy = False
                if rec["macro_f1"] >= self.target_f1:
                    self.done = True
                self.cond.notify_all()

    def _apply_delta(self, delta, weight_scale=1.0):
        self.global_params = add(self.global_params, delta, self.eta * weight_scale)

    def apply_push(self, client_id, base_version, delta, n_samples):
        staleness = self.version - base_version
        self.push_log.append((round((time.monotonic() - self.t0) if self.t0 else 0.0, 3),
                              client_id, int(max(0, staleness))))

        if self.strategy == "sync":
            self.round_pushes[client_id] = (delta, n_samples)
            if len(self.round_pushes) >= self.n_clients:
                tot = sum(n for _, n in self.round_pushes.values()) or 1
                agg = [np.zeros(s, dtype=np.float32) for s in self.shapes]
                for d, n in self.round_pushes.values():
                    for i, dd in enumerate(d):
                        agg[i] += (n / tot) * dd
                self._apply_delta(agg)
                self.round_pushes.clear()
                self._bump_and_eval()
                self.cond.notify_all()

        elif self.strategy == "fedasync":
            s = (1.0 + max(0, staleness)) ** (-self.stale_a)
            self._apply_delta(delta, weight_scale=s)
            self._bump_and_eval()
            self.cond.notify_all()

        elif self.strategy == "fedbuff":
            # FedBuff as published (Nguyen et al. 2022, Sec. 5 "Staleness
            # scaling"): buffered updates are down-weighted by the same
            # polynomial discount s(tau)=(1+tau)^-a that FedAsync uses, with
            # no data-size term. Passing n=1 makes w_i = s(tau_i) exactly.
            s = (1.0 + max(0, staleness)) ** (-self.stale_a)
            self.buffer.append((delta, s, 1))
            if len(self.buffer) >= self.buffer_k:
                self._flush_buffer(use_staleness=True)

        elif self.strategy == "fedbuff_ns":
            # Staleness-agnostic buffered average (data-size weights only).
            # Not FedBuff: kept as an ablation that isolates what the
            # staleness discount contributes on its own.
            self.buffer.append((delta, 1.0, n_samples))
            if len(self.buffer) >= self.buffer_k:
                self._flush_buffer(use_staleness=False)

        elif self.strategy == "staleness":
            s = (1.0 + max(0, staleness)) ** (-self.stale_a)
            self.buffer.append((delta, s, self._size_term(n_samples)))
            if len(self.buffer) >= self.buffer_k:
                self._flush_buffer(use_staleness=True)
        else:
            raise ValueError(self.strategy)

    def _size_term(self, n):
        """f(n) for the buffered weight: n**beta, optionally median-clipped."""
        self._seen_sizes.append(int(n))
        if self.size_clip:
            med = float(np.median(self._seen_sizes))
            return float(min(n, med))
        if self.size_beta == 1.0:
            return float(n)
        return float(n) ** self.size_beta

    def _flush_buffer(self, use_staleness: bool):
        if use_staleness:
            weights = np.array([w * n for _, w, n in self.buffer], dtype=np.float64)
        else:
            weights = np.array([n for _, _, n in self.buffer], dtype=np.float64)
        weights = weights / (weights.sum() or 1.0)
        agg = [np.zeros(s, dtype=np.float32) for s in self.shapes]
        for (delta, _, _), wt in zip(self.buffer, weights):
            for i, d in enumerate(delta):
                agg[i] += wt * d
        self._apply_delta(agg)
        self.buffer.clear()
        self._bump_and_eval()
        self.cond.notify_all()

    # ---- evaluation ----------------------------------------------------
    def _make_stratified_loader(self, cap):
        """At most ``cap`` test rows per class, drawn once and reused.

        Returns None when cap is 0, in which case every checkpoint scores the
        full split, which is the behaviour of earlier versions of this code.
        """
        if cap <= 0:
            return None
        from torch.utils.data import DataLoader, TensorDataset
        X, y = self.test_loader.dataset.tensors
        yn = y.numpy()
        rng = np.random.default_rng(0)
        keep = []
        for c in range(int(self.cfg["n_classes"])):
            idx = np.where(yn == c)[0]
            if len(idx) > cap:
                idx = rng.choice(idx, size=cap, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)
        # precision depends on the class prior, which capping distorts, so each
        # sampled row carries the inverse of its class's sampling rate
        n_cls = int(self.cfg["n_classes"])
        self.eval_weights = np.ones(n_cls, dtype=np.float64)
        kept = yn[keep]
        for c in range(n_cls):
            full_c = int((yn == c).sum())
            kept_c = int((kept == c).sum())
            self.eval_weights[c] = (full_c / kept_c) if kept_c else 0.0
        print(f"[eval] stratified checkpoint set: {len(keep)} of {len(yn)} rows",
              flush=True)
        return DataLoader(TensorDataset(X[keep], y[keep]), batch_size=4096,
                          shuffle=False, num_workers=0)

    def _evaluate_snapshot(self, version, elapsed, params, dn_bytes, up_bytes,
                           full=False):
        """Score a parameter snapshot. Runs on the worker thread, no lock held."""
        set_params(self.eval_model, params)
        self.eval_model.eval()
        n_classes = self.cfg["n_classes"]
        tp = np.zeros(n_classes); fp = np.zeros(n_classes); fn = np.zeros(n_classes)
        correct = total = 0
        loader = self.test_loader if (full or self.fast_loader is None)             else self.fast_loader
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                pred = self.eval_model(X).argmax(1).cpu().numpy()
                yt = y.numpy()
                w = (np.ones(len(yt)) if (full or self.fast_loader is None)
                     else self.eval_weights[yt])
                correct += float((w * (pred == yt)).sum()); total += float(w.sum())
                for c in range(n_classes):
                    tp[c] += float((w * ((pred == c) & (yt == c))).sum())
                    fp[c] += float((w * ((pred == c) & (yt != c))).sum())
                    fn[c] += float((w * ((pred != c) & (yt == c))).sum())
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        macro_f1 = float(f1.mean()); acc = correct / max(total, 1.0)
        rec = {"version": version, "wall_s": round(elapsed, 2),
               "acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
               "downlink_mb": round(dn_bytes / 1e6, 3),
               "uplink_mb": round(up_bytes / 1e6, 3)}
        print(f"[eval] v{version} f1={macro_f1:.4f} acc={acc:.4f} "
              f"t={elapsed:.1f}s up={rec['uplink_mb']}MB", flush=True)
        return rec

    def _evaluate(self):
        set_params(self.model, self.global_params)
        self.model.eval()
        n_classes = self.cfg["n_classes"]
        tp = np.zeros(n_classes); fp = np.zeros(n_classes); fn = np.zeros(n_classes)
        correct = total = 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X = X.to(self.device)
                pred = self.model(X).argmax(1).cpu().numpy()
                yt = y.numpy()
                correct += int((pred == yt).sum()); total += len(yt)
                for c in range(n_classes):
                    tp[c] += int(((pred == c) & (yt == c)).sum())
                    fp[c] += int(((pred == c) & (yt != c)).sum())
                    fn[c] += int(((pred != c) & (yt == c)).sum())
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        macro_f1 = float(f1.mean()); acc = correct / max(total, 1)
        elapsed = (time.monotonic() - self.t0) if self.t0 else 0.0
        rec = {"version": self.version, "wall_s": round(elapsed, 2),
               "acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
               "downlink_mb": round(self.downlink_bytes / 1e6, 3),
               "uplink_mb": round(self.uplink_bytes / 1e6, 3)}
        self.history.append(rec)
        print(f"[eval] v{self.version} f1={macro_f1:.4f} acc={acc:.4f} "
              f"t={elapsed:.1f}s up={rec['uplink_mb']}MB", flush=True)
        if macro_f1 >= self.target_f1:
            self.done = True

    def dump(self):
        st = sorted(s for _, _, s in self.push_log)
        pc: dict[str, int] = {}
        for _, cid, _ in self.push_log:
            pc[cid] = pc.get(cid, 0) + 1
        stale_summary = {
            "n_pushes": len(st),
            "mean": round(sum(st) / len(st), 3) if st else 0.0,
            "p50": st[len(st) // 2] if st else 0,
            "p90": st[int(len(st) * 0.9)] if st else 0,
            "max": st[-1] if st else 0,
            "pushes_per_client": pc,
        }
        # score the complete test split once, so the stratified checkpoint set
        # can be checked against the quantity it stands in for
        final_full = final_fast = None
        if self.fast_loader is not None and self.history:
            wall = self.history[-1]["wall_s"]
            try:
                final_full = self._evaluate_snapshot(
                    self.version, wall, self.global_params,
                    self.downlink_bytes, self.uplink_bytes, full=True)
                final_fast = self._evaluate_snapshot(
                    self.version, wall, self.global_params,
                    self.downlink_bytes, self.uplink_bytes, full=False)
            except Exception as exc:                       # never lose a run
                print(f"[eval] final comparison pass failed: {exc}", flush=True)

        out = {"config": self.cfg, "history": self.history,
               "final_version": self.version,
               "reached_target": any(h["macro_f1"] >= self.target_f1 for h in self.history),
               "final_full_eval": final_full,
               "final_fast_eval": final_fast,
               "lock_wait_s": round(self.lock_wait_s, 3),
               "lock_hold_s": round(self.lock_hold_s, 3),
               "staleness_summary": stale_summary,
               "push_log": self.push_log}
        (self.out_dir / "metrics.json").write_text(json.dumps(out, indent=2))
        print(f"[server] wrote {self.out_dir/'metrics.json'}", flush=True)


def make_app(server: ParameterServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"strategy": server.strategy, "version": server.version}

    @app.get("/pull")
    def pull(after_version: int = -1):
        with server.cond:
            if server.strategy == "sync" and after_version >= 0:
                # barrier: block until the round advances or we are done
                t_end = time.monotonic() + 120
                while server.version <= after_version and not server.done:
                    if not server.cond.wait(timeout=t_end - time.monotonic()):
                        break
            blob = server.pull_blob()
            server.downlink_bytes += len(blob)
            headers = {"X-Version": str(server.version),
                       "X-Done": "1" if server.done else "0"}
        return Response(content=blob, media_type="application/octet-stream",
                        headers=headers)

    @app.post("/push")
    async def push(request: Request):
        base_version = int(request.headers["x-base-version"])
        client_id = request.headers["x-client-id"]
        n_samples = int(request.headers["x-n-samples"])
        blob = await request.body()
        delta = decode(blob, server.shapes)
        t_arrive = time.monotonic()
        with server.cond:
            t_enter = time.monotonic()
            if server.t0 is None:
                server.t0 = time.monotonic()
            server.uplink_bytes += len(blob)
            if not server.done:
                server.apply_push(client_id, base_version, delta, n_samples)
            done = server.done
            server.lock_wait_s += t_enter - t_arrive
            server.lock_hold_s += time.monotonic() - t_enter
        return JSONResponse({"version": server.version, "done": done})

    return app
