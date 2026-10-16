"""Runner for the provenance experiment (single command per phase, fail-closed).

Phases
------
  calib   : clean/planted_a/planted_b/random states on the calibration split;
            discover the candidate direction v from the predeclared P1-minus-M
            contrast; freeze it (hashed) for the holdout.
  holdout : on untouched items test R1 (content exclusion), R2 (damage
            exclusion) and R3 (causal transplant) against a zero sham and a
            norm-matched random-direction transplant. Requires the frozen
            direction file of a completed calib phase.

  uv run python -m experiments.provenance.runner --phase calib
  uv run python -m experiments.provenance.runner --phase holdout [--direction ...]

FP32 by policy: a single-coordinate transplant must be exactly representable,
so a dtype mismatch raises before any experiment runs. Artifacts are
append-only records plus one hashed frozen-direction file; no phase completes
successfully unless every gate was recorded true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.provenance.protocol import (
    Config,
    DATASET,
    DTYPE_POLICY,
    JUDGE_LABELS,
    N_HOLDOUT_MIN,
    nearest_scale,
    pick_direction,
    question_prompt,
    random_unit_direction,
    sha256_json,
    support_match_ok,
)


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
class Operator:
    """Scales MLP outputs of specifiable layer blocks at chosen positions.
    Hooks are attached to every layer below the readout; `scope` selects
    which layers, positions, and strength are active per forward pass."""

    def __init__(self, model, layers):
        self.scale = None
        self.pos = None
        self.active = None
        self._handles = []
        for i in layers:
            self._handles.append((i, model.model.layers[i].mlp.register_forward_hook(self._make(i))))

    def _make(self, idx):
        def hook(module, args, out):
            if self.scale is None or self.pos is None or self.active is None or idx not in self.active:
                return out
            t = out[0] if isinstance(out, tuple) else out
            scaled = t.clone()
            scaled[:, self.pos, :] = scaled[:, self.pos, :] * self.scale
            return (scaled,) + tuple(out[1:]) if isinstance(out, tuple) else scaled
        return hook

    def scope(self, scale, pos, olayers):
        self.scale, self.pos, self.active = scale, pos, tuple(olayers)
        return self

    def clear(self):
        self.scale, self.pos, self.active = None, None, None

    def remove(self):
        for _, h in self._handles:
            h.remove()


class Readout:
    """Captures one block output; optionally additively patches one position."""

    def __init__(self, model, layer_idx):
        self.h = None
        self.patch = None
        self.pos = None
        self._handle = model.model.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, args, out):
        is_tuple = isinstance(out, tuple)
        t = out[0] if is_tuple else out
        if self.patch is not None and self.pos is not None:
            t = t.clone()
            t[:, self.pos, :] = t[:, self.pos, :] + self.patch.to(t.dtype)
            self.h = t.detach()
            return (t,) + tuple(out[1:]) if is_tuple else t
        self.h = t.detach()
        return out

    def clear(self):
        self.patch, self.pos = None, None

    def remove(self):
        self._handle.remove()


# ---------------------------------------------------------------------------
# Forward utilities (batch 1, inference mode)
# ---------------------------------------------------------------------------
def seq_logprob(model, ids_prompt, ids_cont, op=None, scale=None, pos=None, olayers=None) -> float:
    total = ids_prompt + ids_cont
    ids = torch.tensor([total], device=model.device)
    if scale is not None and pos is not None and op is not None:
        op.scope(scale, pos, olayers or ())
    try:
        logits = model(ids).logits
    finally:
        if op is not None:
            op.clear()
    n0 = len(ids_prompt)
    lp = torch.log_softmax(logits[0, n0 - 1: n0 - 1 + len(ids_cont)], dim=-1)
    tgt = torch.tensor(ids_cont, device=logits.device, dtype=torch.long)
    return float(lp.gather(-1, tgt[:, None]).sum())


def greedy_answer(model, tok, question: str, max_new: int) -> str:
    ids = torch.tensor([tok(question_prompt(question)).input_ids], device=model.device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def judge_geometry(tok, question: str, answer: str):
    prefix = f"Q: {question}\nA: "
    enc = tok(prefix + f"{answer}\nWas your answer CORRECT or INCORRECT?\nAnswer:",
              return_offsets_mapping=True)
    start = len(prefix)
    ans_pos = [i for i, (a, b) in enumerate(enc.offset_mapping) if b > start and a < start + len(answer)]
    ids = enc.input_ids
    return ids, ans_pos, len(ids) - 1


def margin(model, tok, ids_judge, op=None, scale=None, pos=None, olayers=None) -> float:
    vals = []
    for lab in JUDGE_LABELS:
        vals.append(seq_logprob(model, ids_judge, tok(lab, add_special_tokens=False).input_ids,
                                op, scale, pos, olayers))
    return vals[0] - vals[1]


def margin_patched(model, tok, ids_judge, readout, delta, pos, op=None, osc=None, opos=None, olayers=None) -> float:
    vals = []
    for lab in JUDGE_LABELS:
        cont = tok(lab, add_special_tokens=False).input_ids
        readout.patch, readout.pos = delta, pos
        try:
            vals.append(seq_logprob(model, ids_judge, cont, op, osc, opos, olayers))
        finally:
            readout.clear()
    return vals[0] - vals[1]


def probe_scoped(model, ids_judge, readout, op, scale, pos, olayers=None):
    """One judgment forward; returns the readout-layer residual at the final
    judgment-prompt token (which is the last token before the label)."""
    ids = torch.tensor([ids_judge], device=model.device)
    if scale is not None and pos is not None:
        op.scope(scale, pos, olayers or ())
    try:
        _ = model(ids)
    finally:
        op.clear()
    return readout.h[0, -1]


def item_rng(item_idx, seed) -> random.Random:
    return random.Random(hashlib.sha256(f"{seed}:{item_idx}".encode()).hexdigest())


# ---------------------------------------------------------------------------
# Per-item construction, shared by both phases
# ---------------------------------------------------------------------------
def scan_operator(model, ids_q, ans_ids, op, block_layers, sup_clean, tau_drop, cfg):
    cont_pos = list(range(len(ids_q), len(ids_q) + len(ans_ids)))
    grid = {s: seq_logprob(model, ids_q, ans_ids, op, s, cont_pos, block_layers) for s in cfg.scale_grid}
    scale, support, drop = nearest_scale(sup_clean, cfg.scale_grid, grid.get)
    return dict(layers=list(block_layers), scale=scale,
                support=round(support, 6), drop=round(drop, 6))


def build_item(model, tok, op, item_idx, cfg, layers):
    question, key = DATASET[item_idx][1], DATASET[item_idx][2]
    ans = greedy_answer(model, tok, question, cfg.gen_max_new)
    rec = dict(item=item_idx, domain=DATASET[item_idx][0], question=question,
               key=key, answer_text=ans, eligible=key.lower() in ans.lower())
    if not rec["eligible"]:
        rec["excluded_reason"] = "greedy answer missing key"
        return rec, None

    ids_q = tok(question_prompt(question)).input_ids
    ans_ids = tok(" " + ans, add_special_tokens=False).input_ids
    sup_clean = seq_logprob(model, ids_q, ans_ids)
    tau_drop = min(1.0, 0.35 * sup_clean)
    ids_judge, ans_pos, readout_pos = judge_geometry(tok, question, ans)

    rng = item_rng(item_idx, cfg.seed)
    m_layers = rng.sample([i for i in range(layers["readout"])
                           if i not in layers["op_a"] and i not in layers["op_b"]],
                          len(layers["op_a"]))

    rec.update(dict(
        sup_clean=round(sup_clean, 6), tau_drop=round(tau_drop, 6),
        op_a=scan_operator(model, ids_q, ans_ids, op, layers["op_a"], sup_clean, tau_drop, cfg),
        op_b=scan_operator(model, ids_q, ans_ids, op, layers["op_b"], sup_clean, tau_drop, cfg),
        op_m=scan_operator(model, ids_q, ans_ids, op, m_layers, sup_clean, tau_drop, cfg),
        readout_layer=layers["readout"], readout_pos=readout_pos,
    ))
    ctx = dict(ids_judge=ids_judge, ans_pos=ans_pos, readout_pos=readout_pos)
    return rec, ctx


def condition_scores(model, op, readout, rec, ctx, layers, U=None, v=None):
    """Returns (scores, hstore): scores are lists (calib) or floats (holdout).
    hstore (calib only) keeps the FULL readout-layer residual [T, D] per
    condition so per-position engagement can be audited."""
    ids_judge, ans_pos, rp = ctx["ids_judge"], ctx["ans_pos"], ctx["readout_pos"]
    out, hstore = {}, {}
    for name, blk, scl in (
        ("clean", None, None),
        ("planted_a", layers["op_a"], rec["op_a"]["scale"]),
        ("planted_b", layers["op_b"], rec["op_b"]["scale"]),
        ("random", tuple(rec["op_m"]["layers"]), rec["op_m"]["scale"]),
    ):
        if blk is not None:
            op.scope(scl, ans_pos, tuple(blk))
        try:
            _ = model(torch.tensor([ids_judge], device=model.device))
        finally:
            op.clear()
        t = readout.h[0].detach().clone()          # [T, D]
        if U is not None:
            hstore[name] = t
            out[name] = [round(x, 5) for x in (t[rp] @ U.T).tolist()]
        else:
            out[name] = float(torch.dot(v.to(t.dtype).to(t.device), t[rp]))
    return out, (hstore if U is not None else None)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def run_calib(model, tok, op, readout, cfg, layers, run_dir, limit):
    U = model.lm_head.weight.detach()[: cfg.candidate_vocab].float()
    U = U / U.norm(dim=1, keepdim=True)
    rows, records, engagement = [], [], []
    for idx in range(min(limit or cfg.n_calib, cfg.n_calib)):
        rec, ctx = build_item(model, tok, op, idx, cfg, layers)
        if ctx is None:
            records.append(rec)
            continue
        sc, hh = condition_scores(model, op, readout, rec, ctx, layers, U=U)
        diff = torch.tensor(sc["planted_a"]) - torch.tensor(sc["random"])
        rows.append(diff)
        rec["scores_planted_minus_random"] = diff.tolist()
        rec["diag"] = dict(
            h_clean_norm=round(float(hh["clean"].norm()), 6),
            l2_planted_clean_rel=round(float((hh["planted_a"] - hh["clean"]).norm() / max(hh["clean"].norm(), 1e-9)), 6),
            l2_planted_random_rel=round(float((hh["planted_a"] - hh["random"]).norm() / max(hh["random"].norm(), 1e-9)), 6),
        )
        ia = torch.tensor(ctx["ans_pos"], dtype=torch.long)
        hc, hp = hh["clean"], hh["planted_a"]
        rec["diag"]["l2_ansspan_rel"] = round(float((hp[ia] - hc[ia]).norm() / max(hc[ia].norm(), 1e-9)), 6)
        rec["diag"]["readout_pos"] = ctx["readout_pos"]
        engagement.append(rec["diag"]["l2_planted_clean_rel"])
        records.append(rec)
        print(f"calib {idx}: eligible, sep_mean={float(diff.mean()):.4f} "
              f"sep_max={float(diff.abs().max()):.4f} engage={rec['diag']['l2_planted_clean_rel']:.4f} "
              f"ansspan={rec['diag']['l2_ansspan_rel']:.4f}")

    eligible = [r for r in records if r.get("eligible")]
    gates = dict(dtype_fp32=model.dtype is torch.float32,
                 calib_eligible_min=len(eligible) >= 16,
                 operator_engagement=(min(engagement, default=0.0) >= cfg.engagement_min))
    summary = None
    if gates["calib_eligible_min"] and gates["operator_engagement"]:
        sel = pick_direction(torch.stack(rows))
        gates["sign_consistency_min"] = sel["calib_sign_consistency"] >= 0.7
        if gates["sign_consistency_min"]:
            dpath = run_dir / "frozen_direction.pt"
            vec = U[sel["candidate_token"]].detach().clone()
            torch.save(dict(token=sel["candidate_token"], vec=vec.cpu(),
                            readout_layer=layers["readout"],
                            candidate_vocab=cfg.candidate_vocab), dpath)
            sha = hashlib.sha256(dpath.read_bytes()).hexdigest()
            (run_dir / "frozen_direction.json").write_text(json.dumps(dict(
                frozen_direction_sha=sha,
                config_sha=sha256_json(json.loads(json.dumps(cfg.__dict__, default=str))),
                model_id=cfg.model_id, layers=layers, direction=sel), indent=2))
            summary = dict(direction=sel, frozen_direction_sha=sha)
    write_jsonl(records, run_dir / "records.jsonl")
    return gates, summary


def run_holdout(model, tok, op, readout, cfg, layers, run_dir, limit, v):
    records, rel_errs, shams = [], [], []
    n_elig = 0
    for idx in range(cfg.n_calib, min(limit or len(DATASET), len(DATASET))):
        rec, ctx = build_item(model, tok, op, idx, cfg, layers)
        if ctx is None:
            records.append(rec)
            continue
        n_elig += 1
        sc, _hh = condition_scores(model, op, readout, rec, ctx, layers, v=v)
        rec["support_match_ok"] = (support_match_ok(rec["op_a"]["drop"], rec["op_m"]["drop"], cfg.match_rel_tol)
                                   and support_match_ok(rec["op_a"]["drop"], rec["op_b"]["drop"], cfg.match_rel_tol))
        if rec["support_match_ok"]:
            ids_judge = ctx["ids_judge"]
            osc, opos, olayers = rec["op_a"]["scale"], ctx["ans_pos"], layers["op_a"]
            delta = sc["clean"] - sc["planted_a"]
            m_p = margin(model, tok, ids_judge, op, osc, opos, olayers)
            vec = v.to(model.device) * delta
            m_rest = margin_patched(model, tok, ids_judge, readout, vec, ctx["readout_pos"], op, osc, opos, olayers)
            realized = float(torch.dot(v.to(model.device), readout.h[0, ctx["readout_pos"]]))
            rel = abs(realized - (sc["planted_a"] + delta)) / max(abs(delta), 1e-9)
            rel_errs.append((abs(realized - (sc["planted_a"] + delta)), rel))
            m_sham = margin_patched(model, tok, ids_judge, readout,
                                    torch.zeros_like(vec), ctx["readout_pos"], op, osc, opos, olayers)
            shams.append(abs(m_sham - m_p))
            rvec = random_unit_direction(v.numel(), item_rng(idx, cfg.seed + 1), model.device)
            m_rand = margin_patched(model, tok, ids_judge, readout,
                                    rvec * abs(delta), ctx["readout_pos"], op, osc, opos, olayers)
            rec.update(dict(score_clean=sc["clean"], score_planted_a=sc["planted_a"],
                            score_planted_b=sc["planted_b"], score_random=sc["random"],
                            delta=round(delta, 6),
                            margin_clean=round(margin(model, tok, ids_judge), 6),
                            margin_planted_a=round(m_p, 6),
                            margin_restored=round(m_rest, 6),
                            margin_sham=round(m_sham, 6),
                            margin_random_transplant=round(m_rand, 6),
                            delta_realized=round(realized, 9),
                            patch_abs_err=round(abs(realized - (sc["planted_a"] + delta)), 12),
                            patch_rel_err=round(rel, 9)))
        records.append(rec)
        print(f"holdout {idx}: match={rec.get('support_match_ok')}")

    matched = [r for r in records if r.get("support_match_ok")]
    abs_errs = [a for a, _ in rel_errs]
    gates = dict(dtype_fp32=model.dtype is torch.float32,
                 frozen_direction_verified=True,
                 matched_holdout_min=len(matched) >= N_HOLDOUT_MIN,
                 support_match_fraction=(len(matched) / max(n_elig, 1)) >= cfg.match_fraction_min,
                 sham_parity=(all(s <= cfg.sham_abs_tol for s in shams) if shams else False),
                 patch_geometry=(max(abs_errs, default=1.0) <= cfg.patch_rel_tol))
    write_jsonl(records, run_dir / "records.jsonl")
    return gates, dict(equeligible=n_elig, matched=len(matched))


# ---------------------------------------------------------------------------
# I/O and CLI
# ---------------------------------------------------------------------------
def write_jsonl(records, path: Path):
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def source_hashes() -> dict:
    base = Path(__file__).resolve().parent
    return {n: hashlib.sha256((base / n).read_bytes()).hexdigest()
            for n in ("protocol.py", "runner.py", "analysis.py")
            if (base / n).exists()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["calib", "holdout"], required=True)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--direction", default=None)
    args = ap.parse_args()

    cfg = Config(model_id=args.model)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    run_dir = Path(args.out) if args.out else Path("assets") / "provenance-v1" / f"{stamp}_{args.phase}"
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    assert model.dtype is torch.float32, f"dtype policy violated: {model.dtype}"

    layers = cfg.resolve_layers(model.config.num_hidden_layers)
    op = Operator(model, list(range(0, layers["readout"])))   # hooks on every layer below readout
    readout = Readout(model, layers["readout"])
    (run_dir / "run_manifest.json").write_text(json.dumps(dict(
        phase=args.phase, model_id=cfg.model_id, device=device, layers=layers,
        torch=torch.__version__, dtype_policy=DTYPE_POLICY,
        config=json.loads(json.dumps(cfg.__dict__, default=str)),
        source_hashes=source_hashes()), indent=2, default=str))
    try:
        with torch.inference_mode():
            if args.phase == "calib":
                gates, summary = run_calib(model, tok, op, readout, cfg, layers, run_dir, args.limit)
                status = dict(phase="calib", gates=gates, summary=summary, complete=bool(summary))
            else:
                with torch.inference_mode():
                    dpath = Path(args.direction) if args.direction else run_dir / "frozen_direction.pt"
                    meta_path = dpath.with_name("frozen_direction.json")
                    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                    sha_ok = dpath.exists() and hashlib.sha256(dpath.read_bytes()).hexdigest() == meta.get("frozen_direction_sha")
                    layers_ok = meta.get("layers") == layers
                    if not (sha_ok and layers_ok):
                        status = dict(phase="holdout", gates={"frozen_direction_verified": sha_ok,
                                                              "resolved_layers_match": layers_ok},
                                      complete=False,
                                      error="missing/unverified frozen direction or layer mismatch - run calib first")
                    else:
                        payload = torch.load(dpath, map_location="cpu", weights_only=True)
                        gates, summary = run_holdout(model, tok, op, readout, cfg, layers, run_dir, args.limit, payload["vec"].float())
                        status = dict(phase="holdout", gates=gates, summary=summary,
                                      complete=all(gates.values()))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status = dict(phase=args.phase, gates={"unhandled_exception": False},
                      complete=False, error=repr(e))
    finally:
        op.remove(), readout.remove()

    status["complete_ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    (run_dir / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2, default=str)[:2000])
    return 0 if status.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
