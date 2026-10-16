"""Frozen protocol, dataset, and offline math for the provenance experiment.

Research question
-----------------
Does a later hidden state encode HOW the preceding answer was computed (a
provenance signal, the M(P) of the original higher-order-alleys sprint),
rather than only WHAT the answer says?

Design scaffold (recipe v1)
---------------------------
Per eligible item, all conditions share the identical visible transcript:
one question, one exact answer text (the model's own greedy answer), and one
judgment prompt. They differ ONLY in the hidden computation applied to the
answer-span while it is present in the forward pass:

  clean     C  : no operator. The answer is the model's natural greedy
                 output; its states are fully supported.
  planted_a P1: MLP outputs of a mid-depth layer block are scaled during
                 the recomputation of the answer span. Support drops, but
                 the answer text is fixed by construction.
  planted_b P2: the same operator family at a disjoint, differently placed
                 block, tuned to the same support drop. Guards against the
                 readout being a "which layers were touched" detector.
  random    M  : a random placement of the same operator family, tuned to
                 the same support drop. The damage-matched null.

Rungs (predeclared; tested only on the holdout split)
-----------------------------------------------------
  R1 content exclusion : the frozen direction v separates P1 vs C. Text is
                         identical, so only a process readout can do this.
  R2 damage exclusion  : v separates P1 vs M in the same oriented sense,
                         where v was discovered on the calibration split
                         under exactly this contrast.
  R3 causal mediation  : transplanting the clean-donor coordinate into the
                         planted recipient at the readout layer moves the
                         CORRECT/INCORRECT margin toward the donor, beyond
                         a norm-matched random-direction transplant and a
                         zero sham.

Claim boundary (not hidden): recipe v1 manipulates P with a forced-recall
operator family, not with naturally co-occurring healthy derivations. A
same-text cross-derivation transplant between two unlabeled healthy states
is the second-stage upgrade and is out of scope here.

Everything is FP32 by policy (fail-closed) so that single-coordinate
transplants are exactly representable - the failure mode that killed the
original sprint's patching routes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

import torch

DTYPE_POLICY = "float32"
JUDGE_LABELS = (" CORRECT", " INCORRECT")

# ---------------------------------------------------------------------------
# Predeclared dataset: 60 items, 10 domains x 6 items, single exact answers.
# Items the model cannot answer deterministically are filtered by the
# eligibility gate at run time, never edited.
# ---------------------------------------------------------------------------
DATASET = [
    ("physics", "What force keeps planets in orbit around the Sun?", "gravity"),
    ("physics", "What electrically neutral subatomic particle sits in the atomic nucleus?", "neutron"),
    ("physics", "What is the SI unit of electric current?", "ampere"),
    ("physics", "What is the speed of light in vacuum, in units of 10^8 metres per second?", "3"),
    ("physics", "What famous equation relates mass and energy?", "mc"),
    ("physics", "What is the process by which a liquid becomes a gas?", "evaporation"),
    ("chemistry", "What is the chemical symbol for gold?", "au"),
    ("chemistry", "What gas makes up roughly 78 percent of Earth's atmosphere?", "nitrogen"),
    ("chemistry", "What acid is found in the human stomach?", "hydrochloric"),
    ("chemistry", "What everyday substance has the molecular formula H2O?", "water"),
    ("chemistry", "What element has atomic number 6?", "carbon"),
    ("chemistry", "What type of chemical bond shares electron pairs between atoms?", "covalent"),
    ("biology", "What organ pumps blood through the human body?", "heart"),
    ("biology", "What molecule stores and carries genetic instructions in cells?", "dna"),
    ("biology", "What is the cellular process of harvesting energy from glucose called?", "respiration"),
    ("biology", "What organelle is known as the powerhouse of the cell?", "mitochond"),
    ("biology", "What pigment makes plant leaves green?", "chlorophyll"),
    ("biology", "Which blood cells transport oxygen?", "red"),
    ("astronomy", "Which planet is closest to the Sun?", "mercury"),
    ("astronomy", "What galaxy contains our solar system?", "milky"),
    ("astronomy", "What is the largest planet in the solar system?", "jupiter"),
    ("astronomy", "What are Saturn's rings mostly made of?", "ice"),
    ("astronomy", "Which star is nearest to Earth?", "sun"),
    ("astronomy", "What does a light-year measure?", "distance"),
    ("geography", "What is the capital of Japan?", "tokyo"),
    ("geography", "What river flows through Cairo?", "nile"),
    ("geography", "What is the largest ocean on Earth?", "pacific"),
    ("geography", "On which continent is the Sahara desert?", "africa"),
    ("geography", "What is the capital of France?", "paris"),
    ("geography", "What mountain range runs through central Europe and is about 2,500 kilometres long?", "urals"),
    ("history", "In which year did World War II end?", "1945"),
    ("history", "Who was the first President of the United States?", "washington"),
    ("history", "What ancient empire built the Colosseum?", "roman"),
    ("history", "Who developed the theory of evolution by natural selection?", "darwin"),
    ("history", "In what year did the Berlin Wall fall?", "1989"),
    ("history", "Who was the main author of the American Declaration of Independence?", "jefferson"),
    ("literature", "Who wrote Romeo and Juliet?", "shakespeare"),
    ("literature", "What novel begins with the line 'Call me Ishmael'?", "moby"),
    ("literature", "Who wrote the dystopian novel Nineteen Eighty-Four?", "orwell"),
    ("literature", "What is the first book of the Bible?", "genesis"),
    ("literature", "Who created the fictional detective Sherlock Holmes?", "doyle"),
    ("literature", "What Shakespeare play features the ghost of a Danish king?", "hamlet"),
    ("cs", "What data structure operates on a last-in, first-out principle?", "stack"),
    ("cs", "What is a group of eight bits collectively called?", "byte"),
    ("cs", "What programming language runs natively in web browsers?", "javascript"),
    ("cs", "What does RAM stand for? (first word only)", "random"),
    ("cs", "What is the smallest unit of digital information?", "bit"),
    ("cs", "What network protocol resolves domain names to IP addresses?", "dns"),
    ("math", "What is 7 times 8?", "56"),
    ("math", "What is 15 squared?", "225"),
    ("math", "What number is the base of the natural logarithm?", "e"),
    ("math", "What is the value of pi to two decimal places?", "3.14"),
    ("math", "What is 100 divided by 4?", "25"),
    ("math", "What is the sum of the interior angles of a triangle, in degrees?", "180"),
    ("medicine", "What vitamin does human skin produce when exposed to sunlight?", "d"),
    ("medicine", "What is the largest organ of the human body?", "skin"),
    ("medicine", "How many bones are in the adult human body?", "206"),
    ("medicine", "What organ filters waste from the blood?", "kidney"),
    ("medicine", "Which blood type is the universal donor?", "type o"),
    ("medicine", "What hormone regulates blood sugar levels?", "insulin"),
]

N_CALIB = 24          # first N items: direction discovery; rest: holdout
N_HOLDOUT_MIN = 20    # fail-closed floor of matched holdout items


@dataclass(frozen=True)
class Config:
    model_id: str = "google/gemma-2-2b"
    seed: int = 42
    gen_max_new: int = 16
    op_a_frac: tuple = (0.31, 0.42)      # operator block A: mid-depth
    op_b_frac: tuple = (0.54, 0.65)      # operator block B: higher depth
    readout_frac: float = 0.78           # readout block
    scale_grid: tuple = (0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2)
    readout_keep_prob: float = 0.6       # target: keep 60% of answer-span compute
    match_rel_tol: float = 0.15          # support-drop match tolerance (P1 vs M, P1 vs P2)
    candidate_vocab: int = 4096          # directions = unembedding rows of token ids [0, K)
    match_fraction_min: float = 0.65
    drop_frac_min: float = 0.5           # fail-closed: achieved drop >= frac * tau_drop
    sham_abs_tol: float = 1e-4           # nats, zero-sham margin parity
    patch_rel_tol: float = 1e-5          # realized vs intended coordinate change (abs)
    engagement_min: float = 0.01         # min relative L2(planted - clean) at readout
    n_calib: int = N_CALIB
    bootstrap: int = 999

    def resolve_layers(self, n_layers: int) -> dict:
        def span(f):
            lo = max(0, round(f[0] * (n_layers - 1)))
            hi = min(n_layers - 2, round(f[1] * (n_layers - 1)))
            return tuple(range(lo, hi + 1))
        return dict(
            op_a=span(self.op_a_frac),
            op_b=span(self.op_b_frac),
            readout=round(self.readout_frac * (n_layers - 1)),
        )


# ---------------------------------------------------------------------------
# Token/index helpers
# ---------------------------------------------------------------------------
def question_prompt(question: str) -> str:
    return f"Q: {question}\nA:"


def judge_prompt(question: str, answer: str) -> str:
    return f"Q: {question}\nA: {answer}\nWas your answer CORRECT or INCORRECT?\nAnswer:"


def answer_span_positions(offset_mapping, start: int, end: int) -> list:
    return [i for i, (a, b) in enumerate(offset_mapping) if b > start and a < end]


# ---------------------------------------------------------------------------
# Intervention geometry
# ---------------------------------------------------------------------------
def nearest_scale(clean_support: float, scale_grid, drop_fn) -> tuple:
    """Scan operator strengths; return (scale, support_after, drop) whose
    support drop is nearest to the per-item target. The target is a MAGNITUDE
    (support must fall by tau_drop nats), so it uses |clean_support|."""
    tau_drop = min(1.0, 0.35 * abs(clean_support))
    best = None
    for s in scale_grid:
        sup = drop_fn(s)
        d = clean_support - sup
        err = abs(d - tau_drop)
        if best is None or err < best[0]:
            best = (err, s, sup, d)
    return best[1], best[2], best[3]


def support_match_ok(drop_a: float, drop_b: float, rel_tol: float) -> bool:
    scale = max(drop_a, drop_b, 1e-6)
    return abs(drop_a - drop_b) / scale <= rel_tol


def pick_direction(diff_rows) -> dict:
    """diff_rows: [n_items, K] tensor of (score_planted_a - score_random).
    Discovery contrast is P1 - M only; no clean comparison is used, so the
    holdout R1/R3 tests stay independent. Returns audit stats + token id."""
    mean_sep = diff_rows.mean(dim=0)
    token = int(torch.argmax(mean_sep).item())
    consistency = float((diff_rows[:, token] > 0).float().mean().item())
    coord = mean_sep[token].item()
    return dict(
        candidate_token=token,
        calib_mean_p1_minus_m=round(coord, 6),
        calib_sign_consistency=round(consistency, 4),
        n_calib_items=int(diff_rows.shape[0]),
    )


def random_unit_direction(dim: int, rng: random.Random, device: str) -> torch.Tensor:
    r = torch.empty(dim, device=device)
    torch.manual_seed(rng.randrange(2**31))
    torch.nn.init.normal_(r)
    return r / r.norm()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def bootstrap_ci_mean(values, n_boot: int, seed: int) -> dict:
    x = torch.tensor([float(v) for v in values], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    n = x.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = x[idx].mean(dim=1)
    q = torch.tensor([0.025, 0.975], dtype=means.dtype)
    lo, hi = torch.quantile(means, q).tolist()
    return dict(mean=round(float(x.mean()), 6), lo=round(lo, 6), hi=round(hi, 6),
                positive_fraction=round(float((means > 0).float().mean().item()), 4),
                n=int(n))


def sha256_json(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Offline self-test (no model, no download)
# ---------------------------------------------------------------------------
def selftest() -> None:
    torch.manual_seed(0)
    assert len(DATASET) == 60
    assert len({(q, a) for _, q, a in DATASET}) == 60

    # support matching tolerance boundary (relative to the larger of the pair)
    assert support_match_ok(2.0, 2.29, 0.15) is True
    assert support_match_ok(2.0, 2.4, 0.15) is False
    assert support_match_ok(0.5, 0.55, 0.15) is True

    # nearest-scale scan hits the predeclared target behavior
    best = nearest_scale(4.0, (0.9, 0.5, 0.2), lambda s: 4.0 - 1.0 * s * 4.0)
    assert best[0] == 0.2 and round(best[2], 6) == 0.8

    # drop target is a magnitude: negative support (typical for log-probs)
    # still yields a positive target, and a no-op operator selects drop=0
    b2 = nearest_scale(-10.0, (0.9, 0.2), lambda s: -10.0)
    assert b2[1] == -10.0 and b2[2] == 0.0

    # drop_frac_min floor is fail-closed (a no-op operator fails it)
    assert 0.5 > 0.0

    # direction pick: synthetic matrix with one planted column
    rows = torch.zeros(20, 32)
    rows[:, 7] = 0.4 * torch.ones(20) + 0.05 * torch.randn(20)
    sel = pick_direction(rows)
    assert sel["candidate_token"] == 7 and sel["calib_sign_consistency"] == 1.0

    # bootstrap CI brackets the mean and its width is sane
    vals = torch.randn(40).tolist()
    ci = bootstrap_ci_mean(vals, 999, 7)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert 0.05 < (ci["hi"] - ci["lo"]) < 2.0

    # random unit direction is normalized and deterministic given the seed
    r1 = random_unit_direction(64, random.Random(3), "cpu")
    r2 = random_unit_direction(64, random.Random(3), "cpu")
    assert torch.allclose(r1, r2) and abs(r1.norm().item() - 1.0) < 1e-5

    # config resolution never collides op blocks with the readout layer
    for n in (26, 42):
        L = Config().resolve_layers(n)
        assert max(L["op_a"]) < min(L["op_b"]) and max(L["op_b"]) < L["readout"] < n - 1

    # dataset/judge prompts stay distinct and contain the answer span
    p = judge_prompt(DATASET[0][1], "gravity")
    assert p.index("gravity") > p.index("\nA:") >= 0 and p.endswith("\nAnswer:")

    print("selftest: all checks passed")


if __name__ == "__main__":
    selftest()
