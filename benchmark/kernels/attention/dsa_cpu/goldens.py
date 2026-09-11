from pathlib import Path

import torch

FIXTURE_DIR = Path(__file__).parent / "fixtures"
POINTS = ["prefill_1k", "prefill_2k", "prefill_4k", "prefill_8k", "decode_8k"]
# Calibrated against a bf16 run of the fp32 reference with a 3x margin; see README.
# A row is "determined" when the golden selects every valid position (padding present):
# its output is then pinned exactly, so it gets the tight class. Rows whose set is a real
# top-2048 selection get the loose class, because a few bf16 boundary flips move the output.
OVERLAP_MIN = 0.97
TIGHT = dict(max_abs=0.05, row_rel_l2=0.03)
LOOSE = dict(max_abs=0.70, row_rel_l2=0.35)


def load(name):
    return torch.load(FIXTURE_DIR / f"{name}.pt", weights_only=True)


def row_overlap(pred_row, gold_row) -> float:
    g = set(gold_row[gold_row >= 0].tolist())
    p = set(pred_row[pred_row >= 0].tolist())
    return len(g & p) / max(1, len(g))


def _close(a, b, tight_rows=None):
    """tight_rows: bool [R] selecting rows checked with TIGHT; the rest use LOOSE (default all tight)."""
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} != {tuple(b.shape)}"
    if not torch.isfinite(a).all():
        return "non-finite values"
    if tight_rows is None:
        tight_rows = torch.ones(a.shape[0], dtype=torch.bool)
    max_abs = (a - b).abs().flatten(1).max(dim=1).values
    rel = (a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)
    lim_abs = torch.where(tight_rows, TIGHT["max_abs"], LOOSE["max_abs"])
    lim_rel = torch.where(tight_rows, TIGHT["row_rel_l2"], LOOSE["row_rel_l2"])
    bad = (max_abs > lim_abs) | (rel > lim_rel)
    if bad.any():
        r = int(bad.nonzero()[0])
        cls = "tight" if tight_rows[r] else "loose"
        return f"{int(bad.sum())} rows out of tolerance; first row {r} ({cls}): max|err|={max_abs[r]:.4f} (limit {lim_abs[r]:.2f}), rel-L2={rel[r]:.4f} (limit {lim_rel[r]:.2f})"
    return None


def check_dense_point(name, out) -> list[str]:
    f = load(name)
    err = _close(out[f["rows"]], f["dense_out"])
    return [f"{name}: dense output mismatch: {err}"] if err else []


def check_sparse_point(name, out, topk) -> list[str]:
    f = load(name)
    if topk is None:
        return [f"{name}: no top-k indices returned"]
    rows = f["rows"]
    if topk.shape[1] != f["topk_idx"].shape[1]:
        return [f"{name}: topk width {topk.shape[1]} != {f['topk_idx'].shape[1]}"]
    problems = []
    for r, gold in zip(rows.tolist(), f["topk_idx"]):
        pred = topk[r]
        ov = row_overlap(pred, gold)
        if ov < OVERLAP_MIN:
            problems.append(f"{name} row {r}: top-k overlap {ov:.4f} < {OVERLAP_MIN}")
        n_gold = int((gold >= 0).sum())
        n_pred = int((pred >= 0).sum())
        if n_gold < gold.numel() and n_pred != n_gold:
            problems.append(f"{name} row {r}: expected all {n_gold} valid positions selected, got {n_pred}")
    err = _close(out[rows], f["sparse_out"], tight_rows=(f["topk_idx"] < 0).any(dim=1))
    if err:
        problems.append(f"{name}: sparse output mismatch: {err}")
    return problems
