import math

import torch

from switchspec.distill import weighted_distill_loss, switch_weights
from switchspec.lid import Sentence
from switchspec.metrics import (cluster_bootstrap_ci, expected_speedup, expected_tokens_per_call,
                                kl, summarise_by_category, tv)
from switchspec.ngram import NgramDrafter
from switchspec.specdec import SamplingConfig
from switchspec.transliterate import cer, make_twin, round_trip_cer


# ---- metrics -------------------------------------------------------------- #
def test_expected_speedup():
    assert math.isclose(expected_speedup(0.0, 4, 0.1), 1 / 1.4)      # slower than baseline
    assert math.isclose(expected_speedup(1.0, 4, 0.0), 5.0)
    a, g, c = 0.8, 3, 0.05
    assert math.isclose(expected_speedup(a, g, c), expected_tokens_per_call(a, g) / (g * c + 1))


def test_tv_kl():
    p = torch.tensor([0.5, 0.5, 0.0]); q = torch.tensor([0.25, 0.25, 0.5])
    assert math.isclose(float(tv(p, q)), 0.5)
    assert float(kl(p, p)) == 0 and float(kl(p, q)) > 0


def test_bootstrap_and_summary():
    vals = [1.0] * 50 + [0.0] * 50
    groups = list(range(100))
    m, lo, hi = cluster_bootstrap_ci(vals, groups)
    assert m == 0.5 and lo < 0.5 < hi
    rows = [{"category": "switch", "v": 0.2, "sent_id": i} for i in range(10)] + \
           [{"category": "inside", "v": 0.9, "sent_id": i} for i in range(10)]
    s = summarise_by_category(rows, "v")
    assert math.isclose(s["switch"]["mean"], 0.2) and s["inside"]["n"] == 10


# ---- distillation loss ---------------------------------------------------- #
def test_distill_loss():
    torch.manual_seed(0)
    B, T, V = 2, 5, 11
    t = torch.randn(B, T, V)
    sw = torch.zeros(B, T); sw[0, 2] = 1
    win = torch.zeros(B, T); win[0, 3] = 1
    valid = torch.ones(B, T)
    zero = weighted_distill_loss(t.clone(), t, sw, win, valid, 2.0, 1.0, 0.5)["loss"]
    assert abs(float(zero)) < 1e-6
    s = torch.randn(B, T, V, requires_grad=True)
    out = weighted_distill_loss(s, t, sw, win, valid, 2.0, 1.0, 0.5)
    out["loss"].backward()
    assert s.grad is not None and "tv_switch" in out
    # upweighting switches should increase the gradient at the switch position
    s2 = s.detach().clone().requires_grad_(True)
    weighted_distill_loss(s2, t, sw, win, valid, 0.0, 0.0, 0.0)["loss"].backward()
    ratio_w = s.grad[0, 2].norm() / s.grad[1, 2].norm()
    ratio_u = s2.grad[0, 2].norm() / s2.grad[1, 2].norm()
    assert ratio_w > ratio_u
    assert switch_weights(sw, win, 2.0, 1.0)[0, 2] == 3.0


# ---- ngram ---------------------------------------------------------------- #
def test_ngram_backoff(tmp_path):
    d = NgramDrafter(3, 10).fit([[1, 2, 3, 1, 2, 4]])
    cfg = SamplingConfig(1.0)
    q = d.dist([1, 2], cfg)                      # trigram context seen: 3 or 4
    assert math.isclose(float(q[3]), 0.5) and math.isclose(float(q[4]), 0.5)
    q = d.dist([9, 3], cfg)                      # backs off to bigram ctx (3,) -> 1
    assert float(q[1]) == 1.0
    q = d.dist([9], cfg)                         # backs off to unigram
    assert math.isclose(float(q.sum()), 1.0) and float(q[9]) == 0
    p = tmp_path / "ng.pkl"; d.save(str(p))
    assert torch.equal(NgramDrafter.load(str(p)).dist([1, 2], cfg), d.dist([1, 2], cfg))


# ---- transliteration ------------------------------------------------------ #
def test_twin_and_cer():
    fake = {"main": "मैं", "kal": "कल"}
    s = Sentence(["main", "kal", "office", "!"], ["lang2", "lang2", "lang1", "other"])
    twin = make_twin(s, lambda w: fake.get(w, w.upper()))
    assert twin.words == ["मैं", "कल", "office", "!"] and twin.tags == s.tags
    full = make_twin(s, lambda w: fake.get(w, w.upper()), fully_native=True)
    assert full.words[2] == "OFFICE"
    assert cer("abcd", "abcd") == 0 and cer("abcd", "abed") == 0.25
    inv = {v: k for k, v in fake.items()}
    assert round_trip_cer(["main", "kal"], fake.get, inv.get) == 0
