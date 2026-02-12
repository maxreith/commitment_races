import pytest
from dataclasses import dataclass, field

from src.num_example import (
    calculate_p_1,
    calculate_commitment_probabilities,
    calculate_crashing_probability,
    calculate_expected_payoff,
    normalize_game,
    to_original_payoff,
    calculate_base_prob_straight,
    calculate_base_prob_move,
    calculate_base_mix_payoff,
    calculate_base_crash_prob,
    calculate_tinf_commit_prob_first_period,
    calculate_tinf_crash_probability,
    calculate_tinf_crash_first_period_pct,
)


@dataclass
class CommitmentGameExample:
    """Expected values for a commitment game numerical example.

    All payoffs (base_mix_payoff, t1_payoff, etc.) are in normalized (b,c) form.
    To add a new example, fill in all fields and append to EXAMPLES.
    """

    id: str
    # Raw payoff matrix
    A: float
    B: float
    C: float
    D: float
    # Normalized parameters
    b: float
    c: float
    # Base game MSNE
    base_prob_straight: float  # (b-1)/(b+c-1)
    base_prob_move: float  # c/(b+c-1)
    base_mix_payoff: float  # bc/(b+c-1)
    base_crash_prob: float  # ((b-1)/(b+c-1))^2
    # T=1 commitment game
    t1_p1: float  # first-period commit probability
    t1_payoff: float  # MSNE expected payoff
    t1_crash_prob: float
    # T=2 commitment game
    t2_probs: list  # [p_1, p_2, p_W]
    t2_payoff: float
    t2_crash_prob: float
    # T -> inf limits
    tinf_p1: float  # (b-c)/b
    tinf_payoff: float  # c
    tinf_crash_prob: float  # p1^2 / (1-(c/b)^2)
    tinf_crash_first_pct: float  # ((b-c)/b)^2 * 100


EXAMPLES = [
    CommitmentGameExample(
        id="chicken_b2_c0.5",
        A=-1, B=1, C=-0.5, D=0,
        b=2, c=0.5,
        base_prob_straight=2 / 3,
        base_prob_move=1 / 3,
        base_mix_payoff=2 / 3,
        base_crash_prob=4 / 9,
        t1_p1=8 / 11,
        t1_payoff=6 / 11,
        t1_crash_prob=(8 / 11) ** 2 + (3 / 11) ** 2 * (2 / 3) ** 2,
        t2_probs=[32 / 43, 8 / 43, 3 / 43],
        t2_payoff=22 / 43,
        t2_crash_prob=(32 / 43) ** 2 + (8 / 43) ** 2 + (3 / 43) ** 2 * (2 / 3) ** 2,
        tinf_p1=3 / 4,
        tinf_payoff=0.5,
        tinf_crash_prob=0.6,
        tinf_crash_first_pct=56.25,
    ),
    # To add a new example, copy the block above and fill in all fields.
]


def _example_id(ex):
    return ex.id


# --- Normalization ---


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_normalize_game(ex):
    b, c = normalize_game(ex.A, ex.B, ex.C, ex.D)
    assert b == pytest.approx(ex.b)
    assert c == pytest.approx(ex.c)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_to_original_payoff_roundtrip(ex):
    assert to_original_payoff(ex.b, ex.A, ex.D) == pytest.approx(ex.B)
    assert to_original_payoff(ex.c, ex.A, ex.D) == pytest.approx(ex.C)
    assert to_original_payoff(0, ex.A, ex.D) == pytest.approx(ex.A)
    assert to_original_payoff(1, ex.A, ex.D) == pytest.approx(ex.D)


# --- Base game ---


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_base_prob_straight(ex):
    assert calculate_base_prob_straight(ex.b, ex.c) == pytest.approx(ex.base_prob_straight)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_base_prob_move(ex):
    assert calculate_base_prob_move(ex.b, ex.c) == pytest.approx(ex.base_prob_move)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_base_mix_payoff(ex):
    assert calculate_base_mix_payoff(ex.b, ex.c) == pytest.approx(ex.base_mix_payoff)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_base_crash_prob(ex):
    assert calculate_base_crash_prob(ex.b, ex.c) == pytest.approx(ex.base_crash_prob)


# --- T=1 commitment game ---


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_p1_t1(ex):
    assert calculate_p_1(1, ex.b, ex.c) == pytest.approx(ex.t1_p1)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_payoff_t1(ex):
    assert calculate_expected_payoff(1, ex.b, ex.c) == pytest.approx(ex.t1_payoff)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_crash_prob_t1(ex):
    assert calculate_crashing_probability(1, ex.b, ex.c) == pytest.approx(ex.t1_crash_prob)


# --- T=2 commitment game ---


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_commitment_probs_t2(ex):
    probs = calculate_commitment_probabilities(2, ex.b, ex.c)
    for computed, expected in zip(probs, ex.t2_probs):
        assert computed == pytest.approx(expected)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_payoff_t2(ex):
    assert calculate_expected_payoff(2, ex.b, ex.c) == pytest.approx(ex.t2_payoff)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_crash_prob_t2(ex):
    assert calculate_crashing_probability(2, ex.b, ex.c) == pytest.approx(ex.t2_crash_prob)


# --- T -> inf limits ---


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_tinf_commit_prob_first_period(ex):
    assert calculate_tinf_commit_prob_first_period(ex.b, ex.c) == pytest.approx(ex.tinf_p1)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_p1_convergence(ex):
    assert calculate_p_1(20, ex.b, ex.c) == pytest.approx(ex.tinf_p1)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_wait_prob_convergence(ex):
    probs = calculate_commitment_probabilities(20, ex.b, ex.c)
    assert probs[-1] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_payoff_convergence(ex):
    assert calculate_expected_payoff(20, ex.b, ex.c) == pytest.approx(ex.tinf_payoff)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_tinf_crash_prob(ex):
    assert calculate_tinf_crash_probability(ex.b, ex.c) == pytest.approx(ex.tinf_crash_prob)


@pytest.mark.parametrize("ex", EXAMPLES, ids=_example_id)
def test_tinf_crash_first_pct(ex):
    assert calculate_tinf_crash_first_period_pct(ex.b, ex.c) == pytest.approx(ex.tinf_crash_first_pct)
