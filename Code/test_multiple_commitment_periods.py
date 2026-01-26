import pdb
from Code.commitment_periods import calculate_p_1, calculate_commitment_probabilities, calculate_crashing_probability, calculate_expected_payoff

def test_calculate_p_1():
    T = 1
    b = 2
    c = 0.5
    expected_p_1 = 8/11
    calculated_p_1 = calculate_p_1(T, b, c)
    assert abs(calculated_p_1 - expected_p_1) < 1e-6, f"Expected {expected_p_1}, but got {calculated_p_1}"

def test_calculate_p_1_multiple_periods():
    T = 2
    b = 2
    c = 0.5
    expected_p_1 = 32/43
    calculated_p_1 = calculate_p_1(T, b, c)
    assert abs(calculated_p_1 - expected_p_1) < 1e-6, f"Expected {expected_p_1}, but got {calculated_p_1}"

def test_calculate_p_1_T_to_infinity():
    T = 20
    b = 2
    c = 0.5
    expected_p_1 = 3/4
    calculated_p_1 = calculate_p_1(T, b, c)
    assert abs(calculated_p_1 - expected_p_1) < 1e-6, f"Expected {expected_p_1}, but got {calculated_p_1}"

def test_calculate_commitment_probabilities():
    T = 2
    b = 2
    c = 0.5
    expected_probabilities = [32/43, 8/43, 3/43]
    calculated_probabilities = calculate_commitment_probabilities(T, b, c)
    for calculated, expected in zip(calculated_probabilities, expected_probabilities):
        assert abs(calculated - expected) < 1e-6, f"Expected {expected}, but got {calculated}"

def test_calculate_commitment_probabilities_T_to_infinity():
    T = 20
    b = 2
    c = 0.5
    expected_wait_probability = 0
    calculated_probabilities = calculate_commitment_probabilities(T, b, c)
    calculated_wait_probability = calculated_probabilities[-1]
    assert abs(calculated_wait_probability - expected_wait_probability) < 1e-6, f"Expected {expected_wait_probability}, but got {calculated_wait_probability}"

def test_calculate_crashing_probability_one_period():
    T = 1
    b = 2
    c = 0.5
    expected_crashing_probability = (8/11)**2 + (3/11)**2 * (2/3)**2
    calculated_crashing_probability = calculate_crashing_probability(T, b, c)
    assert abs(calculated_crashing_probability - expected_crashing_probability) < 1e-6, f"Expected {expected_crashing_probability}, but got {calculated_crashing_probability}"

def test_calculate_crashing_probability_multiple_periods():
    T = 2
    b = 2
    c = 0.5
    expected_crashing_probability = (32/43)**2 + (8/43)**2 + (3/43)**2 * (2/3)**2
    calculated_crashing_probability = calculate_crashing_probability(T, b, c)
    assert abs(calculated_crashing_probability - expected_crashing_probability) < 1e-6, f"Expected {expected_crashing_probability}, but got {calculated_crashing_probability}"

def test_calculate_expected_payoff_single_period():
    T = 1
    b = 2
    c = 0.5
    expected_payoff = -5/11 + 1
    calculated_payoff = calculate_expected_payoff(T, b, c)
    assert abs(calculated_payoff - expected_payoff) < 1e-6, f"Expected {expected_payoff}, but got {calculated_payoff}"

def test_calculate_expected_payoff_multiple_periods():
    T = 2
    b = 2
    c = 0.5
    expected_payoff = -21/43 + 1
    calculated_payoff = calculate_expected_payoff(T, b, c)
    assert abs(calculated_payoff - expected_payoff) < 1e-6, f"Expected {expected_payoff}, but got {calculated_payoff}"

def test_calculate_expected_payoff_T_to_infinity():
    T = 20
    b = 2
    c = 0.5
    expected_payoff = -0.5 + 1
    calculated_payoff = calculate_expected_payoff(T, b, c)
    assert abs(calculated_payoff - expected_payoff) < 1e-6, f"Expected {expected_payoff}, but got {calculated_payoff}"