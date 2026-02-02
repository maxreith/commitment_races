from pathlib import Path


def calculate_p_1(T, b, c):
    """Calculates the probability of commitment in the first period.

    Args:
        T: Total number of commitment periods.
        b: Parameter b in the Game of Chicken.
        c: Parameter c in the Game of Chicken.

    Returns:
        First-period commitment probability.
    """
    rho = c / b
    denominator = (1 - rho**T) / (1 - rho) + (rho**T) * (b + c - 1) / (b - 1)
    return 1 / denominator


def calculate_commitment_probabilities(T, b, c):
    """Calculates commitment probabilities for each period and the wait probability.

    Args:
        T: Total number of commitment periods.
        b: Parameter b in the Game of Chicken.
        c: Parameter c in the Game of Chicken.

    Returns:
        List of probabilities [p_1, ..., p_T, p_W].
    """
    probabilities = []
    probabilities.append(calculate_p_1(T, b, c))
    for t in range(2, T + 1):
        p_t = calculate_p_1(T, b, c) * (c / b) ** (t - 1)
        probabilities.append(p_t)
    p_W = 1 - sum(probabilities)
    probabilities.append(p_W)
    return probabilities


def calculate_crashing_probability(T, b, c):
    """Calculates the overall crashing probability given T commitment periods.

    Args:
        T: Total number of commitment periods.
        b: Parameter b in the Game of Chicken.
        c: Parameter c in the Game of Chicken.

    Returns:
        Total crash probability.
    """
    probabilities = calculate_commitment_probabilities(T, b, c)
    p_crash_in_commitment_stage = sum([p**2 for p in probabilities[:-1]])
    p_reaching_base_game = probabilities[-1] ** 2
    p_crash_in_base_game = p_reaching_base_game * ((b - 1) / (b + c - 1)) ** 2
    return p_crash_in_commitment_stage + p_crash_in_base_game


def calculate_expected_payoff(T, b, c):
    """Calculates the MSNE expected payoff via U(W).

    Args:
        T: Total number of commitment periods.
        b: Parameter b in the Game of Chicken.
        c: Parameter c in the Game of Chicken.

    Returns:
        Expected payoff in normalized form.
    """
    probabilities = calculate_commitment_probabilities(T, b, c)
    commitment_probabilities = probabilities[:-1]
    wait_probability = probabilities[-1]
    expected_payoff = sum([p * c for p in commitment_probabilities]) + wait_probability * (b * c) / (b + c - 1)
    return expected_payoff


def normalize_game(A, B, C, D):
    """Converts raw payoff parameters to normalized b, c.

    Args:
        A: Payoff when both play A_1.
        B: Payoff to A_1 player when opponent plays A_2.
        C: Payoff to A_2 player when opponent plays A_1.
        D: Payoff when both play A_2.

    Returns:
        Tuple (b, c) in normalized form.
    """
    b = (B - A) / (D - A)
    c = (C - A) / (D - A)
    return b, c


def to_original_payoff(normalized_value, A, D):
    """Converts a normalized payoff back to original scale.

    Args:
        normalized_value: Payoff in normalized [0,1] scale.
        A: Original A parameter.
        D: Original D parameter.

    Returns:
        Payoff in original scale: normalized_value * (D - A) + A.
    """
    return normalized_value * (D - A) + A


def calculate_base_prob_straight(b, c):
    """Base game MSNE probability of A_1.

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        Probability (b-1)/(b+c-1).
    """
    return (b - 1) / (b + c - 1)


def calculate_base_prob_move(b, c):
    """Base game MSNE probability of A_2.

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        Probability c/(b+c-1).
    """
    return c / (b + c - 1)


def calculate_base_mix_payoff(b, c):
    """Base game MSNE expected payoff (normalized).

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        Payoff bc/(b+c-1).
    """
    return (b * c) / (b + c - 1)


def calculate_base_crash_prob(b, c):
    """Base game crash probability.

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        Probability ((b-1)/(b+c-1))^2.
    """
    return ((b - 1) / (b + c - 1)) ** 2


def calculate_tinf_commit_prob_first_period(b, c):
    """T->inf first period commitment probability.

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        Probability (b-c)/b.
    """
    return (b - c) / b


def calculate_tinf_crash_probability(b, c):
    """T->inf total crash probability: p1^2 / (1-(c/b)^2).

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        Total crash probability in the T->inf limit.
    """
    p1 = calculate_tinf_commit_prob_first_period(b, c)
    return p1**2 / (1 - (c / b) ** 2)


def calculate_tinf_crash_first_period_pct(b, c):
    """T->inf crash in first period as percentage of total crashes.

    The fraction of all crashes that occur in the first period is
    p1^2 / total_crash = (1 - (c/b)^2) * ((b-c)/b)^2 / (((b-c)/b)^2)
    which simplifies to ((b-c)/b)^2 * 100 / total_crash...

    Actually, the paper text says "this outcome occurs with probability
    p1^2 ~ X%", so this is simply p1^2 * 100.

    Args:
        b: Normalized b parameter.
        c: Normalized c parameter.

    Returns:
        ((b-c)/b)^2 * 100.
    """
    p1 = calculate_tinf_commit_prob_first_period(b, c)
    return p1**2 * 100


def generate_numerical_example(A, B, C, D, output_path):
    """Computes all values and writes \\newcommand definitions to a .tex file.

    Args:
        A: Payoff when both play A_1.
        B: Payoff to A_1 player when opponent plays A_2.
        C: Payoff to A_2 player when opponent plays A_1.
        D: Payoff when both play A_2.
        output_path: Path to write the generated .tex file.
    """
    b, c = normalize_game(A, B, C, D)

    def fmt(value, decimals=2):
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    t2_probs = calculate_commitment_probabilities(2, b, c)

    macros = {
        "capA": f"{A:g}",
        "capB": f"{B:g}",
        "capC": f"{C:g}",
        "capD": f"{D:g}",
        "baseProbStraight": fmt(calculate_base_prob_straight(b, c)),
        "baseProbMove": fmt(calculate_base_prob_move(b, c)),
        "baseMixPayoff": fmt(to_original_payoff(calculate_base_mix_payoff(b, c), A, D)),
        "ToneProbCommitStraight": fmt(calculate_p_1(1, b, c)),
        "ToneWait": fmt(1 - calculate_p_1(1, b, c)),
        "TonePayoff": fmt(to_original_payoff(calculate_expected_payoff(1, b, c), A, D)),
        "TtwoProbCommitStraightFirstPeriod": fmt(t2_probs[0]),
        "TtwoProbCommitStraightSecondPeriod": fmt(t2_probs[1]),
        "TtwoWait": fmt(t2_probs[2]),
        "TtwoPayoff": fmt(to_original_payoff(calculate_expected_payoff(2, b, c), A, D)),
        "TinfCommitProbStraightFirstPeriod": fmt(calculate_tinf_commit_prob_first_period(b, c)),
        "TinfPayoff": fmt(to_original_payoff(c, A, D)),
        "TinfCrashInFirstPeriodPercentage": fmt(calculate_tinf_crash_first_period_pct(b, c), 0),
        "TinfCrashProbability": fmt(calculate_tinf_crash_probability(b, c)),
        "baseProbCrash": fmt(calculate_base_crash_prob(b, c)),
        "ToneProbCrash": fmt(calculate_crashing_probability(1, b, c)),
        "TtwoProbCrash": fmt(calculate_crashing_probability(2, b, c)),
    }

    lines = []
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n")


A_DEFAULT = -10
B_DEFAULT = 1
C_DEFAULT = -1
D_DEFAULT = 0

if __name__ == "__main__":
    generate_numerical_example(
        A_DEFAULT, B_DEFAULT, C_DEFAULT, D_DEFAULT,
        "bld/numerical_example.tex",
    )
