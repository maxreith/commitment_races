"""
commitment_game.py

Build the pure–strategy payoff table of an extended “simultaneous commitment”
game, starting from any normal-form base game (given either as a Python dict
or as a LaTeX tabular).  The output is printed

  • as a readable ASCII matrix; and
  • as a LaTeX tabular,

with every strategy labelled                       A_{it}
meaning “commit to base-game action A_i in period t”,
and ‘W’ meaning “wait”.

Author:  <your-name>
"""

import re
from typing import Dict, Tuple, Union, Optional
import sympy                    # used for parsing symbolic pay-offs
try:
    import nashpy as nash        # optional (only needed if you omit wait_payoff)
except ImportError:
    nash = None                  # handled below


# ────────────────────────────────────────────────────────────────────────────────
# 1.  Parse a LaTeX tabular into a {(row, col): (u_row, u_col)} dictionary
# ────────────────────────────────────────────────────────────────────────────────
def parse_latex_tabular(latex: str) -> Dict[Tuple[str, str], Tuple[Union[float, sympy.Expr],
                                                                     Union[float, sympy.Expr]]]:
    """Return a payoff-dict from a LaTeX tabular."""
    m = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\hline(.*?)\\end\{tabular\}", latex, re.S)
    if not m:
        raise ValueError("Could not find a tabular environment with \\hline.")
    header_block = m.group(1).strip()
    body = m.group(2).strip()
    # Extract column labels
    header_lines = [ln.strip() for ln in header_block.splitlines() if ln.strip()]
    header = header_lines[0]
    col_labels = [re.sub(r"[\$\s]", "", c) for c in header.split('&')[1:]]
    # Clean up any trailing "\\" in the column labels
    col_labels = [re.sub(r"[\$\s]", "", c) for c in header.split('&')[1:]]
    col_labels = [re.sub(r"\\\\$", "", lbl) for lbl in col_labels]
    # Split body into non-empty lines
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    payoff: Dict[Tuple[str, str], Tuple[Union[float, sympy.Expr],
                                          Union[float, sympy.Expr]]] = {}
    for row_line in lines:
        parts = [p.strip() for p in row_line.rstrip(r"\\").split('&')]
        if len(parts) != len(col_labels) + 1:
            raise ValueError("Mismatch between header and row: " + row_line)
        row_label = re.sub(r"[\$\s]", "", parts[0])
        for col_label, cell in zip(col_labels, parts[1:]):
            txt = re.sub(r"[\$\s]", "", cell)
            txt = re.sub(r"\^\\?\{([^}]+)\}", r"**(\1)", txt)
            left, right = txt.split(',', 1)
            def _cast(x: str):
                try:
                    return float(x)
                except ValueError:
                    return sympy.sympify(x)
            payoff[(row_label, col_label)] = (_cast(left), _cast(right))
    return payoff


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Helper: compute one Nash equilibrium payoff of the base game (optional)
# ────────────────────────────────────────────────────────────────────────────────
def compute_nash_payoff(
    base_game: Dict[Tuple[str, str], Tuple[float, float]]
) -> Tuple[float, float]:
    """Return (u_row, u_col) of one mixed NE, using nashpy.support_enumeration."""
    if nash is None:
        raise ImportError("nashpy is not installed; pass wait_payoff instead.")

    rows = sorted({r for r, _ in base_game})
    cols = sorted({c for _, c in base_game})
    R    = [[base_game[(r, c)][0] for c in cols] for r in rows]
    C    = [[base_game[(r, c)][1] for c in cols] for r in rows]

    game = nash.Game(R, C)
    for σ_r, σ_c in game.support_enumeration():          # take the first NE
        u_r = float(σ_r @ sympy.Matrix(R) @ sympy.Matrix(σ_c))
        u_c = float(σ_r @ sympy.Matrix(C) @ sympy.Matrix(σ_c))
        return u_r, u_c
    raise RuntimeError("No Nash equilibrium found.")


# ────────────────────────────────────────────────────────────────────────────────
# 3.  Helper: best-response to a committed action
# ────────────────────────────────────────────────────────────────────────────────
def best_response_to(committed: str,
                     payoff_dict: Dict[Tuple[str, str], Tuple[float, float]],
                     player: str = "row") -> str:
    """Return the action of <player> that maximises her payoff vs *committed*."""
    rows = sorted({r for r, _ in payoff_dict})
    cols = sorted({c for _, c in payoff_dict})

    best_act, best_pay = None, float("-inf")
    if player == "row":
        for r in rows:
            pay = payoff_dict[(r, committed)][0]
            if pay > best_pay:
                best_pay, best_act = pay, r
    else:  # player == "col"
        for c in cols:
            pay = payoff_dict[(committed, c)][1]
            if pay > best_pay:
                best_pay, best_act = pay, c
    return best_act      # type: ignore – always set


# ────────────────────────────────────────────────────────────────────────────────
# 4.  Pretty label:   ("commit", "A_2", 3) →  A_{23}        ("wait",) →  W
# ────────────────────────────────────────────────────────────────────────────────
def strat_label(s: Tuple) -> str:
    """Convert an extended pure strategy tuple into A_{it} or W."""
    if s[0] == "wait":
        return "W"
    action = s[1]                           # e.g.  'A_2'
    t      = s[2]                           # period
    # take the part *after* the underscore so we ignore any digits in 'A'
    i      = action.split('_', 1)[1] if '_' in action else re.sub(r'\D', '', action)
    return f"A_{{{i}{t}}}"


# ────────────────────────────────────────────────────────────────────────────────
# 5.  Main routine
# ────────────────────────────────────────────────────────────────────────────────
def generate_commitment_payoffs(
    base_game  : Union[Dict[Tuple[str, str], Tuple[float, float]], str],
    T          : int,
    wait_payoff: Optional[Tuple[float, float]] = None
) -> Dict[Tuple[Tuple, Tuple], Tuple[float, float]]:
    """
    Build the payoff table of the T-period commitment game and print it.

    Parameters
    ----------
    base_game   : dict OR LaTeX tabular (see module docstring)
    T           : number of commitment periods (≥1)
    wait_payoff : (u_row, u_col) if you want to *set* Wait–Wait payoff manually.
                  If None, the base-game mixed NE payoff is used (needs nashpy).
    """
    # 5-a  Parse / normalise the base game
    base_dict = parse_latex_tabular(base_game) if isinstance(base_game, str) else base_game
    row_actions = sorted({r for r, _ in base_dict})
    col_actions = sorted({c for _, c in base_dict})

    # 5-b  Enumerate extended pure strategies
    row_strats = [("commit", a, t) for t in range(1, T+1) for a in row_actions] + [("wait",)]
    col_strats = [("commit", b, t) for t in range(1, T+1) for b in col_actions] + [("wait",)]
    # 5-c  Payoff if both players wait
    if wait_payoff is not None:
        no_r, no_c = wait_payoff
    else:
        no_r, no_c = compute_nash_payoff(base_dict)

    # 5-d  Build extended payoff dict
    payoffs: Dict[Tuple[Tuple, Tuple], Tuple[float, float]] = {}
    for r in row_strats:
        for c in col_strats:
            t_r = r[2] if r[0] == "commit" else None
            t_c = c[2] if c[0] == "commit" else None

            if t_r is None and t_c is None:         # Wait–Wait
                pay = (no_r, no_c)
            else:                                   # someone commits
                first = min(t for t in (t_r or float('inf'),
                                        t_c or float('inf')))
                if t_r == t_c:                      # simultaneous commit
                    ra, ca = r[1], c[1]
                elif t_r == first:                  # row commits first
                    ra = r[1]
                    ca = best_response_to(ra, base_dict, player="col")
                else:                               # column commits first
                    ca = c[1]
                    ra = best_response_to(ca, base_dict, player="row")

                pay = base_dict[(ra, ca)]
            payoffs[(r, c)] = pay

    # 5-e  ──  ♣  ASCII matrix  ♣  ───────────────────────────────────────────
    row_lbls = [strat_label(s) for s in row_strats]
    col_lbls = [strat_label(s) for s in col_strats]
    rlw      = max(len(lbl) for lbl in row_lbls)
    clw      = max(len(lbl) for lbl in col_lbls + ["(r,c)"])
    print("\nASCII payoff matrix:")
    print(" "*(rlw+2) + "  ".join(f"{lbl:>{clw}}" for lbl in col_lbls))
    for s_row, lbl in zip(row_strats, row_lbls):
        line = f"{lbl:>{rlw}} :"
        for s_col in col_strats:
            u = payoffs[(s_row, s_col)]
            line += f"  ({u[0]},{u[1]})".rjust(clw+3)
        print(line)

    # 5-f  ──  ♣  LaTeX tabular  ♣  ─────────────────────────────────────────
    print("\nLaTeX tabular (copy-paste into your document):")
    fmt = "l|" + "c" * len(col_lbls)
    # Wrap column labels in $
    col_hdr = [f"${lbl}$" for lbl in col_lbls]
    tex = [
        "\\begin{tabular}{" + fmt + "}",
        " & " + " & ".join(col_hdr) + r" \\ \hline"
    ]
    for s_row, lbl in zip(row_strats, row_lbls):
        # Wrap row label in $
        row_label = f"${lbl}$"
        row_cells = [
            f"${u[0]},{u[1]}$"
            for u in (payoffs[(s_row, s_col)] for s_col in col_strats)
        ]
        tex.append(" & ".join([row_label] + row_cells) + r" \\")
    tex.append("\\end{tabular}")
    print("\n".join(tex))

    return payoffs


# ────────────────────────────────────────────────────────────────────────────────
# 6.  Quick example
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    latex_game = r"""
    \begin{tabular}{c|ccc}
    & $A_1$ & $A_2$ & $A_3$ \\
    \hline
    $A_1$ & $1,1$ & $3,0$ & $0.1-10^{2}, 0$ \\
    $A_2$ & $0, 0.1-10^{2}$ & $2, 1-10^{2}$ & $0-10^{2}, 0-10^{2}$ \\
    $A_3$ & $0,3$ & $0,0$ & $1-10^{2}, 2$ \\
    \end{tabular}
    """

    generate_commitment_payoffs(
        base_game   = latex_game,
        T           = 2,
        wait_payoff = (1, 1)
    )