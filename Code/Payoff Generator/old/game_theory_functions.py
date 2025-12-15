import re
from typing import Dict, Tuple, Optional, List
import sympy
import numpy as np

def parse_latex_tabular(latex: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    
    # find tabular environment with \hline
    m = re.search(r"\\begin\{tabular\}.*?\n(.*?)\\hline(.*?)\\end\{tabular\}", latex, re.S)
    if not m:
        raise ValueError("Could not find a tabular environment with \\hline.")
    
    header_block = m.group(1).strip() # all code before \hline
    body = m.group(2).strip() # all code after \hline

    # Extract column labels
    header_lines = [ln.strip() for ln in header_block.splitlines() if ln.strip()]
    header = header_lines[0]
    col_labels = [re.sub(r"[\$\s]", "", c) for c in header.split('&')[1:]]

    # Clean up any trailing "\\" in the column labels
    col_labels = [re.sub(r"[\$\s]", "", c) for c in header.split('&')[1:]]
    col_labels = [re.sub(r"\\\\$", "", lbl) for lbl in col_labels]

    # Split body into non-empty lines
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    payoff: Dict[Tuple[str, str], Tuple[float, float]] = {}
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
                    # Try to convert directly to a float and then format it to remove extra zeros.
                    return float(x)
                except ValueError:
                    # Convert sympy expression to float.
                    return float(sympy.sympify(x))
            payoff[(row_label, col_label)] = (_cast(left), _cast(right))
    return payoff

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

def strat_label(s: Tuple) -> str:
    """Convert an extended pure strategy tuple into a label with time‐period subscript."""
    if s[0] == "wait":
        return "W"
    action = s[1]  # e.g. 'D' or 'D,da'
    t      = s[2]  # period
    # keep the full action label and append "_<period>"
    # so ('commit','D',1)    → "D_1"
    #    ('commit','D,da',1) → "D,da_1"
    return f"{action}_{t}"

def generate_commitment_payoffs(
    base_game  : Dict[Tuple[str, str], Tuple[float, float]],
    T          : int,
    wait_payoff: Optional[Tuple[float, float]],
    delete_dominated: bool = False,
    return_latex: bool = True
) -> Dict[Tuple[Tuple, Tuple], Tuple[float, float]]:
    
    # 3-a  Parse / normalise the base game
    base_dict = parse_latex_tabular(base_game)
    row_actions = sorted({r for r, _ in base_dict})
    col_actions = sorted({c for _, c in base_dict})

    # 3-b  Enumerate extended pure strategies
    row_strats = [("commit", a, t) for t in range(1, T+1) for a in row_actions] + [("wait",)]
    col_strats = [("commit", b, t) for t in range(1, T+1) for b in col_actions] + [("wait",)]

    # 3-c  Payoff if both players wait
    no_r, no_c = wait_payoff

    # 3-d  Build extended payoff dict
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

    # 3-e Remove (weakly) dominated strategies
    if delete_dominated:
        # Remove dominated row strategies.
        # A row strategy r1 is weakly dominated by r2 if for all column strategies
        # the payoff for r2 is at least that for r1 and strictly higher in at least one.
        non_dominated_rows = set(r for r in row_strats)
        for r1 in row_strats:
            for r2 in row_strats:
                if r1 == r2:
                    continue
                # Check r2 vs r1 over all column strategies.
                if (all(payoffs[(r2, c)][0] >= payoffs[(r1, c)][0] for c in col_strats)
                    and any(payoffs[(r2, c)][0] > payoffs[(r1, c)][0] for c in col_strats)):
                    if r1 in non_dominated_rows:
                        non_dominated_rows.remove(r1)

        # Remove dominated column strategies in an analogous manner.
        non_dominated_cols = set(c for c in col_strats)
        for c1 in col_strats:
            for c2 in col_strats:
                if c1 == c2:
                    continue
                if (all(payoffs[(r, c2)][1] >= payoffs[(r, c1)][1] for r in row_strats)
                    and any(payoffs[(r, c2)][1] > payoffs[(r, c1)][1] for r in row_strats)):
                    if c1 in non_dominated_cols:
                        non_dominated_cols.remove(c1)
                        
        # Rebuild the extended payoff dictionary using only non-dominated strategies.
        new_payoffs = {}
        for r in non_dominated_rows:
            for c in non_dominated_cols:
                new_payoffs[(r, c)] = payoffs[(r, c)]
        payoffs = new_payoffs

    # 3-f Return Latex-style payoff dictionary
    if return_latex:
        # pick the strategies to display (after possible deletion)
        if delete_dominated:
            # non_dominated_rows and non_dominated_cols were built above
            disp_row_strats = sorted(non_dominated_rows,
                                     key=lambda s: (s[0], s[1:] if len(s) > 1 else ()))
            disp_col_strats = sorted(non_dominated_cols,
                                     key=lambda s: (s[0], s[1:] if len(s) > 1 else ()))
        else:
            disp_row_strats = row_strats
            disp_col_strats = col_strats

        # build labels
        row_lbls = [strat_label(s) for s in disp_row_strats]
        col_lbls = [strat_label(s) for s in disp_col_strats]
        rlw = max(len(lbl) for lbl in row_lbls)
        clw = max(len(lbl) for lbl in col_lbls + ["(r,c)"])

        # LaTeX tabular
        print("\nLaTeX tabular (copy-paste into your document):")
        fmt = "l|" + "c" * len(col_lbls)
        col_hdr = [f"${lbl}$" for lbl in col_lbls]
        tex = [
            "\\begin{tabular}{" + fmt + "}",
            " & " + " & ".join(col_hdr) + r" \\ \hline"
        ]
        for s_row, lbl in zip(disp_row_strats, row_lbls):
            row_label = f"${lbl}$"
            row_cells = [
                f"${payoffs[(s_row, s_col)][0]},{payoffs[(s_row, s_col)][1]}$"
                for s_col in disp_col_strats
            ]
            tex.append(" & ".join([row_label] + row_cells) + r" \\")
        tex.append("\\end{tabular}")
        print("\n".join(tex))

    else:
        return payoffs

def _strategy_key(s: Tuple) -> Tuple:
    """Sort key for extended pure strategies.
    
    - For commit strategies, sorts by period then by the numeric part of the action.
    - The 'wait' strategy is always sorted last.
    """
    if s[0] == "wait":
        return (float('inf'), 0)
    # For commit, s = ('commit', 'A_2', 1)
    period = s[2]
    m = re.search(r'(\d+)', s[1])
    action_num = int(m.group(1)) if m else 0
    return (period, action_num)

def game_to_matrices(game_dict: Dict[Tuple[Tuple, Tuple], Tuple[float, float]]
                    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple], List[Tuple]]:
    
    # Extract row strategies and column strategies.
    row_strats = sorted({pair[0] for pair in game_dict.keys()}, key=_strategy_key)
    col_strats = sorted({pair[1] for pair in game_dict.keys()}, key=_strategy_key)
    
    # Build payoff matrices.
    n_rows = len(row_strats)
    n_cols = len(col_strats)
    row_matrix = np.zeros((n_rows, n_cols))
    col_matrix = np.zeros((n_rows, n_cols))
    
    for i, r in enumerate(row_strats):
        for j, c in enumerate(col_strats):
            pay = game_dict.get((r, c))
            if pay is None:
                raise ValueError(f"Missing payoff for strategy pair: {(r, c)}")
            row_matrix[i, j] = pay[0]
            col_matrix[i, j] = pay[1]
    
    return row_matrix, col_matrix, row_strats, col_strats