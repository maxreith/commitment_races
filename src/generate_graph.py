"""
Generate graph for paper.
"""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from num_example import (
    normalize_game,
    calculate_commitment_probabilities,
    calculate_crashing_probability,
    calculate_expected_payoff
)

# AAAI rejects Type 3 fonts even inside figures; matplotlib's PDF default is 3.
matplotlib.rcParams['pdf.fonttype'] = 42

# Default T values to plot
T_values = [1, 2, 3, 4, 5]

# Figures are drawn at final print size so LaTeX does not scale them down: a
# scaled figure shrinks its text by the same factor, and AAAI requires in-figure
# text to be at least 9 pt. This is slightly under the 3.325 in column so that
# \includegraphics[width=\linewidth] scales up by ~7%, leaving the 9 pt labels
# above the floor rather than exactly on it.
FIGURE_WIDTH_IN = 3.0
MARKERS = ['o', 's', '^', 'D', 'v']

# T is an ordered quantity, so the series take a single-hue light-to-dark ramp
# rather than a categorical cycle: it encodes the ordering, and its monotone
# lightness survives greyscale printing. The matplotlib default cycle does not --
# its orange and green are dE 0.7 apart under protanopia, i.e. indistinguishable.
SERIES_RAMP = plt.cm.Blues
SERIES_RAMP_RANGE = (0.45, 1.0)


def series_colors(n):
    """Sample the sequential ramp into n evenly spaced, light-to-dark colors."""
    low, high = SERIES_RAMP_RANGE
    if n == 1:
        return [SERIES_RAMP(high)]
    return [SERIES_RAMP(low + (high - low) * i / (n - 1)) for i in range(n)]

def create_probabilities_across_T_graph(A, B, C, D, T_values):
    """
    Create and save commitment probability plot.
    
    Parameters
    ----------
    A, B, C, D : float
        Game parameters
    T_values : list of int
        List of T values to plot (e.g., [1, 2, 3, 4, 5])
    """
    # Normalize the game
    b, c = normalize_game(A, B, C, D)
    
    # Calculate commitment probabilities for each T
    commitment_probabilities = {}
    for T in T_values:
        commitment_probabilities[f'T={T}'] = calculate_commitment_probabilities(T, b, c)[:-1]
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 2.1))

    # Each series is also given its own marker, so the curves stay distinguishable
    # even where the ramp's adjacent steps are close.
    colors = series_colors(len(commitment_probabilities))
    for i, (T_label, probs) in enumerate(commitment_probabilities.items()):
        T = int(T_label.split('=')[1])
        ax.plot(range(1, T + 1), probs, marker=MARKERS[i % len(MARKERS)],
                color=colors[i], label=f'$T={T}$', linewidth=1.1, markersize=3.2)

    # The measure is named in a left-aligned headline rather than a rotated y-axis
    # label, which reads more easily and frees the width for the plot itself.
    ax.set_title('Commitment Probability', loc='center', fontsize=9, pad=6)
    ax.set_xlabel('Period ($t$)', fontsize=9)
    ax.tick_params(labelsize=9, length=2.5, width=0.6)
    ax.set_xticks(range(1, max(T_values) + 1))

    lowest = min(min(probs) for probs in commitment_probabilities.values())
    highest = max(max(probs) for probs in commitment_probabilities.values())
    padding = 0.12 * (highest - lowest)
    ax.set_ylim(lowest - padding, highest + padding)
    ax.set_xlim(0.7, max(T_values) + 0.3)

    ax.grid(True, alpha=0.3, linewidth=0.4)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.6)

    ax.legend(fontsize=9, framealpha=0.9, loc='upper right',
              handlelength=1.6, borderpad=0.35, labelspacing=0.25,
              handletextpad=0.5, borderaxespad=0.3)

    fig.tight_layout(pad=0.2)

    # Create the bld directory at the same level as Code
    bld_dir = Path(__file__).parent.parent / 'bld'
    bld_dir.mkdir(exist_ok=True)

    # Save the plot
    output_path_pdf = bld_dir / 'commitment_probabilities.pdf'
    output_path_png = bld_dir / 'commitment_probabilities.png'

    fig.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(output_path_png, format='png', bbox_inches='tight', dpi=600)
    plt.close(fig)


def create_payoff_and_crash_graph(b_values, c, T_range):
    """
    Create graph showing expected payoff and crashing probability.
    
    Parameters
    ----------
    b_values : list of float
        List of b values to plot
    c : float
        The c parameter value
    T_range : list of int
        Range of T values (e.g., list(range(1, 11)))
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Define color scales
    colors = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef']
    
    for idx, b in enumerate(b_values):
        expected_payoffs = []
        crashing_probabilities = []
        
        for T in T_range:
            expected_payoffs.append(calculate_expected_payoff(T, b, c))
            crashing_probabilities.append(calculate_crashing_probability(T, b, c))
        
        # Plot expected payoff
        ax1.plot(T_range, expected_payoffs, marker='o', label=f'b={b}', 
                color=colors[idx], linewidth=2, markersize=6)
        
        # Plot crashing probability
        ax2.plot(T_range, crashing_probabilities, marker='o', label=f'b={b}', 
                color=colors[idx], linewidth=2, markersize=6)
    
    # Style first subplot
    ax1.set_xlabel('Number of Commitment Periods (T)', fontsize=11)
    ax1.set_ylabel('Expected Payoff', fontsize=11)
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_title(f'Expected Payoff (c={c})', fontsize=12)
    
    # Style second subplot
    ax2.set_xlabel('Number of Commitment Periods (T)', fontsize=11)
    ax2.set_ylabel('Crashing Probability', fontsize=11)
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_title(f'Crashing Probability (c={c})', fontsize=12)
    
    plt.tight_layout()
    
    # Create the bld directory at the same level as Code
    bld_dir = Path(__file__).parent.parent / 'bld'
    bld_dir.mkdir(exist_ok=True)
    
    # Save the plot
    png_path = bld_dir / 'commitment_periods_plot.png'
    pdf_path = bld_dir / 'commitment_periods_plot.pdf'
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    """Main function to generate default commitment probability plot."""
    from num_example import A_DEFAULT, B_DEFAULT, C_DEFAULT, D_DEFAULT
    
    # Generate commitment probability plot
    create_probabilities_across_T_graph(A_DEFAULT, B_DEFAULT, C_DEFAULT, D_DEFAULT, T_values)
    
    # Generate payoff and crash probability plot
    b_values = [1.2, 1.5, 2.0, 2.5, 3.0]
    c = 0.5
    T_range = list(range(1, 11))
    create_payoff_and_crash_graph(b_values, c, T_range)


if __name__ == "__main__":
    main()