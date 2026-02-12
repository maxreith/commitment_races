"""
Generate graph for paper.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from num_example import (
    normalize_game, 
    calculate_commitment_probabilities,
    calculate_crashing_probability,
    calculate_expected_payoff
)

# Default T values to plot
T_values = [1, 2, 3, 4, 5]

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
    
    # Create the plot with styling to match LaTeX paper
    plt.figure(figsize=(7, 5))
    
    # Plot each series
    for T_label, probs in commitment_probabilities.items():
        T = int(T_label.split('=')[1])
        x_values = list(range(1, T + 1))
        y_values = probs
        plt.plot(x_values, y_values, marker='o', label=f'$T={T}$', linewidth=2, markersize=6)
    
    # Styling to match LaTeX document
    plt.xlabel('Period ($t$)', fontsize=11)
    plt.ylabel('Commitment Probability ($p_t$)', fontsize=11)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.xlim(0.5, max(T_values) + 0.5)
    plt.ylim(0, max([max(probs) for probs in commitment_probabilities.values()]) * 1.1)
    plt.xticks(range(1, max(T_values) + 1), fontsize=10)
    plt.yticks(fontsize=10)
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Create the bld directory at the same level as Code
    bld_dir = Path(__file__).parent.parent / 'bld'
    bld_dir.mkdir(exist_ok=True)
    
    # Save the plot
    output_path_pdf = bld_dir / 'commitment_probabilities.pdf'
    output_path_png = bld_dir / 'commitment_probabilities.png'
    
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=300)


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