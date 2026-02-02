"""
Generate graph for paper.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from num_example import normalize_game, calculate_commitment_probabilities

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

def main():
    """Main function to generate default commitment probability plot."""
    from num_example import A_DEFAULT, B_DEFAULT, C_DEFAULT, D_DEFAULT
    create_probabilities_across_T_graph(A_DEFAULT, B_DEFAULT, C_DEFAULT, D_DEFAULT, T_values)


if __name__ == "__main__":
    main()