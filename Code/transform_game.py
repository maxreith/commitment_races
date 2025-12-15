import numpy as np
import nashpy as nash

def transform_game(game):
    "Transforms a base game into a commitment game"
    # Extract original payoff matrices
    payoff_p1, payoff_p2 = game.payoff_matrices

    # Get dimensions of the original game
    m, n = payoff_p1.shape

    # Create new larger payoff matrices, initialized with zeros
    payoff_p1_new = np.zeros((m * 2, n * 2))
    payoff_p2_new = np.zeros((m * 2, n * 2))

    # Copy the original game's payoffs
    payoff_p1_new[:m, :n] = payoff_p1
    payoff_p2_new[:m, :n] = payoff_p2

    # Calculate payoffs for Player 1's new strategy (the new last row)
    for j in range(n):  # For each of Player 2's original strategies
        best_response_payoff_p1_idx = np.argmax(payoff_p1[:, j])
        payoff_p1_new[m:2*m, j] = payoff_p1[best_response_payoff_p1_idx, j]
        payoff_p2_new[m:2*m, j] = payoff_p2[best_response_payoff_p1_idx, j]

    # Calculate payoffs for Player 2's new strategy (the new last column)
    for i in range(m):  # For each of Player 1's original strategies
        best_response_payoff_p2_idx = np.argmax(payoff_p2[i, :])
        payoff_p2_new[i, n:2*n] = payoff_p2[i, best_response_payoff_p2_idx]
        payoff_p1_new[i, n:2*n] = payoff_p1[i, best_response_payoff_p2_idx]
    
    # The bottom right corner
    # Bottom right corner: copy the original game's payoffs
    payoff_p1_new[m:2*m, n:2*n] = payoff_p1
    payoff_p2_new[m:2*m, n:2*n] = payoff_p2

    # Create and return the new game
    return nash.Game(payoff_p1_new, payoff_p2_new)