import numpy as np
import nashpy as nash

def transform_game(game):
    # Extract original payoff matrices
    P1, P2 = game.payoff_matrices

    # Get dimensions of the original game
    m, n = P1.shape

    # Create new larger payoff matrices, initialized with zeros
    P1_new = np.zeros((m * 2, n * 2))
    P2_new = np.zeros((m * 2, n * 2))

    # Copy the original game's payoffs
    P1_new[:m, :n] = P1
    P2_new[:m, :n] = P2

    # Calculate payoffs for Player 1's new strategy (the new last row)
    for j in range(n):  # For each of Player 2's original strategies
        best_response_p1_idx = np.argmax(P1[:, j])
        P1_new[m:2*m, j] = P1[best_response_p1_idx, j]
        P2_new[m:2*m, j] = P2[best_response_p1_idx, j]

    # Calculate payoffs for Player 2's new strategy (the new last column)
    for i in range(m):  # For each of Player 1's original strategies
        best_response_p2_idx = np.argmax(P2[i, :])
        P2_new[i, n:2*n] = P2[i, best_response_p2_idx]
        P1_new[i, n:2*n] = P1[i, best_response_p2_idx]
    
    # The bottom right corner
    # Bottom right corner: copy the original game's payoffs
    P1_new[m:2*m, n:2*n] = P1
    P2_new[m:2*m, n:2*n] = P2

    # Create and return the new game
    return nash.Game(P1_new, P2_new)