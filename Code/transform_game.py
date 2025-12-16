import numpy as np
import nashpy as nash

def transform_game(game):
    "Transforms a base game into a commitment game"
    
    if not isinstance(game, nash.Game):
        raise TypeError("Input is not a nash.Game object, but of type {}".format(type(game)))

    _check_for_unique_best_responses(game)

    payoff_p1, payoff_p2 = game.payoff_matrices

    num_strategies_p1, num_strategies_p2 = payoff_p1.shape

    commitment_matrix_p1 = np.zeros((num_strategies_p1 * 2, num_strategies_p2 * 2))
    commitment_matrix_p2 = np.zeros((num_strategies_p1 * 2, num_strategies_p2 * 2))

    commitment_matrix_p1[:num_strategies_p1, :num_strategies_p2] = payoff_p1
    commitment_matrix_p2[:num_strategies_p1, :num_strategies_p2] = payoff_p2

    for j in range(num_strategies_p2): 
        best_response_payoff_p1_idx = np.argmax(payoff_p1[:, j])
        commitment_matrix_p1[num_strategies_p1:2*num_strategies_p1, j] = payoff_p1[best_response_payoff_p1_idx, j]
        commitment_matrix_p2[num_strategies_p1:2*num_strategies_p1, j] = payoff_p2[best_response_payoff_p1_idx, j]

    for i in range(num_strategies_p1):
        best_response_payoff_p2_idx = np.argmax(payoff_p2[i, :])
        commitment_matrix_p2[i, num_strategies_p2:2*num_strategies_p2] = payoff_p2[i, best_response_payoff_p2_idx]
        commitment_matrix_p1[i, num_strategies_p2:2*num_strategies_p2] = payoff_p1[i, best_response_payoff_p2_idx]
    
    commitment_matrix_p1[num_strategies_p1:2*num_strategies_p1, num_strategies_p2:2*num_strategies_p2] = payoff_p1
    commitment_matrix_p2[num_strategies_p1:2*num_strategies_p1, num_strategies_p2:2*num_strategies_p2] = payoff_p2

    return nash.Game(commitment_matrix_p1, commitment_matrix_p2)

def _check_for_unique_best_responses(game):
    payoff_p1, payoff_p2 = game.payoff_matrices
    num_strategies_p1, num_strategies_p2 = payoff_p1.shape

    multiple_br_dict = {}
    #import pdbp; breakpoint()
    for j in range(num_strategies_p2):
        best_responses = np.where(payoff_p1[:, j] == np.max(payoff_p1[:, j]))[0]
        if len(best_responses) > 1 and not _all_equal(payoff_p2[best_responses, j]):
            multiple_br_dict[f"1 has multiple best responses for P2_strategy_{j}"] = best_responses.tolist()
        
    for i in range(num_strategies_p1):
        best_responses = np.where(payoff_p2[i, :] == np.max(payoff_p2[i, :]))[0]
        if len(best_responses) > 1 and not _all_equal(payoff_p1[i, best_responses]):
            multiple_br_dict[f"2 has multiple best responses for P1_strategy_{i}"] = best_responses.tolist()
    
    if multiple_br_dict:
        raise ValueError(f"Multiple best responses which are payoff relevant: {multiple_br_dict}")
    
def _all_equal(array):
    return np.all(array == array[0])


# Example usage:
base_game = nash.Game([[0, 0], [0, 1]], [[3, 5], [3, 1]])
commitment_game = transform_game(base_game)
print("Commitment Game Payoff Matrices:")
print("Player 1:\n", commitment_game.payoff_matrices[0])
print("Player 2:\n", commitment_game.payoff_matrices[1])
