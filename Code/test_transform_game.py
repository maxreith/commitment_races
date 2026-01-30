import pytest
import numpy as np
import nashpy as nash
from transform_game import _transform_game, _initialize_commitment_matrices, _fill_p1_commitment_region, _fill_p2_commitment_region, _fill_simultaneous_commitment_region


@pytest.fixture
def game_of_chicken():
    """
    Standard 2x2 game of chicken.
    Strategies: [Swerve, Straight]
    Payoff matrix for player 1 (player 2's is transposed for symmetric game):
                P2: Swerve  Straight
    P1: Swerve      0         -1
        Straight    1         -10
    
    Returns
    -------
    nash.Game
        The game of chicken as a nashpy Game object.
    """
    payoff_p1 = np.array([
        [-1, 1],
        [-0.5, 0]
    ])
    payoff_p2 = payoff_p1.T
    return nash.Game(payoff_p1, payoff_p2)

def test_transform_game_with_game_of_chicken(game_of_chicken):
    """
    Test the transformation of the game of chicken into its commitment version.
    """
    transformed_game = _transform_game(game_of_chicken, T=1)
    
    # Check that transformed_game is 4x4 as expected
    assert transformed_game.payoff_matrices[0].shape == (4, 4), "Transformed game should have 4 strategies for each player."

    # Check that expected payoffs are correctly placed in the transformed game
    expected_p1_matrix = np.array([
        [-1, 1, 1, 1],
        [-0.5, 0, -0.5, -0.5],
        [-0.5, 1, -1, 1],
        [-0.5, 1, -0.5, 0]
    ])
    expected_p2_matrix = expected_p1_matrix.T
    np.testing.assert_array_equal(transformed_game.payoff_matrices[0], expected_p1_matrix, "Player 1's payoff matrix is incorrect.")
    np.testing.assert_array_equal(transformed_game.payoff_matrices[1], expected_p2_matrix, "Player 2's payoff matrix is incorrect.")

@pytest.mark.parametrize("T", [0, 1, 2, 5, 10])
def test_initialize_commitment_matrices(game_of_chicken, T):
    """
    Test the initialization of commitment matrices for various values of T.
    """
    payoff_p1, payoff_p2 = game_of_chicken.payoff_matrices
    num_strategies_p1, num_strategies_p2 = payoff_p1.shape
    commitment_matrix_p1, commitment_matrix_p2 = _initialize_commitment_matrices(
        payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, T
    )
    
    # Check dimensions
    expected_size = num_strategies_p1 * (T + 1)
    assert commitment_matrix_p1.shape == (expected_size, expected_size), f"Commitment matrix for player 1 should be {expected_size}x{expected_size} for T={T}."
    assert commitment_matrix_p2.shape == (expected_size, expected_size), f"Commitment matrix for player 2 should be {expected_size}x{expected_size} for T={T}."
    
    # Check that base game payoffs are correctly placed
    np.testing.assert_array_equal(commitment_matrix_p1[:num_strategies_p1, :num_strategies_p2], payoff_p1, f"Base game payoffs not correctly placed in player 1's commitment matrix for T={T}.")
    np.testing.assert_array_equal(commitment_matrix_p2[:num_strategies_p2, :num_strategies_p1], payoff_p2, f"Base game payoffs not correctly placed in player 2's commitment matrix for T={T}.")

def test_fill_p1_commitment_region(game_of_chicken):
    """
    Test filling the player 1 commitment region.
    """
    T = 1
    payoff_p1, payoff_p2 = game_of_chicken.payoff_matrices
    num_strategies_p1, num_strategies_p2 = payoff_p1.shape
    commitment_matrix_p1, commitment_matrix_p2 = _initialize_commitment_matrices(
        payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, T
    )
    
    _fill_p1_commitment_region(commitment_matrix_p1, commitment_matrix_p2, 
                               payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, t=1)
    
    # Check that the commitment region for player 1 is filled correctly
    for j in range(num_strategies_p2):
        best_response_idx = np.argmax(payoff_p2[j,:])
        expected_payoff_p1 = payoff_p1[j, best_response_idx]
        expected_payoff_p2 = payoff_p2[j, best_response_idx]
        np.testing.assert_array_equal(
            commitment_matrix_p1[j, num_strategies_p2*1:num_strategies_p2*2],
            expected_payoff_p1,
            err_msg=f"Player 1's commitment region not filled correctly for column {j}."
        )
        np.testing.assert_array_equal(
            commitment_matrix_p2[j, num_strategies_p2*1:num_strategies_p2*2],
            expected_payoff_p2,
            err_msg=f"Player 2's commitment region not filled correctly for column {j}."
        )