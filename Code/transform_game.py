import numpy as np
import nashpy as nash

def compute_base_and_commitment_equilibria(payoff_player1, filter_for_symmetric = False, round_to_2 = False):
    """
    Compare equilibria of symmetric base game with its commitment game transformation.
    
    Parameters
    ---------- 
    base_game: np.array
        Payoff array of one player.
    filter_for_symmetric: bool, optional
        Whether to only focus on symmetric equilibria
    round_to_2: bool, optional
        Whether to round equilibria to 2 decimal places

    Returns
    -------
    tuple
        (equilibria_dict, games_dict) where games_dict contains the nash.Game objects
    """
    payoff_player2 = payoff_player1.T
    base_game = nash.Game(payoff_player1, payoff_player2)
    commitment_game = _transform_game(base_game)
    
    equilibria = {
        "base_game_equilibria": list(base_game.support_enumeration()),
        "commitment_game_equilibria": list(commitment_game.support_enumeration())
    }
    
    games = {
        "base_game": base_game,
        "commitment_game": commitment_game
    }
    
    if round_to_2:
        equilibria = _round_equilibria(equilibria)
    if filter_for_symmetric:
        equilibria = _return_only_symmetric_equilibira(equilibria)
    
    return equilibria, games

def print_equilibria(equilibrium_dic, games_dic=None):
    """
    Print equilibria in a readable format with payoffs if games provided.
    
    Parameters
    ----------
    equilibrium_dic : dict
        Dictionary containing equilibria for different games.
    games_dic : dict, optional
        Dictionary containing nash.Game objects to calculate payoffs.
    """
    for game_type, equilibria_list in equilibrium_dic.items():
        print(f"\n{game_type.replace('_', ' ').title()}:")
        if not equilibria_list:
            print("  No equilibria found.")
        else:
            # Get corresponding game for payoff calculation
            game_key = game_type.replace('_equilibria', '')
            game = games_dic.get(game_key) if games_dic else None
            
            for i, eq in enumerate(equilibria_list, 1):
                print(f"  Equilibrium {i}: P1={eq[0]}, P2={eq[1]}")
                
                if game is not None:
                    # Calculate expected payoffs using the game's payoff matrices
                    payoff_p1_matrix, payoff_p2_matrix = game.payoff_matrices
                    payoff_p1 = eq[0] @ payoff_p1_matrix @ eq[1]
                    payoff_p2 = eq[0] @ payoff_p2_matrix @ eq[1]
                    print(f"    Expected payoffs: P1={payoff_p1:.3f}, P2={payoff_p2:.3f}")

def _transform_game(game, T):
    "Transforms a base game into a commitment game with T commitment periods."
    
    if not isinstance(game, nash.Game):
        raise TypeError("Input is not a nash.Game object, but of type {}".format(type(game)))

    _check_for_unique_best_responses(game)

    if not (isinstance(T, int) and T > 0):
        raise ValueError("T must be a positive integer.")

    payoff_p1, payoff_p2 = game.payoff_matrices
    num_strategies_p1, num_strategies_p2 = payoff_p1.shape

    # Initialize commitment matrices
    commitment_matrix_p1, commitment_matrix_p2 = _initialize_commitment_matrices(
        payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, T
    )

    for t in range(1, T+1):
        # Fill player 1 commitment region (rows num_strategies_p1:2*num_strategies_p1, cols 0:num_strategies_p2)
        _fill_p1_commitment_region(
            commitment_matrix_p1, commitment_matrix_p2, payoff_p1, payoff_p2, 
            num_strategies_p1, num_strategies_p2, t
        )

        # Fill player 2 commitment region (rows 0:num_strategies_p1, cols num_strategies_p2:2*num_strategies_p2)
        _fill_p2_commitment_region(
            commitment_matrix_p1, commitment_matrix_p2, payoff_p1, payoff_p2,
            num_strategies_p1, num_strategies_p2, t
        )

        # Fill simultaneous commitment region
        _fill_simultaneous_commitment_region(
            commitment_matrix_p1, commitment_matrix_p2, payoff_p1, payoff_p2,
            num_strategies_p1, num_strategies_p2
        )

    return nash.Game(commitment_matrix_p1, commitment_matrix_p2)


def _initialize_commitment_matrices(payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, T):
    """Initialize commitment game matrices with base game payoffs in top-left quadrant."""
    commitment_matrix_p1 = np.zeros((num_strategies_p1 * (T+1), num_strategies_p2 * (T+1)))
    commitment_matrix_p2 = np.zeros((num_strategies_p1 * (T+1), num_strategies_p2 * (T+1)))
    
    commitment_matrix_p1[:num_strategies_p1, :num_strategies_p2] = payoff_p1
    commitment_matrix_p2[:num_strategies_p1, :num_strategies_p2] = payoff_p2
    
    return commitment_matrix_p1, commitment_matrix_p2


def _fill_p2_commitment_region(commitment_matrix_p1, commitment_matrix_p2, 
                                payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, t):
    """Fill the region where player 2 commits (bottom-left quadrant)."""
    for j in range(num_strategies_p2):
        best_response_idx = np.argmax(payoff_p1[:, j])
        commitment_matrix_p1[num_strategies_p1*t:num_strategies_p1*(t+1), j] = payoff_p1[best_response_idx, j]
        commitment_matrix_p2[num_strategies_p1*t:num_strategies_p1*(t+1), j] = payoff_p2[best_response_idx, j]


def _fill_p1_commitment_region(commitment_matrix_p1, commitment_matrix_p2,
                                payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2, t):
    """Fill the region where player 1 commits (top-right quadrant)."""
    for i in range(num_strategies_p1):
        best_response_idx = np.argmax(payoff_p2[i, :])
        commitment_matrix_p2[i, num_strategies_p2*t:num_strategies_p2*(t+1)] = payoff_p2[i, best_response_idx]
        commitment_matrix_p1[i, num_strategies_p2*t:num_strategies_p2*(t+1)] = payoff_p1[i, best_response_idx]


def _fill_simultaneous_commitment_region(commitment_matrix_p1, commitment_matrix_p2,
                                          payoff_p1, payoff_p2, num_strategies_p1, num_strategies_p2):
    """Fill the region where both players commit simultaneously (bottom-right quadrant)."""
    commitment_matrix_p1[num_strategies_p1:2*num_strategies_p1, num_strategies_p2:2*num_strategies_p2] = payoff_p1
    commitment_matrix_p2[num_strategies_p1:2*num_strategies_p1, num_strategies_p2:2*num_strategies_p2] = payoff_p2


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

def _return_only_symmetric_equilibira(equilibria, tolerance=1e-10):
    """
    Filter equilibria dictionary to only include symmetric equilibria.
    
    Parameters
    ----------
    equilibria : dict
        Dictionary with equilibria generators/lists for different games.
    tolerance : float, optional
        Numerical tolerance for comparing strategies (default: 1e-10).
    
    Returns
    -------
    dict
        Filtered dictionary containing only symmetric equilibria.
    """
    return {
    key: [equilibrium for equilibrium in equilibria_list 
          if np.allclose(equilibrium[0], equilibrium[1], atol=tolerance)]
    for key, equilibria_list in equilibria.items()}

def _all_equal(array):
    return np.all(array == array[0])

def _round_equilibria(equilibria, decimals=2):
    """Round equilibrium strategies to specified decimal places."""
    return {
        key: [(np.round(eq[0], decimals), np.round(eq[1], decimals)) for eq in equilibria_list]
        for key, equilibria_list in equilibria.items()
    }