def calculate_p_1(T, b, c):
    """
    Calculates the probability of commitment in the first period for the Game of Chicken.
    
    :param T: int
        Total number of commitment periods.
    :param b: float
        Parameter b in the Game of Chicken.
    :param c: float
        Parameter c in the Game of Chicken.
    """
    rho = c/b
    denominator=(1-rho**T)/(1-rho)+(rho**T)*(b+c-1)/(b-1)
    return 1/denominator

def calculate_commitment_probabilities(T, b, c):
    """
    Calculates the probabilities of commitment in each period and the probability of no commitment in the Game of Chicken.
    
    :param T: int
        Total number of commitment periods.
    :param b: float
        Parameter b in the Game of Chicken.
    :param c: float
        Parameter c in the Game of Chicken.
    """
    probabilities = []
    probabilities.append(calculate_p_1(T, b, c))  
    for t in range(2, T+1):
        p_t = calculate_p_1(T, b, c) * (c/b)**(t-1)
        probabilities.append(p_t)
    p_W = 1 - sum(probabilities)
    probabilities.append(p_W)
    return probabilities

def calculate_crashing_probability(T, b, c):
    """
    Calculates the overall crashing probability given T commitment periods in the Game of Chicken.
    
    :param T: int
        Total number of commitment periods.
    :param b: float
        Parameter b in the Game of Chicken.
    :param c: float
        Parameter c in the Game of Chicken.
    """
    probabilities = calculate_commitment_probabilities(T, b, c)
    p_crash_in_commitment_stage = sum([p**2 for p in probabilities[:-1]])
    p_reaching_base_game = probabilities[-1]**2 
    p_crash_in_base_game = p_reaching_base_game * ((b-1)/(b+c-1))**2
    return p_crash_in_commitment_stage + p_crash_in_base_game

def calculate_expected_payoff(T, b, c):
    """
    This formula follows the expected payoff for waiting U(W). In an MSNE, U(W) is equal to the expected payoff for playing any other strategy, so there are many ways to calculate the expected payoff of the MSNE:
    
    :param T: int
        Total number of commitment periods.
    :param b: float
        Parameter b in the Game of Chicken.
    :param c: float
        Parameter c in the Game of Chicken.
    """
    probabilities = calculate_commitment_probabilities(T, b, c)
    commitment_probabilities = probabilities[:-1]
    wait_probability = probabilities[-1]
    expected_payoff = sum([p * c for p in commitment_probabilities]) + wait_probability * (b * c) / (b + c - 1)
    return expected_payoff