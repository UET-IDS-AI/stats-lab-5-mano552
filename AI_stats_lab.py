import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)

    count = np.sum((samples > a) & (samples < b))

    return count / n


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    return coefficient * exponent


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.
    """


    P_A = 0.3
    P_B = 0.7

    f_A = np.exp(-((time - 40) ** 2) / 4)
    f_B = np.exp(-((time - 45) ** 2) / 4)

    numerator = f_B * P_B
    denominator = (f_A * P_A) + (f_B * P_B)

    return numerator / denominator


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """

    P_A = 0.3
    P_B = 0.7

    groups = np.random.choice(["A", "B"], size=n, p=[P_A, P_B])

    times = np.zeros(n)

    for i in range(n):
        if groups[i] == "A":
            times[i] = np.random.normal(40, 2)
        else:
            times[i] = np.random.normal(45, 2)

    mask = np.abs(times - time) < 0.5

    selected_groups = groups[mask]

    if len(selected_groups) == 0:
        return 0

    return np.sum(selected_groups == "B") / len(selected_groups)
