"""Helper tools to map biased noise model to one-timestep model."""


import numpy as np
import math
# from matplotlib import pyplot as plt
import json
import logging

from qcext.models.codes import Color666CodeNoisy

logger = logging.getLogger(__name__)


my_vals = {"p": 0.001, "lambda_": 100, "n_high": 1, "n_low": 3, 'cutoff': 50} ### CHANGE BACK TO DEFAULT VALUES


def comb(a, b):
	return math.factorial(a) / (math.factorial(b) * math.factorial(a - b))

def compute_p_eff(time_steps, p, cutoff=math.inf):
    """Compute effective error rate p for single time step simulation."""
    return sum([comb(time_steps, k) * p**k * (1-p)**(time_steps-k)
                for k in range(1, min(time_steps+1, cutoff), 2)])


def extract_probabilities(p, lambda_, n_high, n_low):
    """Extract p_high and p_low from a parameterized noise model."""
    n = n_high + n_low
    p_low = n*p/(n_high*(lambda_+1) + n_low)
    p_high = p_low*(lambda_+1)
    return p_high, p_low


def construct_noise_model(p_high, p_low, n_high, n_low):
    """Construct p and lambda_ corresponding to given error rates."""
    n = n_high + n_low
    p = (n_high*p_high + n_low*p_low)/n
    lambda_ = (p_high/p_low - 1)
    return p, lambda_


def compute_model_eff(time_steps, p, lambda_, n_high, n_low, cutoff=math.inf):
    """Compute an effective error model that compresses multiple time steps.

    Error model is returned as a tuple (p_eff, lambda_eff).
    """
    p_high, p_low = extract_probabilities(p, lambda_, n_high, n_low)
    p_high_eff = compute_p_eff(time_steps, p_high, cutoff=cutoff)
    p_low_eff = compute_p_eff(time_steps, p_low, cutoff=cutoff)
    p_eff, lambda_eff = construct_noise_model(p_high_eff, p_low_eff, n_high,
                                              n_low)
    return p_eff, lambda_eff


def list_to_bash_array(list_):
    """Format a list of numbers as a bash array (4 d.p.)"""
    string_list = [str(_) if type(_) is int else "{:.4f}".format(_) for _ in list_]
    bash_array = "( " + " ".join(string_list) + " )"
    return bash_array


def get_params(time_steps_list, p, lambda_, *args, code=None, cutoff=math.inf,
               type="list"):
    n_high = my_vals["n_high"]  # defaults.
    n_low = my_vals["n_low"]
    try:  # attempt to overwrite defaults in order of preference.
        n_high = len(code.noisy_qubits)
        n_low = code.n_k_d[0] - n_high
    except AttributeError:
        pass
    try:
        n_high = args[0]
        n_low = args[1]
    except IndexError:
        pass

    logger.debug("n_high={}\n n_low={}".format(n_high, n_low))

    model_effs = [
        compute_model_eff(time_steps, p, lambda_, n_high, n_low, cutoff=cutoff)
        for time_steps in time_steps_list
    ]
    p_effs, lambda_effs = (_ for _ in zip(*model_effs))

    if type == "bash array":
        return list_to_bash_array(p_effs), list_to_bash_array(lambda_effs)
    else:
        return p_effs, lambda_effs




def main():
    time_steps_list = list(range(1, 2500))
    model_effs = [compute_model_eff(time_steps, my_vals["p"],
                                    my_vals["lambda_"], my_vals["n_high"],
                                    my_vals["n_low"],
                                    cutoff=my_vals["cutoff"])
                  for time_steps in time_steps_list]
    p_effs, lambda_effs = (list(_) for _ in zip(*model_effs))

    # PLOTTING

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time_steps')
    ax1.set_ylabel('p_eff', color=color)
    ax1.plot(time_steps_list, p_effs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('lambda', color=color)  # we already handled the x-label with ax1
    ax2.plot(time_steps_list, lambda_effs, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle("Effective noise models at $p=0.001$, $lambda=100$ and varying time steps")
    plt.savefig("effective_error_models.png")

    # SAVE FILE

    with open("effective_error_models.json", "w") as f:
        json.dump([time_steps_list, p_effs, lambda_effs], f)


def test():
    logging.basicConfig(level=logging.DEBUG)

    p = 0.001
    lambda_ = 100
    time_steps_list = [200, 210, 220, 230, 240, 250]
    code9 = Color666CodeNoisy(9)
    p_effs_sh, lambda_effs_sh = get_params(time_steps_list, p, lambda_,
                                           code=code9)
    print("Effective error rates: {}\nEffective biases: {}"
          .format(p_effs_sh, lambda_effs_sh))


if __name__ == "__main__":
    test()
