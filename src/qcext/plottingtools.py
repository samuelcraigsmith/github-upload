"""Tools for extracting data from qecsim and qcext for plotting."""
import math
import os
import json


def read_data(path):
    """
    Return data from all .json files in path as a list of data points.

    Haven't checked that this actually works...
    """

    cwd_init = os.getcwd()  # will return to cwd to avoid side effects.
    os.chdir(path)

    data_files = [data_file for data_file in os.listdir()
                  if os.path.splitext(data_file)[-1] == ".json"
                  and data_file[0] != "."]  # ignore hidden files

    data_set = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            data_point = json.load(f)
            if "code" not in data_point[0].keys():
                raise ValueError("Path contains unrecognisable files.")
            data_set.append(data_point)

    os.chdir(cwd_init)

    return data_set


def extract_threshold_data(data_set, x_data=None, x_data_label=None, c=0.01):
    """
    Extract thresholds from a list of data points.

    Data set are a list of python dictionaries. The error bars are attained by
    treating the fail rate as a normally distributed random variable. This
    breaks down as f goes to zero. For all f<c, we take f=c to calculate
    generous error bars. For n=500 runs for each data point, c=0.01 is
    appropriate.

    Where x_data is in "error_probability", "time_steps", list (custom)
    and defaults to "error_probability". x_data_label defaults to x_data if
    x_data is a string or the empty string.

    Return a list of threshold data and a list of labels for the data.
    """

    # defaults
    x_data = "error_probability" if x_data is None else x_data
    if x_data_label is None:
        x_data_label = x_data if type(x_data) is str else ""

    xs = {}
    n_runs = {}
    n_fails = {}
    threshold_data = {}
    for i, data_point in enumerate(data_set):
        data_point = data_point[0]  # qecsim returns a dict in a list.
        d = str(data_point["n_k_d"][2])
        if d not in threshold_data.keys():
            xs[d] = []
            n_runs[d] = []
            n_fails[d] = []
            threshold_data[d] = [[], [], []]

        # index into x_data instead of zip due to conditional nature of x_data.
        x = data_point[x_data] if type(x_data) is str else x_data[i]
        try:
            i = xs[d].index(x)
            n_runs[d][i] += data_point["n_run"]
            n_fails[d][i] += data_point["n_fail"]
        except ValueError:
            xs[d].append(x)
            n_runs[d].append(data_point["n_run"])
            n_fails[d].append(data_point["n_fail"])

    for d in threshold_data.keys():
        threshold_data[d][0] = xs[d]
        threshold_data[d][1] = [nf / nr for nr, nf in
                                zip(n_runs[d], n_fails[d])]
        threshold_data[d][2] = [
            math.sqrt(max(f, c) * (1 - max(f, c)) / nr)
            for f, nr in zip(threshold_data[d][1], n_runs[d])
        ]

    return threshold_data, [x_data_label, "f", "df"]

# def threshold_plot(data_set):
#     fig, ax = plt.subplots()
#     # extract code_distances.
#     for d in code_distances:
#         # extract p_series, f_series, err_series.
#         ax.errorbar(p_series, f_series, err_series, label=d, linestyle="None")
#     ax.legend()
#     ax.set_title("Threshold of colour code with restricted error model"
#                  + "(100% bias, n_cyc=5)")
#     ax.set_xlabel("Depolarising parameter p")
#     ax.set_ylabel("Failure rate")
#     return ax

# still playing around with git
