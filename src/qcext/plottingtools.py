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

    data_files = [data_file for data_file in os.listdir() if
                  os.path.splitext(data_file)[-1] == ".json"]
    data_set = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            data_point = json.load(f)
            if "code" not in data_point[0].keys():
                raise ValueError("Path contains unrecognisable files.")
            data_set.append(data_point)

    os.chdir(cwd_init)

    return data_set


def extract_threshold_data(data_set, c=0.01):
    """
    Extract thresholds from a list of data points.

    Data set are a list of python dictionaries. The error bars are attained by
    treating the fail rate as a normally distributed random variable. This
    breaks down as f goes to zero. For all f<c, we take f=c to calculate
    generous error bars. For n=500 runs for each data point, c=0.01 is
    appropriate.
    """
    threshold_data = {}
    unique_data_points = set()
    for data_point in data_set:
        data_point = data_point[0]  # qecsim returns a dict in a list.
        d = str(data_point["n_k_d"][2])
        if d not in threshold_data.keys():
            threshold_data[d] = [[], [], []]
        p = data_point["error_probability"]

        if (d, p) not in unique_data_points:
            f = data_point["n_fail"] / data_point["n_run"]
            df = math.sqrt(max(f, c) * (1 - max(f, c)) / (data_point["n_run"]))

            threshold_data[d][0].append(p)
            threshold_data[d][1].append(f)
            threshold_data[d][2].append(df)

        else:
            # need to find the entry corresponding to (d, p). This is not fast.
            pass
    return threshold_data

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
