#!/usr/bin/env python3
import matplotlib.pyplot as plt
import csv
import glob
import os
import pprint
import traceback
import numpy as np
import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd

# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


@click.command()
@click.argument("logdir-or-logfile")
@click.option(
    "--write-pkl/--no-write-pkl", help="save to pickle file or not", default=False
)
@click.option(
    "--write-csv/--no-write-csv", help="save to csv file or not", default=True
)
@click.option("--out-dir", "-o", help="output directory", default=".")
def main(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: str):
    """This is a enhanced version of
    https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b
    This script exctracts variables from all logs from tensorflow event
    files ("event*"),
    writes them to Pandas and finally stores them a csv-file or
    pickle-file including all (readable) runs of the logging directory.
    Example usage:
    # create csv file from all tensorflow logs in provided directory (.)
    # and write it to folder "./converted"
    tflogs2pandas.py . --write-csv --no-write-pkl --o converted
    # creaste csv file from tensorflow logfile only and write into
    # and write it to folder "./converted"
    tflogs2pandas.py tflog.hostname.12345 --write-csv --no-write-pkl --o converted
    """
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths)
        pp.pprint("Head of created dataframe")
        pp.pprint(all_logs.head())

        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            print("saving to csv file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.csv")
            print(out_file)
            all_logs.to_csv(out_file, index=None)
        if write_pkl:
            print("saving to pickle file")
            out_file = os.path.join(out_dir, "all_training_logs_in_one_file.pkl")
            print(out_file)
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")

    df = pd.read_csv("all_training_logs_in_one_file.csv")
    # print("Contents in csv file:\n", df)
    x = df.step
    y = df.value
    # plt.plot(x, y)
    # plt.show()
    # fig = plt.figure()
    smooth = []
    smooth.append(df.ewm(alpha=0.05).mean())
    plt.plot(x, y, alpha=0.4, label='raw episode rewards')
    plt.plot(smooth[0].step, smooth[0].value, label='smoothened rewards')
    plt.plot([0, 50000], [-1.19, -1.19], label='no learning reward')
    plt.title('Learning curve for Task 1')
    plt.xlabel('Episode number')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()
    plt.axis([0, 50000, -1.6, 0])
    plt.savefig("kaas.png")
    # df = pd.read_csv("mse_data.csv")
    # print(df)
    #
    # TSBOARD_SMOOTHING = [0.5, 0.85, 0.99]
    #
    # smooth = []
    # for ts_factor in TSBOARD_SMOOTHING:
    #     smooth.append(df.ewm(alpha=(1 - ts_factor)).mean())
    #
    # for ptx in range(3):
    #     plt.subplot(1, 3, ptx + 1)
    #     plt.plot(df["value"], alpha=0.4)
    #     plt.plot(smooth[ptx]["value"])
    #     plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING[ptx]))
    #     plt.grid(alpha=0.3)
    #
    # plt.show()

if __name__ == "__main__":
    main()







