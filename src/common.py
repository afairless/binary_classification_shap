
import time
import numpy as np
from pathlib import Path
from typing import Callable
from dataclasses import dataclass


##################################################
# FUNCTIONS FOR TEXT FILES
##################################################

def read_text_file(
    text_filename: str | Path, return_string: bool=False, 
    keep_newlines: bool=False):
    """
    Reads text file
    If 'return_string' is 'True', returns text in file as a single string
    If 'return_string' is 'False', returns list so that each line of text in
        file is a separate list item
    If 'keep_newlines' is 'True', newline markers '\n' are retained; otherwise,
        they are deleted

    :param text_filename: string specifying filepath or filename of text file
    :param return_string: Boolean indicating whether contents of text file
        should be returned as a single string or as a list of strings
    :return:
    """

    text_list = []

    try:
        with open(text_filename) as text:
            if return_string:
                # read entire text file as single string
                if keep_newlines:
                    text_list = text.read()
                else:
                    text_list = text.read().replace('\n', '')
            else:
                # read each line of text file as separate item in a list
                for line in text:
                    if keep_newlines:
                        text_list.append(line)
                    else:
                        text_list.append(line.rstrip('\n'))
            text.close()

        return text_list

    except:

        return ['There was an error when trying to read text file']


def write_list_to_text_file(
    a_list: list[str], text_filename: Path | str, overwrite: bool=False):
    """
    Writes a list of strings to a text file
    If 'overwrite' is 'True', any existing file by the name of 'text_filename'
        will be overwritten
    If 'overwrite' is 'False', list of strings will be appended to any existing
        file by the name of 'text_filename'

    :param a_list: a list of strings to be written to a text file
    :param text_filename: a string denoting the filepath or filename of text
        file
    :param overwrite: Boolean indicating whether to overwrite any existing text
        file or to append 'a_list' to that file's contents
    :return:
    """

    if overwrite:
        append_or_overwrite = 'w'
    else:
        append_or_overwrite = 'a'

    try:
        text_file = open(text_filename, append_or_overwrite, encoding='utf-8')
        for e in a_list:
            text_file.write(str(e))
            text_file.write('\n')

    finally:
        text_file.close()


##################################################
# FUNCTIONS FOR LOOP ITERATION COUNTING
##################################################


def seconds_to_formatted_time_string(seconds: float) -> str:
    """
    Given the number of seconds, returns a formatted string showing the time
        duration
    """

    hour = int(seconds / (60 * 60))
    minute = int((seconds % (60 * 60)) / 60)
    second = seconds % 60

    return '{}:{:>02}:{:>05.2f}'.format(hour, minute, second)


def print_loop_status_with_elapsed_time(
    the_iter: int, every_nth_iter: int, total_iter: int, start_time: float):
    """
    Prints message providing loop's progress for user

    :param the_iter: index that increments by 1 as loop progresses
    :param every_nth_iter: message should be printed every nth increment
    :param total_iter: total number of increments that loop will run
    :param start_time: starting time for the loop, which should be
        calculated by 'import time; start_time = time.time()'
    """

    current_time = time.ctime(int(time.time()))

    every_nth_iter_integer = max(round(every_nth_iter), 1)

    if the_iter % every_nth_iter_integer == 0:
        print('Processing loop iteration {i} of {t}, which is {p:0f}%, at {c}'
              .format(i=the_iter + 1,
                      t=total_iter,
                      p=(100 * (the_iter + 1) / total_iter),
                      c=current_time))
        elapsed_time = time.time() - start_time

        print('Elapsed time: {}'.format(seconds_to_formatted_time_string(
            elapsed_time)))


##################################################
# FUNCTIONS FOR MODELING
##################################################


@dataclass
class BestThresholdMetric:
    threshold: float
    metric: float


def get_binary_classification_threshold_and_best_metric(
    true_y_binary_categories: np.ndarray,
    y_prediction_probabilities: np.ndarray,
    scoring_func: Callable) -> BestThresholdMetric:
    """
    Given a vector of true/correct binary categories and a vector of 
        classification probabilities, find the classification threshold (to a
        precision of 0.01) that maximizes the metric given by 'scoring_func'
    """

    assert y_prediction_probabilities.ndim == 1
    assert true_y_binary_categories.ndim == 1
    assert len(y_prediction_probabilities) == len(true_y_binary_categories)

    thresholds = np.linspace(0.01, 0.99, 99)
    metrics = [
        scoring_func(true_y_binary_categories, y_prediction_probabilities > t) 
        for t in thresholds]
    best_metric = np.max(metrics)
    threshold = thresholds[np.argmax(metrics)]

    threshold_metric = BestThresholdMetric(threshold, best_metric)

    return threshold_metric


