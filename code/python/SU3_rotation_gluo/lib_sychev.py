from typing import Sequence, List, Union

import numpy as np


# bins contain at least bin_size elements
def np_1dim_array_bin_copytokenizer(data, bin_size):
    size = len(data)

    nbins = size // bin_size
    if nbins == 0:
        return None

    bin_sizes = [bin_size for i in range(nbins)]
    residual_size = size - nbins * bin_size

    j = 0
    while residual_size > 0:
        bin_sizes[j] += 1
        residual_size -= 1
        j = (j + 1) % nbins

    bin_border_indices = [0]
    partial = 0
    for i in range(nbins):
        partial += bin_sizes[i]
        bin_border_indices.append(partial)

    result = []

    for i in range(nbins):
        result.append(np.array(data[bin_border_indices[i]: bin_border_indices[i + 1]]))

    return result


def np_1dim_array_jackknife_copytokenizer(data, bin_size):
    size = len(data)

    nbins = size // bin_size
    if nbins < 2:
        return None

    bins = np_1dim_array_bin_copytokenizer(data, bin_size)

    for i in range(nbins):
        result = []
        for j in range(nbins):
            if j == i:
                continue

            for item in bins[j]:
                result.append(item)

        yield result


def mean(data):
    sum = 0
    counter = 0

    for piece in data:
        sum += piece
        counter += 1

    return sum / counter


def st_div(data):
    avg = mean(data)
    sum = 0
    counter = 0

    for piece in data:
        sum += (piece - avg) ** 2
        counter += 1

    return np.sqrt(sum / (counter - 1))


def bootstrap_meanstd_handler(data, bin_size, target_function):
    bins = np_1dim_array_bin_copytokenizer(data, bin_size)
    nbins = len(bins)
    partial_results = [target_function(bin) for bin in bins]

    return mean(partial_results), st_div(partial_results) / np.sqrt(nbins)


def jackknife_meanstd_handler(data, bin_size, target_function):
    bins = np_1dim_array_jackknife_copytokenizer(data, bin_size)
    nbins = len(bins)
    partial_results = [target_function(bin) for bin in bins]

    return mean(partial_results), st_div(partial_results) * (nbins - 1) / np.sqrt(nbins)


def int_log_range(_min: Union[int, float], _max: Union[int, float],
                  factor: float) -> List[int]:
    """Return list of ints that approximates geometric series.

    Args:
        _min: Floor of the resulting sequence (required _min >= 1).
        _max: Ceiling of the resulting sequence (required _max > _min).
        factor: Specifies step of the geometric series (required factor > 1).

    Returns:
        Resulting list of integers forming approximately geometric series.
    """
    result = [int(_min)]
    current = float(_min)
    while current * factor < _max:
        current *= factor
        if int(current) != result[-1]:
            result.append(int(current))
    return result
