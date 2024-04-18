from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from os import path
import ast

from utils.math_utils import distance

# from utils.decorators import coordinates_validator


# @coordinates_validator("pickup_loc", "dropoff_loc", "all_pickups")
def calc_nn(
    pickup_loc: Tuple[float], dropoff_loc: Tuple[float], all_pickups: List[Tuple[float]]
) -> List[float]:
    """
    Pre-calculate the distance from dropoff to all other pickup as a cheap nearest neighbor heuristic

    :param pickup_loc: X, Y location of the pickup for this load
    :type pickup_loc: Tuple[float]
    :param dropoff_loc: X, Y location of the dropoff for this load
    :type dropoff_loc: Tuple[float]
    :param all_pickups: list of all other pickup locations
    :type all_pickups: List[Tuple[float]]
    :return: list of distances from this dropoff to all other pickups
    :rtype: List[float]
    """
    return [
        distance(dropoff_loc, x) if x != pickup_loc else np.inf for x in all_pickups
    ]


def sorted_indices(d2p: List[float]) -> List[int]:
    """
    return the index values for a list sorted smallest -> largest

    :param d2p: list of distances from a dropoff to all other pickups
    :type d2p: List[float]
    :return: list of indices sorted by distance
    :rtype: List[int]
    """
    return list(np.argsort(d2p))


def load_data(filename: str, origin: Optional[Tuple[float]] = None) -> pd.DataFrame:
    """
    Load the data from the file and pre-calculate all distances

    :param filename: filename or path to the file
    :type filename: str
    :param origin: coordinates of the origin location (0, 0) by default
    :type origin: Optional[Tuple[float]]
    :return: dataframe of the loads with pre-calculated distances
    :rtype: pd.DataFrame
    """
    origin = origin or (0.0, 0.0)
    sub_dir = "training_problems"

    if sub_dir not in filename:
        # - load data from path/file ... what does the data look like?
        filename = path.join(sub_dir, filename)

    loads_df = pd.read_csv(filename, sep=" ")

    # Convert location data from strings to Tuple[float, float]
    loads_df["pickup"] = loads_df["pickup"].apply(ast.literal_eval)
    loads_df["dropoff"] = loads_df["dropoff"].apply(ast.literal_eval)

    # add a row for the origin at the end of the list
    loads_df.loc[len(loads_df)] = [0, origin, origin]

    # - use numpy or pandas to compute all distances up front (o->p, p->d, d->o)
    loads_df["o2p"] = loads_df.apply(
        lambda row: distance(origin, row["pickup"]), axis=1
    )
    loads_df["p2d"] = loads_df.apply(
        lambda row: distance(row["pickup"], row["dropoff"]), axis=1
    )
    loads_df["d2o"] = loads_df.apply(
        lambda row: distance(row["dropoff"], origin), axis=1
    )
    loads_df["pdo"] = loads_df.apply(lambda row: row["p2d"] + row["d2o"], axis=1)

    # pre-compute d -> all other p in advance for NN heuristic
    pickups = list(loads_df["pickup"][:-1])  # exclude the origin
    loads_df["d2p"] = loads_df.apply(
        lambda row: calc_nn(row["pickup"], row["dropoff"], pickups), axis=1
    )
    loads_df["nn"] = loads_df.apply(lambda row: sorted_indices(row["d2p"]), axis=1)
    loads_df["completed"] = False

    return loads_df
