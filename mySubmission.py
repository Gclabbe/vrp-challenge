import math
import time
from argparse import ArgumentParser
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from os import path
import ast


def distance(a: Tuple[float], b: Tuple[float]) -> float:
    """
    Calculate the Euclidean distance between two points

    :param a: starting location
    :type a: Tuple[float]
    :param b: ending location
    :type b: Tuple[float]
    :return: distance between the two points
    :rtype: float
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


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
    pickups = loads_df["pickup"][:-1]  # exclude the origin
    loads_df["d2p"] = loads_df.apply(
        lambda row: calc_nn(row["pickup"], row["dropoff"], pickups), axis=1
    )
    loads_df["nn"] = loads_df.apply(lambda row: sorted_indices(row["d2p"]), axis=1)
    loads_df["completed"] = False

    return loads_df


def get_next_job(
    loads_df: pd.DataFrame,
    current_job_idx: int,
    current_time: float,
    max_time: int = 720,
) -> Tuple[int, float]:
    """
    Get the next job to add to the route using a very simple nearest neighbor heuristic

    :param loads_df: dataframe of the loads with pre-calculated distances
    :type loads_df: pd.DataFrame
    :param current_job_idx: index of the job we are currently at
    :type current_job_idx: int
    :param current_time: how much time has been spent already
    :type current_time: float
    :param max_time: maximum time allowed for the driver
    :type max_time: int
    :return: index of the next job to add to the route and the new time used after completing it
    :rtype: Tuple[int, float]
    """
    origin_idx = len(loads_df) - 1
    nn_list = np.array(loads_df.iloc[current_job_idx]["nn"])
    completed = loads_df["completed"].values[nn_list]

    # numpy + list comprehension voodoo.  See if we can speed up all lookup operations
    travel_to_next_pickup_times = np.array(loads_df.iloc[current_job_idx]["d2p"])[
        nn_list
    ]
    next_pickup_to_origin_time = loads_df["pdo"].values[nn_list]

    # pre-calculate the potential total time to complete next job and return to origin
    potential_times = (
        current_time + travel_to_next_pickup_times + next_pickup_to_origin_time
    )

    # let's remove completed jobs from the list or if they are projected to go over time
    mask = (potential_times <= max_time) & (~completed)
    open_jobs = nn_list[mask]

    if open_jobs.size > 0:
        next_job_idx = open_jobs[0]
        travel_to_next_pickup_time = travel_to_next_pickup_times[mask][0]
        next_pickup_to_dropoff_time = loads_df.at[next_job_idx, "p2d"]
        new_time = (
            current_time + travel_to_next_pickup_time + next_pickup_to_dropoff_time
        )

        return next_job_idx.item(), new_time

    # if we get here, we can't fit any more jobs in
    return_to_origin_time = loads_df.at[current_job_idx, "d2o"]
    new_time = current_time + return_to_origin_time
    return origin_idx, new_time


def create_routes(
    loads_df: pd.DataFrame,
    driver_hours_limit: int = 12,
) -> Tuple[pd.DataFrame, dict]:
    """
    Create a route for the driver based on the nearest neighbor heuristic

    :param loads_df: dataframe of the loads with pre-calculated distances
    :type loads_df: pd.DataFrame
    :param driver_hours_limit: how many hours the driver can work in one shift
    :type driver_hours_limit: int
    :return: updated dataframe of the loads and the route taken
    :rtype: Tuple[pd.DataFrame, dict]
    """
    driver_time_limit = driver_hours_limit * 60  # convert hours to minutes
    route = []
    new_time = 0
    origin_idx = len(loads_df) - 1
    current_job_idx = origin_idx

    open_jobs = loads_df.index[loads_df["completed"] == False].tolist()[:-1]

    while open_jobs:
        next_job_idx, new_time = get_next_job(
            loads_df,
            current_job_idx=current_job_idx,
            current_time=new_time,
            max_time=driver_time_limit,
        )

        if next_job_idx == origin_idx:
            # if next job is the origin, we are done
            break

        loads_df.at[next_job_idx, "completed"] = True
        route.append(loads_df.at[next_job_idx, "loadNumber"])
        current_job_idx = next_job_idx
        open_jobs.remove(next_job_idx)

    return loads_df, {"route": route, "time": new_time}


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "filename", type=str, default="", help="file to run the program on"
    )
    args = parser.parse_args()

    # start = time.time()
    loads_df = load_data(filename=args.filename)
    # print(f"Time to load data: {time.time() - start :.5f} seconds")

    # starta = time.time()
    # start = time.time()
    origin = (0, 0)
    origin_idx = len(loads_df) - 1
    driver_hours_limit = 12  # hours
    routes = []
    minutes = []

    open_jobs = set(loads_df.index[loads_df["completed"] == False].tolist())
    open_jobs.discard(0)
    # print(f"Time to process data: {time.time() - start :.5f} seconds")

    while open_jobs:
        # start = time.time()
        loads_df, driver_data = create_routes(
            loads_df=loads_df,
            driver_hours_limit=driver_hours_limit,
        )
        if driver_data["time"] > 0:
            routes.append(driver_data["route"])
            minutes.append(driver_data["time"])

        closed_this_route = set(driver_data["route"])
        open_jobs -= closed_this_route
        # print(f"Time to create route: {time.time() - start :.5f} seconds")

        # print("*" * 20)
        # print(driver_data)
        # print("*" * 20)
        # # print(loads_df)

    # print(f"Execution time: {time.time() - starta :0.5f} seconds")
    # print(len(routes))
    # # print(routes)
    # print(500 * len(routes) + sum(minutes))

    for this_route in routes:
        print(this_route)
