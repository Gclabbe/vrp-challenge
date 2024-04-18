import math
from typing import List, Optional, Tuple

# import numpy as np
# import pandas as pd

# from utils.decorators import coordinates_validator


# @coordinates_validator("a", "b")
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


def total_cost(minutes: List[float], driver_cost: int = 500) -> float:
    """
    Calculate the total cost of the routes

    :param minutes: list of minutes for each driver
    :type minutes: List[float]
    :param driver_cost: cost per driver (in minutes basically)
    :type driver_cost: int
    :return: total cost of the routes
    :rtype: float
    """
    n_drivers = len(minutes)
    route_mins = sum(minutes)
    return n_drivers * driver_cost + route_mins
