import functools
from typing import Callable


def coordinates_validator(*param_names: str):
    """
    Decorator to make sure coordinates come in as tuples of X, Y numbers

    :param param_names:  string names of the parameters to validate
    :type param_names: str
    :return: decorator function
    :rtype: Callable
    """

    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # Retrieve the target parameter names
            params = function.__code__.co_varnames[: function.__code__.co_argcount]

            # get positional and keyword arguments
            args_dict = dict(zip(params, args))
            args_dict.update(kwargs)

            # Perform validation on specified parameters
            for name in param_names:
                coordinates = args_dict.get(name)

                # if simple tuple, check the two elements
                if isinstance(coordinates, tuple):
                    if len(coordinates) != 2 or not all(
                        isinstance(num, (int, float)) for num in coordinates
                    ):
                        raise ValueError(
                            f"Parameter {name} must be a tuple of two numbers (int or float)."
                        )
                # if list of tuples, check each tuple
                elif isinstance(coordinates, list):
                    # Validate if it's a list of tuples
                    if not all(
                        isinstance(coord, tuple)
                        and len(coord) == 2
                        and all(isinstance(num, (int, float)) for num in coord)
                        for coord in coordinates
                    ):
                        raise ValueError(
                            f"Parameter {name} must be a list of coordinate tuples (float, float)."
                        )
                # if neither of those, raise
                else:
                    raise ValueError(
                        f"Parameter {name} must be a list of tuples or a tuple."
                    )

            # Call the original function
            return function(*args, **kwargs)

        return wrapper

    return decorator
