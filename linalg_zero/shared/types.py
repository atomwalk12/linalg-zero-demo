from collections.abc import Callable
from typing import Any, get_origin, get_type_hints


def assert_lib_returns(tested_types: set[type], lib_functions: dict[str, Callable[..., Any]]) -> list[type]:
    """
    This function extracts all function return types from the library.
    The reward functions used during GRPO were tested against these types.
    """
    return_types = set()

    for func in lib_functions.values():
        type_hints = get_type_hints(func)
        if "return" in type_hints:
            type_hint = type_hints["return"]
            base_type = get_origin(type_hint) or type_hint

            if base_type not in tested_types:
                raise ValueError(f"Unexpected return type: {type_hint}")

            return_types.add(base_type)

    if len(return_types) == 0:
        raise ValueError("No return types found")

    return list(return_types)


LibTypes = float | int | list
