from collections.abc import Callable
from typing import Any

from sympy import Matrix
from sympy.matrices.exceptions import NonSquareMatrixError
from transformers.utils.chat_template_utils import get_json_schema

from linalg_zero.generator.difficulty_config import Precision
from linalg_zero.generator.sympy.template_engine import MathFormatter
from linalg_zero.shared.types import assert_lib_returns


def matrix_transpose(matrix: list[list[float | int]]) -> list[list[float | int]]:
    """Return the transpose of a matrix.

    Args:
        matrix: Matrix represented as a list of rows (list[list[float | int]]).

    Returns:
        list[list[float | int]]: Transposed matrix (rows and columns swapped).

    Examples:
        >>> matrix_transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
        >>> matrix_transpose([[1]])
        [[1]]
    """
    try:
        sym_matrix = Matrix(matrix)
        transpose_result = sym_matrix.T

        result = MathFormatter.sympy_to_primitive(transpose_result, precision=Precision.MATRIX_TRANSPOSE)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result

    except Exception as e:
        raise ValueError(f"Cannot calculate matrix transpose: {e}") from e

    raise TypeError(f"Expected list of lists, got {type(result)}")


def matrix_cofactor(matrix: list[list[float | int]]) -> list[list[float | int]]:
    """Return the cofactor matrix of a square matrix.

    Args:
        matrix: Square matrix as a list of rows (list[list[float | int]], n x n).

    Returns:
        list[list[float | int]]: Cofactor matrix with the same shape as the input.

    Raises:
        ValueError: If the input matrix is not square.

    Examples:
        >>> matrix_cofactor([[1, 2], [3, 4]])
        [[4, -3], [-2, 1]]
        >>> matrix_cofactor([[1]])
        [[1]]
    """
    try:
        sym_matrix = Matrix(matrix)

        cofactor_result = sym_matrix.cofactor_matrix()

        result = MathFormatter.sympy_to_primitive(cofactor_result, precision=Precision.MATRIX_COFACTOR)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result

    except NonSquareMatrixError as e:
        raise ValueError(f"Matrix must be square for cofactor calculation: {e}") from e
    except Exception as e:
        raise ValueError(f"Cannot calculate cofactor matrix: {e}") from e

    raise TypeError(f"Expected list of lists, got {type(result)}")


def determinant(matrix: list[list[float | int]]) -> float:
    """Return the determinant of a square matrix.

    Args:
        matrix: Square matrix as a list of rows (list[list[float | int]], n x n).

    Returns:
        float: Determinant value.

    Examples:
        >>> determinant([[1, 2], [3, 4]])
        -2.0
        >>> determinant([[2, 0], [0, 3]])
        6.0
    """
    try:
        sym_matrix = Matrix(matrix)

        det_result = sym_matrix.det()
        result = MathFormatter.sympy_to_primitive(det_result, precision=Precision.DETERMINANT)

        if isinstance(result, int | float):
            return float(result)

    except NonSquareMatrixError as e:
        raise ValueError("Matrix must be square") from e
    except Exception as e:
        raise ValueError(f"Cannot calculate determinant: {e}") from e

    raise TypeError(f"Expected numeric result, got {type(result)}")


def frobenius_norm(matrix: list[list[float | int]]) -> float:
    """Return the Frobenius norm of a matrix.

    Args:
        matrix: Matrix as a list of rows (list[list[float | int]]).

    Returns:
        float: Frobenius norm value.

    Examples:
        >>> frobenius_norm([[1, 2], [3, 4]])
        5.48
        >>> frobenius_norm([[0, 0], [0, 0]])
        0.0
    """
    try:
        sym_matrix = Matrix(matrix)

        # Calculate Frobenius norm: sqrt(sum of squared elements)
        norm_result = sym_matrix.norm()
        result = MathFormatter.sympy_to_primitive(norm_result, precision=Precision.FROBENIUS_NORM)

        if isinstance(result, int | float):
            return float(result)

    except Exception as e:
        raise ValueError(f"Cannot calculate Frobenius norm: {e}") from e

    raise TypeError(f"Expected numeric result, got {type(result)}")


def matrix_rank(matrix: list[list[float | int]]) -> int:
    """Return the rank of a matrix.

    Args:
        matrix: Matrix as a list of rows (list[list[float | int]]).

    Returns:
        int: Rank (non-negative integer).

    Examples:
        >>> matrix_rank([[1, 2], [3, 4]])
        2
        >>> matrix_rank([[1, 2], [2, 4]])
        1
    """
    try:
        sym_matrix = Matrix(matrix)
        rank_result = sym_matrix.rank()

        if isinstance(rank_result, int):
            return rank_result

    except Exception as e:
        raise ValueError(f"Cannot calculate matrix rank: {e}") from e

    raise TypeError(f"Expected integer result, got {type(rank_result)}")


def matrix_trace(matrix: list[list[float | int]]) -> float:
    """Return the trace of a square matrix.

    Args:
        matrix: Square matrix as a list of rows (list[list[float | int]], n x n).

    Returns:
        float: Trace (sum of diagonal entries).

    Examples:
        >>> matrix_trace([[1, 2], [3, 4]])
        5.0
        >>> matrix_trace([[5]])
        5.0
    """
    try:
        sym_matrix = Matrix(matrix)

        trace_result = sym_matrix.trace()
        result = MathFormatter.sympy_to_primitive(trace_result, precision=Precision.MATRIX_TRACE)

        if isinstance(result, int | float):
            return float(result)

    except NonSquareMatrixError as e:
        raise ValueError("Trace is only defined for square matrices.") from e
    except Exception as e:
        raise ValueError(f"Cannot calculate matrix trace: {e}") from e

    raise TypeError(f"Expected numeric result, got {type(result)}")


def get_lib() -> dict[str, Callable[..., Any]]:
    """Return the library of available functions."""
    return {
        # Matrix results
        "matrix_transpose": matrix_transpose,
        "matrix_cofactor": matrix_cofactor,
        # Scalar results
        "determinant": determinant,
        "frobenius_norm": frobenius_norm,
        "matrix_rank": matrix_rank,
        "matrix_trace": matrix_trace,
    }


def get_lib_fn_names() -> list[str]:
    """Return the names of the functions in the library."""
    return list(get_lib().keys())


def get_tools() -> list[dict[str, Any]]:
    """Returns the tool representation of the functions in the library."""
    return [get_json_schema(func) for func in get_lib().values()]


def get_lib_types_list() -> list[type]:
    """
    Get the list of library return types.
    This is a check to ensure grpo training uses well-tested types in math-verify.
    This only influences the reward functions, and will likely work with other types
    as well. Make sure the types defined below coincide by using this function.
    """
    return assert_lib_returns({float, int, list}, get_lib())
