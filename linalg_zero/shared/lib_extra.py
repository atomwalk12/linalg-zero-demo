from sympy import Matrix, ShapeError
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.solvers.solvers import NonInvertibleMatrixError

from linalg_zero.generator.difficulty_config import Precision
from linalg_zero.generator.sympy.template_engine import MathFormatter


def multiply_matrices(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    """Multiplies two matrices, or a matrix and a vector, or two vectors.

    Examples:
        >>> multiply_matrices([[1, 2], [3, 4]], [[2, 0], [1, 3]])
        [[4, 6], [10, 12]]
        >>> multiply_matrices([[1, 2], [3, 4]], [[5], [6]])  # matrix x vector
        [[17], [39]]
        >>> multiply_matrices([[1, 0], [0, 1]], [[5, 6], [7, 8]])  # Identity x matrix
        [[5, 6], [7, 8]]

    Args:
        matrix_a (list[list[float]]): The first matrix or vector.
        matrix_b (list[list[float]]): The second matrix or vector.

    Returns:
        list[list[float]]: The product of the two matrices, matrix and vector, or two vectors.
    """
    try:
        sym_a = Matrix(matrix_a)
        sym_b = Matrix(matrix_b)
        result_matrix: Matrix = sym_a * sym_b
        result = MathFormatter.sympy_to_primitive(result_matrix, precision=Precision.MATRIX_VECTOR_MULTIPLICATION)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result
        else:
            raise TypeError(f"Expected list of lists, got {type(result)}")
    except ShapeError as e:
        raise ValueError(f"Matrix dimensions incompatible for multiplication: {e}") from e


def solve_linear_system(matrix_a: list[list[float | int]], vector_b: list[float | int]) -> list[list[float | int]]:
    """Solve the linear system Ax = b for x using SymPy.

    Examples:
        >>> solve_linear_system([[2, 1], [1, 3]], [7, 8])
        [[2.0], [3.0]]
        >>> solve_linear_system([[1, 0], [0, 1]], [5, 3])  # Identity matrix
        [[5.0], [3.0]]

    Args:
        matrix_a: The coefficient matrix as a list of lists.
        vector_b: The right-hand side vector as a list.

    Returns:
        The solution vector x as a list.
    """
    try:
        sym_a = Matrix(matrix_a)
        sym_b = Matrix(vector_b)

        solution_matrix = sym_a.LUsolve(sym_b)

        result = MathFormatter.sympy_to_primitive(solution_matrix, precision=Precision.LINEAR_SYSTEM_SOLVER)

        if isinstance(result, list):
            return result

    except NonInvertibleMatrixError as e:
        raise NonInvertibleMatrixError(f"Cannot solve linear system: {e}") from e
    except ShapeError as e:
        raise ShapeError(f"Matrix dimensions incompatible for solving linear system: {e}") from e

    raise TypeError(f"Expected list, got {type(result)}")


def matrix_inverse(matrix: list[list[float | int]]) -> list[list[float | int]]:
    """Calculate the inverse of a square matrix using SymPy.

    The inverse of a matrix A is the unique matrix A⁻¹ such that A * A⁻¹ = A⁻¹ * A = I,
    where I is the identity matrix. Only defined for square, invertible matrices.

    Examples:
        >>> matrix_inverse([[1, 2], [3, 4]])
        [[-2.0, 1.0], [1.5, -0.5]]
        >>> matrix_inverse([[2, 0], [0, 3]])
        [[0.5, 0.0], [0.0, 0.33333333]]
        >>> matrix_inverse([[1]])  # 1x1 matrix
        [[1.0]]

    Args:
        matrix: The square invertible matrix as a list of lists.

    Returns:
        The inverse of the matrix as a list of lists.

    Raises:
        ValueError: If the matrix is not square or not invertible.
    """
    try:
        sym_matrix = Matrix(matrix)
        inverse_result = sym_matrix.inv()
        result = MathFormatter.sympy_to_primitive(inverse_result, precision=Precision.MATRIX_INVERSE)

        if isinstance(result, list) and all(isinstance(row, list) for row in result):
            return result

    except NonSquareMatrixError as e:
        raise ValueError("Matrix inverse is only defined for square matrices.") from e
    except NonInvertibleMatrixError as e:
        raise ValueError("Matrix is not invertible (determinant is zero).") from e
    except Exception as e:
        raise ValueError(f"Cannot calculate matrix inverse: {e}") from e

    raise TypeError(f"Expected list of lists, got {type(result)}")
