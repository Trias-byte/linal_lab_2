from abc import ABC, abstractmethod
from decimal import Decimal
import copy

class ABCMatrix(ABC):
    """Abstract base class for matrices."""

    @abstractmethod
    def __init__(self, mat: list[list[Decimal | complex | float]]):
        """Initializes the matrix from a list of lists."""
        pass

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """Returns the size of the matrix as a tuple (rows, columns)."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Returns a string representation of the matrix."""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        """Checks if two matrices are equal."""
        pass

    @abstractmethod
    def __neg__(self) -> 'ABCMatrix':
        """Returns the matrix obtained by multiplying the original by -1."""
        pass

    @abstractmethod
    def __add__(self, other) -> 'ABCMatrix':
        """Performs matrix addition."""
        pass

    @abstractmethod
    def __sub__(self, other) -> 'ABCMatrix':
        """Performs matrix subtraction."""
        pass

    @abstractmethod
    def __mul__(self, other) -> 'ABCMatrix':
        """Performs matrix multiplication by a scalar or another matrix."""
        pass

    @abstractmethod
    def __rmul__(self, other) -> 'ABCMatrix':
        """Performs scalar multiplication on the matrix (commutative)."""
        pass

    @abstractmethod
    def __getitem__(self, key: tuple[int, int] | int) -> Decimal | complex | float | list[Decimal | complex | float]:
        """Returns the matrix element by index (row, column) or the entire row."""
        pass

    @abstractmethod
    def augment(self, other: 'ABCMatrix') -> 'ABCMatrix':
        """Augments the matrix by another matrix on the right (column-wise concatenation)."""
        pass

    @abstractmethod
    def transpose(self) -> 'ABCMatrix':
        """Returns the transposed matrix."""
        pass

    @abstractmethod
    def determinant(self) -> Decimal | complex | float:
        """Calculates the determinant of a square matrix."""
        pass


class BaseMatrix(ABCMatrix):
    """Implementation of a matrix using a list of lists."""
    def __init__(self, mat: list[list[Decimal | complex | float]]):
        """Initializes the matrix from a list of lists."""
        if not mat or not all(isinstance(row, list) and len(row) == len(mat[0]) for row in mat):
             raise ValueError("Input must be a non-empty list of lists with consistent row lengths.")

        self.mat = mat
        self._size = (len(self.mat), len(self.mat[0]))

    @property
    def size(self) -> tuple[int, int]:
        """Returns the size of the matrix as a tuple (rows, columns)."""
        return self._size

    def __str__(self) -> str:
        """Returns a string representation of the matrix."""
        return "\n" + "\n".join(str(x) for x in self.mat) + "\n"

    def __eq__(self, other) -> bool:
        """Checks if two matrices are equal."""
        if not isinstance(other, ABCMatrix) or self.size != other.size:
            return False

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self[i, j] != other[i, j]:
                    return False
        return True

    def __ne__(self, other) -> bool:
        """Checks if two matrices are not equal."""
        return not (self == other)

    def __getitem__(self, key: tuple[int, int] | int) -> Decimal | complex | float | list[Decimal | complex | float]:
        """Returns the matrix element by index (row, column) or the entire row."""
        if isinstance(key, tuple) and len(key) == 2:
           row, col = key
           if 0 <= row < self.size[0] and 0 <= col < self.size[1]:
               return self.mat[row][col]
           else:
               raise IndexError("Index out of matrix bounds.")
        elif isinstance(key, int):
           row = key
           if 0 <= row < self.size[0]:
               return self.mat[row]
           else:
               raise IndexError("Row index out of bounds.")
        else:
           raise TypeError("Index must be a tuple (row, col) or an integer (row).")

    def __neg__(self) -> 'BaseMatrix':
        """Returns the matrix with all elements negated."""
        new_mat = [[-element for element in row] for row in self.mat]
        return BaseMatrix(new_mat)

    def __add__(self, other) -> 'BaseMatrix':
        """Performs matrix addition."""
        if not isinstance(other, ABCMatrix) or self.size != other.size:
            raise ValueError("Matrices must have the same dimensions for addition.")

        new_mat = []
        for i in range(self.size[0]):
            row = []
            for j in range(self.size[1]):
                row.append(self[i, j] + other[i, j])
            new_mat.append(row)
        return BaseMatrix(new_mat)

    def __sub__(self, other) -> 'BaseMatrix':
        """Performs matrix subtraction."""
        if not isinstance(other, ABCMatrix) or self.size != other.size:
            raise ValueError("Matrices must have the same dimensions for subtraction.")

        new_mat = []
        for i in range(self.size[0]):
            row = []
            for j in range(self.size[1]):
                row.append(self[i, j] - other[i, j])
            new_mat.append(row)
        return BaseMatrix(new_mat)

    def __mul__(self, other) -> 'BaseMatrix':
        """Performs matrix multiplication by a scalar or another matrix."""
        if isinstance(other, (Decimal, complex, float, int)):
            new_mat = [[self[i, j] * other for j in range(self.size[1])] for i in range(self.size[0])]
            return BaseMatrix(new_mat)

        elif isinstance(other, ABCMatrix):
            if self.size[1] != other.size[0]:
                raise ValueError(f"Matrix dimensions {self.size} and {other.size} are incompatible for multiplication.")

            rows_self, cols_self = self.size
            rows_other, cols_other = other.size

            new_mat = []
            for i in range(rows_self):
                row = []
                for j in range(cols_other):
                    element = sum(self[i, k] * other[k, j] for k in range(cols_self))
                    row.append(element)
                new_mat.append(row)
            return BaseMatrix(new_mat)

        else:
            raise TypeError(f"Multiplication is only supported with a number or another matrix, not {type(other).__name__}")

    def __rmul__(self, other) -> 'BaseMatrix':
        """Performs scalar multiplication on the matrix (commutative)."""
        return self.__mul__(other)

    def get_matrix(self) -> list[list[Decimal | complex | float]]:
        """Returns the internal representation of the matrix (list of lists)."""
        return self.mat

    def augment(self, other: 'ABCMatrix') -> 'BaseMatrix':
        """Augments the matrix by another matrix on the right (column-wise concatenation)."""
        if not isinstance(other, ABCMatrix) or other.size[0] != self.size[0]:
            raise ValueError("Matrices must have the same number of rows for augmentation.")

        new_mat = [self.mat[i] + other.mat[i] for i in range(self.size[0])]
        return BaseMatrix(new_mat)

    def transpose(self) -> 'BaseMatrix':
        """Returns the transposed matrix."""
        rows, cols = self.size
        new_mat = []
        for j in range(cols):
            row = []
            for i in range(rows):
                row.append(self[i, j])
            new_mat.append(row)
        return BaseMatrix(new_mat)

    def determinant(self) -> Decimal | complex | float:
        """Calculates the determinant of a square matrix using Gaussian elimination."""
        if self.size[0] != self.size[1]:
            raise ValueError("Matrix must be square to calculate the determinant.")

        n = self.size[0]
        matrix_copy = copy.deepcopy(self.mat)
        det = 1

        for col in range(n):
            pivot_row = None
            for row in range(col, n):
                if abs(matrix_copy[row][col]) > 1e-9:
                    pivot_row = row
                    break

            if pivot_row is None:
                return 0

            if pivot_row != col:
                matrix_copy[col], matrix_copy[pivot_row] = matrix_copy[pivot_row], matrix_copy[col]
                det *= -1

            pivot_element = matrix_copy[col][col]
            det *= pivot_element

            if abs(pivot_element) < 1e-9:
                 return 0.0

            for row in range(col + 1, n):
                factor = matrix_copy[row][col] / pivot_element
                for c in range(col, n):
                    matrix_copy[row][c] -= factor * matrix_copy[col][c]

        return det


Matrix = BaseMatrix