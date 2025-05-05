from abc import ABC, abstractmethod
from decimal import Decimal, getcontext
import copy
import math

from abc import ABC, abstractmethod
from decimal import Decimal, getcontext
import copy


class ABCMatrix(ABC):
    """Abstract base class for matrices."""

    @abstractmethod
    def __init__(self, mat: list[list[None | Decimal | complex | float]]):
        pass

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        pass

    @property
    @abstractmethod
    def dtype(self) -> type:
        """Returns the primary data type of the matrix (Decimal, complex, float)."""
        pass

    @abstractmethod
    def to_list(self) -> list[list[Decimal | complex | float]]:
        """Returns the matrix as a standard list of lists."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __neg__(self) -> "ABCMatrix":
        pass

    @abstractmethod
    def __add__(self, other) -> "ABCMatrix":
        pass

    @abstractmethod
    def __sub__(self) -> "ABCMatrix":
        pass

    @abstractmethod
    def __mul__(self, other) -> "ABCMatrix":
        pass

    @abstractmethod
    def __rmul__(self, other) -> "ABCMatrix":
        pass

    @abstractmethod
    def __getitem__(
        self, key: tuple[int, int] | int
    ) -> Decimal | complex | float | list[Decimal | complex | float]:
        pass

    @abstractmethod
    def copy(self) -> "ABCMatrix":
        """Returns a deep copy of the matrix."""
        pass

    @abstractmethod
    def __setitem__(
        self, key: tuple[int, int], value: Decimal | complex | float
    ) -> None:
        """Sets the value at the specified (row, col) index."""
        pass

    @abstractmethod
    def get_row(self, row_index: int) -> list[Decimal | complex | float]:
        """Returns the specified row as a list."""
        pass

    @abstractmethod
    def get_loc(self, row_index: int, col_index: int) -> Decimal | complex | float:
        """Returns the element at the specified (row, col) index."""
        pass

    @abstractmethod
    def set_row(
        self, row_index: int, row_data: list[Decimal | complex | float]
    ) -> None:
        """Sets the entire row at the specified index."""
        pass

    @abstractmethod
    def set_loc(
        self, row_index: int, col_index: int, value: Decimal | complex | float
    ) -> None:
        """Sets the element at the specified (row, col) index."""
        pass

    @abstractmethod
    def augment(self, other: "ABCMatrix") -> "ABCMatrix":
        pass

    @abstractmethod
    def transpose(self) -> "ABCMatrix":
        pass

    @abstractmethod
    def determinant(self) -> Decimal | complex | float:
        pass
    
    def handle_missing_values(self, mat: list[list[None| Decimal | complex | float]]):
        pass

class BaseMatrix(ABCMatrix):
    """Implementation of a matrix using a list of lists."""

    def __init__(self, mat: list[list[None | Decimal | complex | float]]):
        if not mat or not isinstance(mat[0], list):
            raise ValueError("Input must be a non-empty list of lists.")
        if not mat[0]:
            raise ValueError("Input rows cannot be empty.")

        num_cols = len(mat[0])
        if not all(isinstance(row, list) and len(row) == num_cols for row in mat):
            raise ValueError(
                "Input must be a list of lists with consistent row lengths."
            )

        self._mat = []
        self._dtype = Decimal

        # Detect type and convert elements
        for r_idx, row in enumerate(mat):
            new_row = []
            for c_idx, val in enumerate(row):
                try:
                    if isinstance(val, complex):
                        current_val = complex(val)
                        self._dtype = complex
                    elif isinstance(val, float) and self._dtype is not complex:
                        current_val = Decimal(str(val))
                        if self._dtype is Decimal:
                            self._dtype = float
                    elif (
                        isinstance(val, int)
                        and not isinstance(val, bool)
                        and self._dtype is not complex
                    ):
                        current_val = Decimal(val)
                    else:
                        current_val = val
                        if not isinstance(
                            current_val, (Decimal, complex, float, int, bool)
                        ):
                            raise TypeError(
                                f"Unsupported type at row {r_idx}, col {c_idx}: {type(val)}"
                            )

                    new_row.append(current_val)

                except Exception as e:
                    raise TypeError(
                        f"Error processing value '{val}' at row {r_idx}, col {c_idx}: {e}"
                    ) from e
            self._mat.append(new_row)

        # Final pass to ensure consistency if complex was detected late
        if self._dtype is complex:
            for r in range(len(self._mat)):
                for c in range(len(self._mat[r])):
                    if not isinstance(self._mat[r][c], complex):
                        try:
                            self._mat[r][c] = complex(self._mat[r][c])
                        except TypeError:
                            # Handle cases like trying complex(Decimal(...)) if needed
                            # Usually float(Decimal(...)) works before complex()
                            try:
                                self._mat[r][c] = complex(float(self._mat[r][c]))
                            except Exception as e_conv:
                                raise TypeError(
                                    f"Cannot convert element at ({r},{c}) to complex. Original value: {mat[r][c]}. Error: {e_conv}"
                                )

        self._size = (len(self._mat), num_cols)

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def dtype(self) -> type:
        return self._dtype

    def to_list(self) -> list[list[Decimal | complex | float]]:
        """Returns a deep copy of the matrix as a standard list of lists."""
        return copy.deepcopy(self._mat)

    def __str__(self) -> str:
        """Returns a string representation of the matrix."""
        if not self._mat:
            return "[]"
        mat_to_print = self._mat
        rows, cols = self.size

        col_widths = [0] * cols
        for j in range(cols):
            for i in range(rows):
                # Format complex numbers nicely
                element_str = str(mat_to_print[i][j])
                if isinstance(mat_to_print[i][j], complex):
                    element_str = (
                        f"{mat_to_print[i][j].real:.5g}{mat_to_print[i][j].imag:+.5g}j"
                    )
                elif isinstance(mat_to_print[i][j], Decimal):
                    # Adjust formatting for Decimals if needed
                    element_str = f"{mat_to_print[i][j]:.5g}"
                elif isinstance(mat_to_print[i][j], float):
                    element_str = f"{mat_to_print[i][j]:.5g}"

                col_widths[j] = max(col_widths[j], len(element_str))

        s = "[\n"
        for i in range(rows):
            row_str = " ["
            for j in range(cols):
                element_str = str(mat_to_print[i][j])
                if isinstance(mat_to_print[i][j], complex):
                    element_str = (
                        f"{mat_to_print[i][j].real:.5g}{mat_to_print[i][j].imag:+.5g}j"
                    )
                elif isinstance(mat_to_print[i][j], Decimal):
                    element_str = f"{mat_to_print[i][j]:.5g}"
                elif isinstance(mat_to_print[i][j], float):
                    element_str = f"{mat_to_print[i][j]:.5g}"

                row_str += element_str.rjust(col_widths[j]) + (
                    " " if j < cols - 1 else ""
                )
            row_str += "]"
            s += row_str + ("\n" if i < rows - 1 else "")
        s += "\n]"
        return s

    def __eq__(self, other) -> bool:
        """Checks if two matrices are equal element-wise."""
        if not isinstance(other, ABCMatrix) or self.size != other.size:
            return False

        rows, cols = self.size
        # Define a tolerance for float/complex/Decimal comparison
        tolerance = (
            Decimal("1e-9") if self.dtype is Decimal or other.dtype is Decimal else 1e-9
        )

        for i in range(rows):
            for j in range(cols):
                val_self = self[i, j]
                val_other = other[i, j]

                try:
                    # Attempt comparison with tolerance
                    if isinstance(val_self, complex) or isinstance(val_other, complex):
                        if abs(complex(val_self) - complex(val_other)) > tolerance:
                            return False
                    elif isinstance(val_self, Decimal) or isinstance(
                        val_other, Decimal
                    ):
                        # Convert both to Decimal for comparison if one is Decimal
                        if (
                            abs(Decimal(str(val_self)) - Decimal(str(val_other)))
                            > tolerance
                        ):
                            return False
                    else:  # Assume float or int
                        if abs(float(val_self) - float(val_other)) > tolerance:
                            return False
                except (TypeError, ValueError):
                    # If types are fundamentally incompatible for comparison
                    return False
        return True

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __getitem__(
        self, key: tuple[int, int] | int
    ) -> Decimal | complex | float | list[Decimal | complex | float]:
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if 0 <= row < self.size[0] and 0 <= col < self.size[1]:
                return self._mat[row][col]
            else:
                raise IndexError(
                    f"Index ({row}, {col}) out of matrix bounds {self.size}."
                )
        elif isinstance(key, int):
            row = key
            if 0 <= row < self.size[0]:
                return list(self._mat[row])  # Return a copy
            else:
                raise IndexError(
                    f"Row index {row} out of bounds for size {self.size[0]}."
                )
        else:
            raise TypeError(
                f"Index must be a tuple (row, col) or an integer (row), got {type(key)}."
            )

    # Added methods
    def copy(self) -> "BaseMatrix":
        """Returns a deep copy of the matrix."""
        return BaseMatrix(copy.deepcopy(self._mat))

    def __setitem__(
        self, key: tuple[int, int], value: Decimal | complex | float
    ) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Index must be a tuple (row, col), got {type(key)}.")
        row, col = key
        rows, cols = self.size
        if not (0 <= row < rows and 0 <= col < cols):
            raise IndexError(f"Index ({row}, {col}) out of matrix bounds {self.size}.")
        # Basic type validation (can be more strict if needed)
        if not isinstance(value, (Decimal, complex, float, int)):
            raise TypeError(f"Unsupported value type: {type(value)}")

        # Convert value to matrix's dtype if possible
        try:
            if self.dtype is complex:
                self._mat[row][col] = complex(value)
            elif self.dtype is float:
                self._mat[row][col] = float(value)
            elif self.dtype is Decimal:
                self._mat[row][col] = Decimal(str(value))
            else:
                # Fallback or stricter type checking
                self._mat[row][col] = value
        except (TypeError, ValueError):
            raise TypeError(
                f"Cannot convert value {value} to matrix dtype {self.dtype}"
            )

    def get_row(self, row_index: int) -> list[Decimal | complex | float]:
        """Returns the specified row as a list."""
        return self[row_index]  # Delegate to __getitem__

    def get_loc(self, row_index: int, col_index: int) -> Decimal | complex | float:
        """Returns the element at the specified (row, col) index."""
        return self[row_index, col_index]  # Delegate to __getitem__
        # TODO remake

    def set_row(
        self, row_index: int, row_data: list[Decimal | complex | float]
    ) -> None:
        rows, cols = self.size
        if not (0 <= row_index < rows):
            raise IndexError(f"Row index {row_index} out of bounds for size {rows}.")
        if not isinstance(row_data, list) or len(row_data) != cols:
            raise ValueError(
                f"Row data must be a list of length {cols}, got {len(row_data)}."
            )
        # Basic type validation for elements (can be more strict)
        if not all(isinstance(val, (Decimal, complex, float, int)) for val in row_data):
            raise TypeError("Row data contains unsupported types.")

        # Convert row data elements to matrix's dtype if possible
        try:
            if self.dtype is complex:
                self._mat[row_index] = [complex(val) for val in row_data]
            elif self.dtype is float:
                self._mat[row_index] = [float(val) for val in row_data]
            elif self.dtype is Decimal:
                self._mat[row_index] = [Decimal(str(val)) for val in row_data]
            else:
                # Fallback
                self._mat[row_index] = list(row_data)  # Ensure it's a new list

        except (TypeError, ValueError):
            raise TypeError(
                f"Cannot convert elements in row data to matrix dtype {self.dtype}"
            )

    def set_loc(
        self, row_index: int, col_index: int, value: Decimal | complex | float
    ) -> None:
        """Sets the element at the specified (row, col) index."""
        self[row_index, col_index] = value  # Delegate to __setitem__

    def __neg__(self) -> "BaseMatrix":
        new_mat = [[-element for element in row] for row in self._mat]
        return BaseMatrix(new_mat)

    def _check_compatibility(self, other, operation: str):
        if not isinstance(other, ABCMatrix):
            raise TypeError(
                f"Unsupported operand type(s) for {operation}: '{type(self).__name__}' and '{type(other).__name__}'"
            )
        if self.size != other.size:
            raise ValueError(
                f"Matrices must have the same dimensions for {operation}. Sizes are {self.size} and {other.size}"
            )

    def __add__(self, other) -> "BaseMatrix":
        self._check_compatibility(other, "addition")
        res_dtype = (
            complex
            if self.dtype is complex or other.dtype is complex
            else float
            if self.dtype is float or other.dtype is float
            else Decimal
        )
        new_mat = [
            [
                res_dtype(self[i, j]) + res_dtype(other[i, j])
                for j in range(self.size[1])
            ]
            for i in range(self.size[0])
        ]
        return BaseMatrix(new_mat)

    def __sub__(self, other) -> "BaseMatrix":
        self._check_compatibility(other, "subtraction")
        res_dtype = (
            complex
            if self.dtype is complex or other.dtype is complex
            else float
            if self.dtype is float or other.dtype is float
            else Decimal
        )
        new_mat = [
            [
                res_dtype(self[i, j]) - res_dtype(other[i, j])
                for j in range(self.size[1])
            ]
            for i in range(self.size[0])
        ]
        return BaseMatrix(new_mat)

    def __mul__(self, other) -> "BaseMatrix":
        if isinstance(other, (Decimal, complex, float, int)):
            scalar = other
            res_dtype = self.dtype
            if isinstance(scalar, complex):
                res_dtype = complex
                scalar = complex(scalar)
            elif isinstance(scalar, (float, Decimal)) and res_dtype is Decimal:
                res_dtype = Decimal
                scalar = Decimal(scalar)
            elif not isinstance(scalar, (complex, float, Decimal)):
                try:
                    scalar = Decimal(scalar)
                except:
                    scalar = float(scalar)
                    if res_dtype is Decimal:
                        res_dtype = float

            scalar = res_dtype(scalar) if res_dtype is not complex else complex(scalar)

            new_mat = [
                [self[i, j] * scalar for j in range(self.size[1])]
                for i in range(self.size[0])
            ]
            return BaseMatrix(new_mat)

        elif isinstance(other, ABCMatrix):
            if self.size[1] != other.size[0]:
                raise ValueError(
                    f"Matrix dimensions {self.size} and {other.size} are incompatible for multiplication."
                )

            rows_self, cols_self = self.size
            rows_other, cols_other = other.size

            res_dtype = (
                complex
                if self.dtype is complex or other.dtype is complex
                else float
                if self.dtype is float or other.dtype is float
                else Decimal
            )
            zero_val = res_dtype(0)

            new_mat_data = [[zero_val] * cols_other for _ in range(rows_self)]

            for i in range(rows_self):
                for j in range(cols_other):
                    sum_val = zero_val
                    for k in range(cols_self):
                        term1 = (
                            res_dtype(self[i, k])
                            if res_dtype is not complex
                            else complex(self[i, k])
                        )
                        term2 = (
                            res_dtype(other[k, j])
                            if res_dtype is not complex
                            else complex(other[k, j])
                        )
                        sum_val += term1 * term2
                    new_mat_data[i][j] = sum_val
            return BaseMatrix(new_mat_data)

        else:
            return NotImplemented

    def __rmul__(self, other) -> "BaseMatrix":
        if isinstance(other, (Decimal, complex, float, int)):
            return self.__mul__(other)
        else:
            return NotImplemented

    def augment(self, other: "ABCMatrix") -> "BaseMatrix":
        if not isinstance(other, ABCMatrix):
            raise TypeError(
                f"Can only augment with another matrix, not {type(other).__name__}"
            )
        if other.size[0] != self.size[0]:
            raise ValueError(
                f"Matrices must have the same number of rows for augmentation ({self.size[0]} != {other.size[0]})."
            )

        # Get dense representations using the public method
        mat_self = self.to_list()
        mat_other = other.to_list()

        new_mat = [mat_self[i] + mat_other[i] for i in range(self.size[0])]
        return BaseMatrix(new_mat)

    def transpose(self) -> "BaseMatrix":
        rows, cols = self.size
        # Initialize with zeros of the matrix's type
        new_mat = [[self.dtype(0)] * rows for _ in range(cols)]
        for j in range(cols):
            for i in range(rows):
                new_mat[j][i] = self[i, j]
        return BaseMatrix(new_mat)

    def determinant(self) -> Decimal | complex | float:
        """Calculates the determinant using Gaussian elimination with pivoting."""
        if self.size[0] != self.size[1]:
            raise ValueError("Matrix must be square to calculate the determinant.")

        n = self.size[0]
        # Work on a copy, ensuring elements match self.dtype
        matrix_copy = [[self.dtype(el) for el in row] for row in self._mat]
        det = self.dtype(1)
        pivot_swaps = 0
        zero_threshold = (
            Decimal("1e-12")
            if self.dtype is Decimal
            else (1e-12 if self.dtype is float else 1e-12j).real
        )  # Use real part for complex threshold

        for col in range(n):
            pivot_row = col
            # Use magnitude for complex numbers, abs for others
            max_abs_val = abs(matrix_copy[col][col])

            for row in range(col + 1, n):
                current_abs_val = abs(matrix_copy[row][col])
                if current_abs_val > max_abs_val:
                    max_abs_val = current_abs_val
                    pivot_row = row

            if max_abs_val < zero_threshold:
                return self.dtype(0)  # Matrix is singular

            if pivot_row != col:
                matrix_copy[col], matrix_copy[pivot_row] = (
                    matrix_copy[pivot_row],
                    matrix_copy[col],
                )
                pivot_swaps += 1

            pivot_element = matrix_copy[col][col]

            for row in range(col + 1, n):
                val_to_elim = matrix_copy[row][col]
                if self.dtype is complex:
                    factor = complex(val_to_elim) / complex(pivot_element)
                elif self.dtype is Decimal:
                    factor = Decimal(str(val_to_elim)) / Decimal(str(pivot_element))
                else:
                    factor = float(val_to_elim) / float(pivot_element)

                matrix_copy[row][col] = self.dtype(0)
                for c in range(col + 1, n):
                    term_to_subtract = factor * matrix_copy[col][c]
                    matrix_copy[row][c] = matrix_copy[row][c] - term_to_subtract

        for i in range(n):
            det *= matrix_copy[i][i]

        if pivot_swaps % 2 != 0:
            det *= -1

        if abs(det) < zero_threshold:
            return self.dtype(0)

        return det
    
    def handle_missing_values(self, mat: list[list[None | Decimal | complex | float]]) -> list[list[Decimal | complex | float]]:
        rows, cols = (len(mat), len(mat[0]))
        filled_mat_data = copy.deepcopy(mat)
        for j in range(cols):
            col_sum = 0.0
            non_nan_count = 0
            column_has_nan = False 
            for i in range(rows):
                try:
                    value = float(mat[i, j])
                    if not math.isnan(value):
                        col_sum += value
                        non_nan_count += 1
                    else:
                        column_has_nan = True
                except (TypeError, ValueError):
                    pass
            col_mean = 0.0
            if non_nan_count > 0:
                col_mean = col_sum / non_nan_count
            if column_has_nan:
                for i in range(rows):
                    try:
                        value = float(filled_mat_data[i][j])
                        if math.isnan(value):
                            filled_mat_data[i][j] = type(mat[i,j])(col_mean) if not isinstance(mat[i,j], complex) else float(col_mean) # Попытка сохранить тип данных

                    except (TypeError, ValueError):
                        pass
        return filled_mat_data



class CSRMatrix(ABCMatrix):
    """Sparse matrix using Compressed Sparse Row (CSR) format."""

    def __init__(self, mat: list[list[Decimal | complex | float]]):
        if not mat or not isinstance(mat[0], list):
            raise ValueError("Input must be a non-empty list of lists.")
        if not mat[0]:
            raise ValueError("Input rows cannot be empty.")

        rows = len(mat)
        cols = len(mat[0])
        if not all(isinstance(row, list) and len(row) == cols for row in mat):
            raise ValueError(
                "Input must be a list of lists with consistent row lengths."
            )

        self._size = (rows, cols)
        self.data = []
        self.indices = []
        self.indptr = [0] * (rows + 1)
        self._dtype = Decimal
        nnz = 0
        zero_threshold = 1e-12  # General threshold

        temp_dtype = Decimal

        for r in range(rows):
            for c in range(cols):
                val = mat[r][c]
                current_val = val

                # Type detection and initial conversion (similar to BaseMatrix)
                if isinstance(val, complex):
                    current_val = complex(val)
                    if temp_dtype is not complex:
                        temp_dtype = complex
                elif isinstance(val, float):
                    current_val = Decimal(str(val))  # Try Decimal first
                    if temp_dtype is Decimal:
                        temp_dtype = float
                elif isinstance(val, int) and not isinstance(val, bool):
                    current_val = Decimal(val)

                # Check if non-zero using appropriate comparison
                is_non_zero = False
                if isinstance(current_val, complex):
                    if abs(current_val) > zero_threshold:
                        is_non_zero = True
                elif isinstance(current_val, (Decimal, float)):
                    if abs(current_val) > Decimal(str(zero_threshold)):
                        is_non_zero = True
                elif current_val != 0:
                    is_non_zero = True

                if is_non_zero:
                    self.data.append(
                        current_val
                    )  # Add potentially mixed types initially
                    self.indices.append(c)
                    nnz += 1
            self.indptr[r + 1] = nnz

        # Finalize dtype and ensure data consistency
        self._dtype = temp_dtype
        if self._dtype is complex:
            self.data = [complex(d) for d in self.data]
        elif self._dtype is float:
            self.data = [float(d) if isinstance(d, Decimal) else d for d in self.data]

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    @property
    def dtype(self) -> type:
        return self._dtype

    def _to_dense(self) -> list[list[Decimal | complex | float]]:
        """Helper to convert CSR to dense list of lists."""
        rows, cols = self.size
        dense_mat = [[self.dtype(0)] * cols for _ in range(rows)]
        for r in range(rows):
            for i in range(self.indptr[r], self.indptr[r + 1]):
                dense_mat[r][self.indices[i]] = self.data[i]
        return dense_mat

    def to_list(self) -> list[list[Decimal | complex | float]]:
        """Returns the matrix as a standard list of lists."""
        return self._to_dense()

    def __str__(self) -> str:
        # Reuse BaseMatrix's __str__ logic for consistent formatting
        # Create a temporary BaseMatrix instance for formatting purposes
        temp_base = BaseMatrix(self.to_list())
        return temp_base.__str__()

    def __eq__(self, other) -> bool:
        # Delegate to BaseMatrix's __eq__ for robust element-wise comparison
        return BaseMatrix.__eq__(self, other)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __getitem__(
        self, key: tuple[int, int] | int
    ) -> Decimal | complex | float | list[Decimal | complex | float]:
        rows, cols = self.size
        zero_val = self.dtype(0)

        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if not (0 <= row < rows and 0 <= col < cols):
                raise IndexError(
                    f"Index ({row}, {col}) out of matrix bounds {self.size}."
                )

            row_start = self.indptr[row]
            row_end = self.indptr[row + 1]

            # Simple linear search for index
            for i in range(row_start, row_end):
                if self.indices[i] == col:
                    return self.data[i]
            return zero_val  # Not found -> zero

        elif isinstance(key, int):
            row = key
            if not (0 <= row < rows):
                raise IndexError(f"Row index {row} out of bounds for size {rows}.")

            dense_row = [zero_val] * cols
            row_start = self.indptr[row]
            row_end = self.indptr[row + 1]
            for i in range(row_start, row_end):
                dense_row[self.indices[i]] = self.data[i]
            return dense_row
        else:
            raise TypeError(
                f"Index must be a tuple (row, col) or an integer (row), got {type(key)}."
            )

    def copy(self) -> "CSRMatrix":
        """Returns a copy of the matrix."""
        cls = type(self)
        new_csr = cls.__new__(cls)
        new_csr.data = list(self.data)
        new_csr.indices = list(self.indices)
        new_csr.indptr = list(self.indptr)
        new_csr._size = self._size
        new_csr._dtype = self._dtype
        return new_csr

    def __setitem__(
        self, key: tuple[int, int], value: Decimal | complex | float
    ) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Index must be a tuple (row, col), got {type(key)}.")
        row, col = key
        rows, cols = self.size
        if not (0 <= row < rows and 0 <= col < cols):
            raise IndexError(f"Index ({row}, {col}) out of matrix bounds {self.size}.")

        # Basic value type validation
        if not isinstance(value, (Decimal, complex, float, int)):
            raise TypeError(f"Unsupported value type: {type(value)}")

        # Convert value to matrix's dtype if possible
        try:
            if self.dtype is complex:
                typed_value = complex(value)
            elif self.dtype is float:
                typed_value = float(value)
            elif self.dtype is Decimal:
                typed_value = Decimal(str(value))
            else:
                typed_value = value
        except (TypeError, ValueError):
            raise TypeError(
                f"Cannot convert value {value} to matrix dtype {self.dtype}"
            )

        row_start = self.indptr[row]
        row_end = self.indptr[row + 1]

        # Find if element exists at (row, col)
        found_idx = -1
        for i in range(row_start, row_end):
            if self.indices[i] == col:
                found_idx = i
                break

        is_zero = False
        # Use appropriate comparison for zero check
        zero_threshold = (
            Decimal("1e-12")
            if self.dtype is Decimal
            else (1e-12 if self.dtype is float else 1e-12j).real
        )
        if isinstance(typed_value, complex):
            if abs(typed_value) < zero_threshold:
                is_zero = True
        elif isinstance(typed_value, (Decimal, float)):
            if abs(typed_value) < Decimal(str(zero_threshold)):
                is_zero = True
        elif typed_value == 0:
            is_zero = True

        if found_idx != -1:  # Element exists
            if is_zero:
                # Remove element
                del self.data[found_idx]
                del self.indices[found_idx]
                # Decrement indptr for all subsequent rows
                for r_idx in range(row + 1, rows + 1):
                    self.indptr[r_idx] -= 1
            else:
                # Update element
                self.data[found_idx] = typed_value
        else:  # Element does not exist
            if not is_zero:
                # Find insertion point to maintain sorted indices
                insert_idx = row_start
                while insert_idx < row_end and self.indices[insert_idx] < col:
                    insert_idx += 1

                # Insert element
                self.data.insert(insert_idx, typed_value)
                self.indices.insert(insert_idx, col)
                # Increment indptr for all subsequent rows
                for r_idx in range(row + 1, rows + 1):
                    self.indptr[r_idx] += 1

    def get_row(self, row_index: int) -> list[Decimal | complex | float]:
        """Returns the specified row as a list."""
        return self[row_index]  # Delegate to __getitem__

    def get_loc(self, row_index: int, col_index: int) -> Decimal | complex | float:
        """Returns the element at the specified (row, col) index."""
        return self[row_index, col_index]  # Delegate to __getitem__

    def set_row(
        self, row_index: int, row_data: list[Decimal | complex | float]
    ) -> None:
        rows, cols = self.size
        if not (0 <= row_index < rows):
            raise IndexError(f"Row index {row_index} out of bounds for size {rows}.")
        if not isinstance(row_data, list) or len(row_data) != cols:
            raise ValueError(
                f"Row data must be a list of length {cols}, got {len(row_data)}."
            )
        # Basic type validation for elements
        if not all(isinstance(val, (Decimal, complex, float, int)) for val in row_data):
            raise TypeError("Row data contains unsupported types.")

        # --- Inefficient but simple implementation via dense conversion ---
        # Convert to dense, update the row, convert back to CSR
        dense_mat = self.to_list()

        # Convert row data elements to matrix's dtype if possible
        try:
            if self.dtype is complex:
                dense_mat[row_index] = [complex(val) for val in row_data]
            elif self.dtype is float:
                dense_mat[row_index] = [float(val) for val in row_data]
            elif self.dtype is Decimal:
                dense_mat[row_index] = [Decimal(str(val)) for val in row_data]
            else:
                dense_mat[row_index] = list(row_data)  # Ensure it's a new list
        except (TypeError, ValueError):
            raise TypeError(
                f"Cannot convert elements in row data to matrix dtype {self.dtype}"
            )

        # Rebuild CSR from the updated dense matrix
        # This is inefficient for large sparse matrices, but simple.
        new_csr = CSRMatrix(dense_mat)
        self.data = new_csr.data
        self.indices = new_csr.indices
        self.indptr = new_csr.indptr
        self._dtype = (
            new_csr.dtype
        )  # Dtype might change if row_data introduces new types

        # --- End of inefficient implementation ---

    def set_loc(
        self, row_index: int, col_index: int, value: Decimal | complex | float
    ) -> None:
        """Sets the element at the specified (row, col) index."""
        self[row_index, col_index] = value  # Delegate to __setitem__

    def __neg__(self) -> "CSRMatrix":
        """Returns the matrix with all non-zero elements negated."""
        cls = type(self)
        new_csr = cls.__new__(cls)
        new_csr.data = [-d for d in self.data]
        new_csr.indices = list(self.indices)
        new_csr.indptr = list(self.indptr)
        new_csr._size = self.size
        new_csr._dtype = self.dtype
        return new_csr

    def __add__(self, other) -> "CSRMatrix":
        """Performs matrix addition. Result is CSR."""
        if not isinstance(other, ABCMatrix):
            return NotImplemented
        if self.size != other.size:
            raise ValueError(
                f"Matrices must have the same dimensions for addition. Sizes are {self.size} and {other.size}"
            )

        res_dtype = (
            complex
            if self.dtype is complex or other.dtype is complex
            else float
            if self.dtype is float or other.dtype is float
            else Decimal
        )
        new_dense = [
            [
                res_dtype(self[i, j]) + res_dtype(other[i, j])
                for j in range(self.size[1])
            ]
            for i in range(self.size[0])
        ]
        return CSRMatrix(new_dense)

    def __sub__(self, other) -> "CSRMatrix":
        """Performs matrix subtraction. Result is CSR."""
        if not isinstance(other, ABCMatrix):
            return NotImplemented
        if self.size != other.size:
            raise ValueError(
                f"Matrices must have the same dimensions for subtraction. Sizes are {self.size} and {other.size}"
            )

        res_dtype = (
            complex
            if self.dtype is complex or other.dtype is complex
            else float
            if self.dtype is float or other.dtype is float
            else Decimal
        )
        new_dense = [
            [
                res_dtype(self[i, j]) - res_dtype(other[i, j])
                for j in range(self.size[1])
            ]
            for i in range(self.size[0])
        ]
        return CSRMatrix(new_dense)

    def __mul__(self, other) -> "CSRMatrix":
        """Performs multiplication by a scalar or another matrix."""
        cls = type(self)
        if isinstance(other, (Decimal, complex, float, int)):
            scalar = other
            new_dtype = self.dtype
            # Determine result type based on scalar
            if isinstance(scalar, complex):
                new_dtype = complex
                scalar = complex(scalar)
            elif isinstance(scalar, float):
                scalar = float(scalar)
                if new_dtype is Decimal:
                    new_dtype = float
            elif not isinstance(scalar, Decimal):
                try:
                    scalar = Decimal(scalar)
                except:
                    scalar = float(scalar)
                    if new_dtype is Decimal:
                        new_dtype = float

            scalar = new_dtype(scalar) if new_dtype is not complex else complex(scalar)

            new_csr = cls.__new__(cls)
            new_csr.data = [new_dtype(d) * scalar for d in self.data]
            new_csr.indices = list(self.indices)
            new_csr.indptr = list(self.indptr)
            new_csr._size = self.size
            new_csr._dtype = new_dtype
            return new_csr

        elif isinstance(other, ABCMatrix):
            # Matrix multiplication (dense intermediate for simplicity)
            if self.size[1] != other.size[0]:
                raise ValueError(
                    f"Matrix dimensions {self.size} and {other.size} are incompatible for multiplication."
                )

            rows_self, cols_self = self.size
            rows_other, cols_other = other.size
            res_dtype = (
                complex
                if self.dtype is complex or other.dtype is complex
                else float
                if self.dtype is float or other.dtype is float
                else Decimal
            )
            zero_val = res_dtype(0)
            new_mat_data = [[zero_val] * cols_other for _ in range(rows_self)]

            for i in range(rows_self):
                row_start_self = self.indptr[i]
                row_end_self = self.indptr[i + 1]
                if row_start_self == row_end_self:
                    continue  # Skip empty rows

                for k_idx in range(row_start_self, row_end_self):
                    k = self.indices[k_idx]
                    val_self = self.data[k_idx]

                    for j in range(cols_other):
                        term2 = other[k, j]
                        t1 = (
                            res_dtype(val_self)
                            if res_dtype is not complex
                            else complex(val_self)
                        )
                        t2 = (
                            res_dtype(term2)
                            if res_dtype is not complex
                            else complex(term2)
                        )
                        new_mat_data[i][j] += t1 * t2

            return CSRMatrix(new_mat_data)
        else:
            return NotImplemented

    def __rmul__(self, other) -> "CSRMatrix":
        if isinstance(other, (Decimal, complex, float, int)):
            return self.__mul__(other)
        else:
            return NotImplemented

    def augment(self, other: "ABCMatrix") -> "CSRMatrix":
        """Augments the matrix by another matrix on the right. Returns CSR."""
        if not isinstance(other, ABCMatrix):
            raise TypeError(
                f"Can only augment with another matrix, not {type(other).__name__}"
            )
        if other.size[0] != self.size[0]:
            raise ValueError(
                f"Matrices must have the same number of rows for augmentation ({self.size[0]} != {other.size[0]})."
            )

        # Convert both to dense, augment, convert back to CSR
        dense_self = self.to_list()
        dense_other = other.to_list()

        new_dense = [dense_self[i] + dense_other[i] for i in range(self.size[0])]
        return CSRMatrix(new_dense)

    def transpose(self) -> "CSRMatrix":
        """Returns the transposed matrix. Returns CSR."""
        # Convert to dense, transpose, convert back (inefficient but correct)
        dense_self = self.to_list()
        rows, cols = self.size
        new_dense = [[self.dtype(0)] * rows for _ in range(cols)]
        for j in range(cols):
            for i in range(rows):
                new_dense[j][i] = dense_self[i][j]
        return CSRMatrix(new_dense)

    def determinant(self) -> Decimal | complex | float:
        """Calculates the determinant. Converts to dense first."""
        if self.size[0] != self.size[1]:
            raise ValueError("Matrix must be square to calculate the determinant.")

        temp_base_mat = BaseMatrix(self.to_list())
        return temp_base_mat.determinant()

    def handle_missing_values(self, mat: list[list[None | Decimal | complex | float]]) -> list[list[Decimal | complex | float]]:
        rows, cols = (len(mat), len(mat[0]))
        filled_mat_data = copy.deepcopy(mat)
        for j in range(cols):
            col_sum = 0.0
            non_nan_count = 0
            column_has_nan = False 
            for i in range(rows):
                try:
                    value = float(mat[i, j])
                    if not math.isnan(value):
                        col_sum += value
                        non_nan_count += 1
                    else:
                        column_has_nan = True
                except (TypeError, ValueError):
                    pass
            col_mean = 0.0
            if non_nan_count > 0:
                col_mean = col_sum / non_nan_count
            if column_has_nan:
                for i in range(rows):
                    try:
                        value = float(filled_mat_data[i][j])
                        if math.isnan(value):
                            filled_mat_data[i][j] = type(mat[i,j])(col_mean) if not isinstance(mat[i,j], complex) else float(col_mean) # Попытка сохранить тип данных

                    except (TypeError, ValueError):
                        pass
        return filled_mat_data