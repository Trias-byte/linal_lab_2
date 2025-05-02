from abc import ABC, abstractmethod
from decimal import Decimal, getcontext
import copy

# Set Decimal precision if needed (optional)
# getcontext().prec = 28

class ABCMatrix(ABC):
    """Abstract base class for matrices."""

    @abstractmethod
    def __init__(self, mat: list[list[Decimal | complex | float]]):
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
    def __neg__(self) -> 'ABCMatrix':
        pass

    @abstractmethod
    def __add__(self, other) -> 'ABCMatrix':
        pass

    @abstractmethod
    def __sub__(self, other) -> 'ABCMatrix':
        pass

    @abstractmethod
    def __mul__(self, other) -> 'ABCMatrix':
        pass

    @abstractmethod
    def __rmul__(self, other) -> 'ABCMatrix':
        pass

    @abstractmethod
    def __getitem__(self, key: tuple[int, int] | int) -> Decimal | complex | float | list[Decimal | complex | float]:
        pass

    @abstractmethod
    def augment(self, other: 'ABCMatrix') -> 'ABCMatrix':
        pass

    @abstractmethod
    def transpose(self) -> 'ABCMatrix':
        pass

    @abstractmethod
    def determinant(self) -> Decimal | complex | float:
        pass


class BaseMatrix(ABCMatrix):
    """Implementation of a matrix using a list of lists."""

    def __init__(self, mat: list[list[Decimal | complex | float]]):
        if not mat or not isinstance(mat[0], list): # Check if mat is list of lists and non-empty
             raise ValueError("Input must be a non-empty list of lists.")
        if not mat[0]: # Check if first row is non-empty
             raise ValueError("Input rows cannot be empty.")

        num_cols = len(mat[0])
        if not all(isinstance(row, list) and len(row) == num_cols for row in mat):
             raise ValueError("Input must be a list of lists with consistent row lengths.")

        self._mat = [] # Use _mat internally to avoid potential conflicts
        self._dtype = Decimal # Start with Decimal preference

        # Detect type and convert elements
        for r_idx, row in enumerate(mat):
            new_row = []
            for c_idx, val in enumerate(row):
                try:
                    if isinstance(val, complex):
                        current_val = complex(val)
                        self._dtype = complex # Complex overrides other types
                    elif isinstance(val, float) and self._dtype is not complex:
                        current_val = Decimal(str(val)) # Try Decimal first for floats
                        if self._dtype is Decimal: # Only downgrade if currently Decimal
                             self._dtype = float # Found a float, downgrade preference if not complex
                    elif isinstance(val, int) and not isinstance(val, bool) and self._dtype is not complex:
                         current_val = Decimal(val) # Convert ints to Decimal if not complex
                    else:
                         current_val = val # Keep as is (e.g., existing Decimal, bool)
                         if not isinstance(current_val, (Decimal, complex, float, int, bool)):
                             raise TypeError(f"Unsupported type at row {r_idx}, col {c_idx}: {type(val)}")

                    new_row.append(current_val)

                except Exception as e:
                     raise TypeError(f"Error processing value '{val}' at row {r_idx}, col {c_idx}: {e}") from e
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
                                raise TypeError(f"Cannot convert element at ({r},{c}) to complex. Original value: {mat[r][c]}. Error: {e_conv}")


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
        if not self._mat: return "[]"
        # Use the matrix data directly
        mat_to_print = self._mat
        rows, cols = self.size

        col_widths = [0] * cols
        for j in range(cols):
            for i in range(rows):
                 # Format complex numbers nicely
                 element_str = str(mat_to_print[i][j])
                 if isinstance(mat_to_print[i][j], complex):
                     element_str = f"{mat_to_print[i][j].real:.5g}{mat_to_print[i][j].imag:+.5g}j"
                 elif isinstance(mat_to_print[i][j], Decimal):
                      # Adjust formatting for Decimals if needed
                      element_str = f"{mat_to_print[i][j]:.5g}" # Example: limit significant digits
                 elif isinstance(mat_to_print[i][j], float):
                      element_str = f"{mat_to_print[i][j]:.5g}"

                 col_widths[j] = max(col_widths[j], len(element_str))

        s = "[\n"
        for i in range(rows):
            row_str = " ["
            for j in range(cols):
                element_str = str(mat_to_print[i][j])
                # Apply formatting again for alignment
                if isinstance(mat_to_print[i][j], complex):
                     element_str = f"{mat_to_print[i][j].real:.5g}{mat_to_print[i][j].imag:+.5g}j"
                elif isinstance(mat_to_print[i][j], Decimal):
                     element_str = f"{mat_to_print[i][j]:.5g}"
                elif isinstance(mat_to_print[i][j], float):
                     element_str = f"{mat_to_print[i][j]:.5g}"

                row_str += element_str.rjust(col_widths[j]) + (" " if j < cols - 1 else "")
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
        tolerance = Decimal('1e-9') if self.dtype is Decimal or other.dtype is Decimal else 1e-9

        for i in range(rows):
            for j in range(cols):
                val_self = self[i, j]
                val_other = other[i, j]

                try:
                    # Attempt comparison with tolerance
                    if isinstance(val_self, complex) or isinstance(val_other, complex):
                        if abs(complex(val_self) - complex(val_other)) > tolerance:
                            return False
                    elif isinstance(val_self, Decimal) or isinstance(val_other, Decimal):
                         # Convert both to Decimal for comparison if one is Decimal
                        if abs(Decimal(str(val_self)) - Decimal(str(val_other))) > tolerance:
                             return False
                    else: # Assume float or int
                        if abs(float(val_self) - float(val_other)) > tolerance:
                            return False
                except (TypeError, ValueError):
                     # If types are fundamentally incompatible for comparison
                     return False
        return True

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __getitem__(self, key: tuple[int, int] | int) -> Decimal | complex | float | list[Decimal | complex | float]:
        if isinstance(key, tuple) and len(key) == 2:
           row, col = key
           if 0 <= row < self.size[0] and 0 <= col < self.size[1]:
               return self._mat[row][col]
           else:
               raise IndexError(f"Index ({row}, {col}) out of matrix bounds {self.size}.")
        elif isinstance(key, int):
           row = key
           if 0 <= row < self.size[0]:
               return list(self._mat[row]) # Return a copy
           else:
               raise IndexError(f"Row index {row} out of bounds for size {self.size[0]}.")
        else:
           raise TypeError(f"Index must be a tuple (row, col) or an integer (row), got {type(key)}.")

    def __neg__(self) -> 'BaseMatrix':
        new_mat = [[-element for element in row] for row in self._mat]
        return BaseMatrix(new_mat)

    def _check_compatibility(self, other, operation: str):
        if not isinstance(other, ABCMatrix):
             raise TypeError(f"Unsupported operand type(s) for {operation}: '{type(self).__name__}' and '{type(other).__name__}'")
        if self.size != other.size:
            raise ValueError(f"Matrices must have the same dimensions for {operation}. Sizes are {self.size} and {other.size}")

    def __add__(self, other) -> 'BaseMatrix':
        self._check_compatibility(other, "addition")
        # Determine resulting type
        res_dtype = complex if self.dtype is complex or other.dtype is complex else float if self.dtype is float or other.dtype is float else Decimal
        new_mat = [[res_dtype(self[i, j]) + res_dtype(other[i, j]) for j in range(self.size[1])] for i in range(self.size[0])]
        return BaseMatrix(new_mat)

    def __sub__(self, other) -> 'BaseMatrix':
        self._check_compatibility(other, "subtraction")
        res_dtype = complex if self.dtype is complex or other.dtype is complex else float if self.dtype is float or other.dtype is float else Decimal
        new_mat = [[res_dtype(self[i, j]) - res_dtype(other[i, j]) for j in range(self.size[1])] for i in range(self.size[0])]
        return BaseMatrix(new_mat)

    def __mul__(self, other) -> 'BaseMatrix':
        if isinstance(other, (Decimal, complex, float, int)):
            scalar = other
            res_dtype = self.dtype
            # Determine result type based on scalar
            if isinstance(scalar, complex):
                 res_dtype = complex
                 scalar = complex(scalar)
            elif isinstance(scalar, (float, Decimal)) and res_dtype is Decimal:
                 res_dtype = Decimal 
                 scalar = Decimal(scalar)
            elif not isinstance(scalar, (complex, float, Decimal)):
                 try: # Try Decimal promotion first
                     scalar = Decimal(scalar)
                 except:
                     scalar = float(scalar) # Fallback to float
                     if res_dtype is Decimal: res_dtype = float

            # Ensure scalar has the target type for multiplication consistency
            scalar = res_dtype(scalar) if res_dtype is not complex else complex(scalar)

            new_mat = [[self[i, j] * scalar for j in range(self.size[1])] for i in range(self.size[0])]
            return BaseMatrix(new_mat)

        elif isinstance(other, ABCMatrix):
            if self.size[1] != other.size[0]:
                raise ValueError(f"Matrix dimensions {self.size} and {other.size} are incompatible for multiplication.")

            rows_self, cols_self = self.size
            rows_other, cols_other = other.size

            # Determine result type
            res_dtype = complex if self.dtype is complex or other.dtype is complex else float if self.dtype is float or other.dtype is float else Decimal
            zero_val = res_dtype(0)

            new_mat_data = [[zero_val] * cols_other for _ in range(rows_self)]

            for i in range(rows_self):
                for j in range(cols_other):
                    sum_val = zero_val
                    for k in range(cols_self):
                        # Ensure consistent types for multiplication
                        term1 = res_dtype(self[i, k]) if res_dtype is not complex else complex(self[i, k])
                        term2 = res_dtype(other[k, j]) if res_dtype is not complex else complex(other[k, j])
                        sum_val += term1 * term2
                    new_mat_data[i][j] = sum_val
            return BaseMatrix(new_mat_data)

        else:
            return NotImplemented

    def __rmul__(self, other) -> 'BaseMatrix':
        if isinstance(other, (Decimal, complex, float, int)):
             return self.__mul__(other)
        else:
             return NotImplemented

    def augment(self, other: 'ABCMatrix') -> 'BaseMatrix':
        if not isinstance(other, ABCMatrix):
             raise TypeError(f"Can only augment with another matrix, not {type(other).__name__}")
        if other.size[0] != self.size[0]:
            raise ValueError(f"Matrices must have the same number of rows for augmentation ({self.size[0]} != {other.size[0]}).")

        # Get dense representations using the public method
        mat_self = self.to_list()
        mat_other = other.to_list()

        new_mat = [mat_self[i] + mat_other[i] for i in range(self.size[0])]
        return BaseMatrix(new_mat)

    def transpose(self) -> 'BaseMatrix':
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
        zero_threshold = Decimal('1e-12') if self.dtype is Decimal else (1e-12 if self.dtype is float else 1e-12j).real # Use real part for complex threshold

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
                return self.dtype(0) # Matrix is singular

            if pivot_row != col:
                matrix_copy[col], matrix_copy[pivot_row] = matrix_copy[pivot_row], matrix_copy[col]
                pivot_swaps += 1

            pivot_element = matrix_copy[col][col]
            # No need to divide the pivot row itself in this standard LU approach

            for row in range(col + 1, n):
                # Ensure factor calculation uses consistent types
                # FIX: Cast to complex if needed before division
                val_to_elim = matrix_copy[row][col]
                if self.dtype is complex:
                     factor = complex(val_to_elim) / complex(pivot_element)
                elif self.dtype is Decimal:
                     # Ensure Decimal division
                     factor = Decimal(str(val_to_elim)) / Decimal(str(pivot_element))
                else: # Float
                     factor = float(val_to_elim) / float(pivot_element)

                matrix_copy[row][col] = self.dtype(0) # Set eliminated element to zero
                for c in range(col + 1, n):
                     # Ensure subtraction uses consistent types
                     term_to_subtract = factor * matrix_copy[col][c]
                     matrix_copy[row][c] = matrix_copy[row][c] - term_to_subtract


        # Determinant is the product of the diagonal elements of the upper triangular matrix
        for i in range(n):
            det *= matrix_copy[i][i]

        if pivot_swaps % 2 != 0:
            det *= -1

        # Final check for near-zero determinant
        if abs(det) < zero_threshold:
             return self.dtype(0)

        return det


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
            raise ValueError("Input must be a list of lists with consistent row lengths.")

        self._size = (rows, cols)
        self.data = []
        self.indices = []
        self.indptr = [0] * (rows + 1)
        self._dtype = Decimal # Default preference
        nnz = 0
        zero_threshold = 1e-12 # General threshold

        temp_dtype = Decimal # Track detected type during build

        for r in range(rows):
            for c in range(cols):
                val = mat[r][c]
                current_val = val

                # Type detection and initial conversion (similar to BaseMatrix)
                if isinstance(val, complex):
                    current_val = complex(val)
                    if temp_dtype is not complex: temp_dtype = complex
                elif isinstance(val, float):
                    current_val = Decimal(str(val)) # Try Decimal first
                    if temp_dtype is Decimal: temp_dtype = float
                elif isinstance(val, int) and not isinstance(val, bool):
                     current_val = Decimal(val)
                # else: keep original type if not complex/float/int (e.g., existing Decimal)

                # Check if non-zero using appropriate comparison
                is_non_zero = False
                if isinstance(current_val, complex):
                    if abs(current_val) > zero_threshold: is_non_zero = True
                elif isinstance(current_val, (Decimal, float)):
                    if abs(current_val) > Decimal(str(zero_threshold)): is_non_zero = True
                elif current_val != 0: # For integers/bools
                    is_non_zero = True


                if is_non_zero:
                    self.data.append(current_val) # Add potentially mixed types initially
                    self.indices.append(c)
                    nnz += 1
            self.indptr[r + 1] = nnz

        # Finalize dtype and ensure data consistency
        self._dtype = temp_dtype
        if self._dtype is complex:
            self.data = [complex(d) for d in self.data]
        elif self._dtype is float:
             # Convert only Decimals to float, keep existing floats
             self.data = [float(d) if isinstance(d, Decimal) else d for d in self.data]
        # If dtype remains Decimal, data might contain Decimals and original ints/bools


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
            for i in range(self.indptr[r], self.indptr[r+1]):
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

    def __getitem__(self, key: tuple[int, int] | int) -> Decimal | complex | float | list[Decimal | complex | float]:
        rows, cols = self.size
        zero_val = self.dtype(0)

        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if not (0 <= row < rows and 0 <= col < cols):
                raise IndexError(f"Index ({row}, {col}) out of matrix bounds {self.size}.")

            row_start = self.indptr[row]
            row_end = self.indptr[row+1]

            # Simple linear search (binary search is better for very long rows)
            for i in range(row_start, row_end):
                if self.indices[i] == col:
                    return self.data[i]
            return zero_val # Not found -> zero

        elif isinstance(key, int):
            row = key
            if not (0 <= row < rows):
                raise IndexError(f"Row index {row} out of bounds for size {rows}.")

            dense_row = [zero_val] * cols
            row_start = self.indptr[row]
            row_end = self.indptr[row+1]
            for i in range(row_start, row_end):
                dense_row[self.indices[i]] = self.data[i]
            return dense_row
        else:
           raise TypeError(f"Index must be a tuple (row, col) or an integer (row), got {type(key)}.")

    def __neg__(self) -> 'CSRMatrix':
        """Returns the matrix with all non-zero elements negated."""
        cls = type(self)
        # FIX: Bypass __init__ by using __new__ and setting attributes manually
        new_csr = cls.__new__(cls)
        new_csr.data = [-d for d in self.data]
        new_csr.indices = list(self.indices) # Keep structure
        new_csr.indptr = list(self.indptr)   # Keep structure
        new_csr._size = self.size
        new_csr._dtype = self.dtype
        return new_csr

    def __add__(self, other) -> 'CSRMatrix':
        """Performs matrix addition. Result is CSR."""
        if not isinstance(other, ABCMatrix):
             return NotImplemented
        if self.size != other.size:
            raise ValueError(f"Matrices must have the same dimensions for addition. Sizes are {self.size} and {other.size}")

        # Use BaseMatrix's logic for element access via __getitem__
        # Build dense result first, then convert back to CSR (inefficient but correct)
        res_dtype = complex if self.dtype is complex or other.dtype is complex else float if self.dtype is float or other.dtype is float else Decimal
        new_dense = [[res_dtype(self[i, j]) + res_dtype(other[i, j]) for j in range(self.size[1])] for i in range(self.size[0])]
        return CSRMatrix(new_dense)


    def __sub__(self, other) -> 'CSRMatrix':
        """Performs matrix subtraction. Result is CSR."""
        if not isinstance(other, ABCMatrix):
             return NotImplemented
        if self.size != other.size:
            raise ValueError(f"Matrices must have the same dimensions for subtraction. Sizes are {self.size} and {other.size}")

        res_dtype = complex if self.dtype is complex or other.dtype is complex else float if self.dtype is float or other.dtype is float else Decimal
        new_dense = [[res_dtype(self[i, j]) - res_dtype(other[i, j]) for j in range(self.size[1])] for i in range(self.size[0])]
        return CSRMatrix(new_dense)

    def __mul__(self, other) -> 'CSRMatrix':
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
                if new_dtype is Decimal: new_dtype = float
            elif not isinstance(scalar, Decimal):
                try: scalar = Decimal(scalar)
                except:
                    scalar = float(scalar)
                    if new_dtype is Decimal: new_dtype = float

            # Cast scalar to determined type if necessary for operation
            scalar = new_dtype(scalar) if new_dtype is not complex else complex(scalar)

            # FIX: Bypass __init__
            new_csr = cls.__new__(cls)
            # Perform multiplication, ensuring result matches new_dtype
            new_csr.data = [new_dtype(d) * scalar for d in self.data]
            new_csr.indices = list(self.indices)
            new_csr.indptr = list(self.indptr)
            new_csr._size = self.size
            new_csr._dtype = new_dtype
            return new_csr

        elif isinstance(other, ABCMatrix):
             # Matrix multiplication (dense intermediate for simplicity)
            if self.size[1] != other.size[0]:
                raise ValueError(f"Matrix dimensions {self.size} and {other.size} are incompatible for multiplication.")

            rows_self, cols_self = self.size
            rows_other, cols_other = other.size
            res_dtype = complex if self.dtype is complex or other.dtype is complex else float if self.dtype is float or other.dtype is float else Decimal
            zero_val = res_dtype(0)
            new_mat_data = [[zero_val] * cols_other for _ in range(rows_self)]

            # Slightly optimized: Iterate through self's non-zeros
            for i in range(rows_self):
                 row_start_self = self.indptr[i]
                 row_end_self = self.indptr[i+1]
                 if row_start_self == row_end_self: continue # Skip empty rows

                 for k_idx in range(row_start_self, row_end_self):
                     k = self.indices[k_idx]
                     val_self = self.data[k_idx]

                     # Get relevant elements from other matrix's k-th row
                     # If other is CSR, can be optimized, otherwise use getitem
                     for j in range(cols_other):
                          term2 = other[k, j] # Use getitem for flexibility
                          # Ensure types are compatible for multiplication
                          t1 = res_dtype(val_self) if res_dtype is not complex else complex(val_self)
                          t2 = res_dtype(term2) if res_dtype is not complex else complex(term2)
                          new_mat_data[i][j] += t1 * t2

            return CSRMatrix(new_mat_data) # Convert dense result to CSR
        else:
            return NotImplemented

    def __rmul__(self, other) -> 'CSRMatrix':
        if isinstance(other, (Decimal, complex, float, int)):
            return self.__mul__(other)
        else:
            return NotImplemented

    def augment(self, other: 'ABCMatrix') -> 'CSRMatrix':
        """Augments the matrix by another matrix on the right. Returns CSR."""
        if not isinstance(other, ABCMatrix):
             raise TypeError(f"Can only augment with another matrix, not {type(other).__name__}")
        if other.size[0] != self.size[0]:
            raise ValueError(f"Matrices must have the same number of rows for augmentation ({self.size[0]} != {other.size[0]}).")

        # Convert both to dense, augment, convert back to CSR
        dense_self = self.to_list()
        dense_other = other.to_list() # Use public method

        new_dense = [dense_self[i] + dense_other[i] for i in range(self.size[0])]
        return CSRMatrix(new_dense)

    def transpose(self) -> 'CSRMatrix':
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

        # Create a temporary BaseMatrix from dense representation
        # Use to_list() to get the dense data
        temp_base_mat = BaseMatrix(self.to_list())
        return temp_base_mat.determinant() # Delegate calculation


# Optional: Define Matrix alias if needed elsewhere
Matrix = BaseMatrix