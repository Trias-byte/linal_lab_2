class Matrix:
    def __init__(self, mat:list[list]):
        self.mat = mat
        self.size = (len(self.mat), len(self.mat[0]))

    def __str__(self):
        return "\n" + "\n".join(str(x) for x in self.mat) + "\n"
    
    def __eq__(self, b):
        return all([x in self.mat for x in b.mat ]) and self.size == b.size
    
    def __ne__(self, b):
        return not (self == b)

    def __getitem__(self, key):
        if isinstance(key, (tuple)):
           return self.mat[key[0]][key[1]]
        elif isinstance(key, (int)):
           return self.mat[key]
        return self.mat[key[0]][key[1]]
        
    def __add__(self, b):
        if self.size != self.size:
            raise ValueError("Разная размерность матриц")
        mat = []
        for i in range(self.size[0]): 
            row = []
            for y in range(self.size[1]):
                row.append(self[(i, y)] + b[(i, y)])
            mat.append(row)
        return Matrix(mat)
    
    def __sub__(self, b):
        if self.size != self.size:
            raise ValueError("Разная размерность матриц")
        mat = []
        for i in range(self.size[0]): 
            row = []
            for y in range(self.size[1]):
                row.append(self[(i, y)] - b[(i, y)])
            mat.append(row)
        return Matrix(mat)
    def __mul__(self, b):
        if isinstance(b, (int, float)):
            mat = []
            for i in range(self.size[0]): 
                row = []
                for y in range(self.size[1]):
                    row.append(self[(i, y)] * b)
                mat.append(row)
            return Matrix(mat)
        
        elif isinstance(b, (Matrix, list)):
            if self.size[1] != b.size[0]:
                raise ValueError(f"Разная размерность матриц {self.size} и {b.size}" )
            mat = []
            for i in range(self.size[0]): 
                row = []
                for y in range(b.size[1]):
                    row.append(sum([self[(i, k)] * b[(k, y)] for k in range(self.size[1])]))
                mat.append(row)
            return Matrix(mat)
        
        else:
            raise TypeError("Можно умножать только на число (int или float)")
    def __rmul__(self, b):
        return self.__mul__(b)
    

    def __neg__(self):
        pass

    def get_matrix(self):
        return self.mat
        
    def augment(self, b: 'Matrix'):
        if b.size[0] != self.size[0]:
            raise "Разная размерность"
        return [self.mat[i] + b.mat[i] for i in b.size[0] ]
        
    def transpose(self):
        a = []
        for i in range(self.size[1]): 
            row = []
            for j in range(self.size[0]):
                row.append(self.mat[j][i])
            a.append(row)
        return Matrix(a)
    
    def determinant(self) -> float:
        """
        Вычисляет определитель квадратной матрицы методом Гаусса.
        Вход: квадратная матрица A (n x n)
        Выход: определитель (float)
        Raises: ValueError, если матрица не квадратная
        """
        if self.size[0] != self.size[1]:
            raise ValueError("Матрица должна быть квадратной")

        n = self.size[1]
        det = 1.0
        matrix = self.mat  # Создаем копию, чтобы не менять исходную матрицу

        for col in range(n):
            # Поиск ненулевого элемента в текущем столбце
            pivot_row = None
            for row in range(col, n):
                if abs(matrix[row][col]) > 1e-10:
                    pivot_row = row
                    break

            if pivot_row is None:
                return 0.0  # Все элементы столбца нулевые -> det = 0

            # Перестановка строк
            if pivot_row != col:
                matrix.swap_rows(col, pivot_row)
                det *= -1  # При перестановке строк знак определителя меняется

            # Нормализация ведущей строки
            pivot = matrix[col][col]
            det *= pivot  # Умножаем определитель на ведущий элемент

            # Исключение элементов ниже ведущего
            for row in range(col + 1, n):
                factor = matrix[row][col] / pivot
                for c in range(col, n):
                    matrix[row][c] -= factor * matrix[col][c]

        return det
