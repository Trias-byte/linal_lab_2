import unittest
from decimal import Decimal
from matrix import BaseMatrix, CSRMatrix # Import necessary classes

# --- Unit Tests ---
class TestMatrixImplementations(unittest.TestCase):

    def setUp(self):
        # Define sample matrices (use Decimal for precision where needed)
        self.m1_data = [[Decimal(1), Decimal(0), Decimal(2)],
                        [Decimal(0), Decimal(3), Decimal(0)],
                        [Decimal(4), Decimal(0), Decimal(5)]]
        self.m2_data = [[Decimal(6), Decimal(1), Decimal(0)],
                        [Decimal(0), Decimal(7), Decimal(2)],
                        [Decimal(8), Decimal(0), Decimal(9)]]
        self.m_zeros_data = [[0, 0], [0, 0]]
        self.m_identity_data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.m_rect_data = [[1, 2, 3], [4, 5, 6]]
        # Correct complex data init - use python complex type
        self.m_complex_data = [[1+1j, 0], [2, 3-2j]]

        # Instantiate both BaseMatrix and CSRMatrix
        self.base1 = BaseMatrix(self.m1_data)
        self.csr1 = CSRMatrix(self.m1_data)
        self.base2 = BaseMatrix(self.m2_data)
        self.csr2 = CSRMatrix(self.m2_data)
        self.base_zeros = BaseMatrix(self.m_zeros_data)
        self.csr_zeros = CSRMatrix(self.m_zeros_data)
        self.base_id = BaseMatrix(self.m_identity_data)
        self.csr_id = CSRMatrix(self.m_identity_data)
        self.base_rect = BaseMatrix(self.m_rect_data)
        self.csr_rect = CSRMatrix(self.m_rect_data)
        self.base_complex = BaseMatrix(self.m_complex_data)
        self.csr_complex = CSRMatrix(self.m_complex_data)


    def test_initialization_and_size(self):
        self.assertEqual(self.base1.size, (3, 3))
        self.assertEqual(self.csr1.size, (3, 3))
        self.assertEqual(self.base_rect.size, (2, 3))
        self.assertEqual(self.csr_rect.size, (2, 3))
        self.assertEqual(self.base_complex.size, (2, 2))
        self.assertEqual(self.csr_complex.size, (2, 2))

        # Test CSR internal structure lightly
        self.assertEqual(self.csr1.data, [Decimal(1), Decimal(2), Decimal(3), Decimal(4), Decimal(5)])
        self.assertEqual(self.csr1.indices, [0, 2, 1, 0, 2])
        self.assertEqual(self.csr1.indptr, [0, 2, 3, 5])
        # Test CSR complex data
        # Note: order depends on traversal, assuming row-major
        self.assertEqual(self.csr_complex.data, [1+1j, 2+0j, 3-2j])
        self.assertEqual(self.csr_complex.indices, [0, 0, 1]) # Indices for non-zero elements
        self.assertEqual(self.csr_complex.indptr, [0, 1, 3]) # Row pointers

    def test_to_list(self):
         self.assertEqual(self.base1.to_list(), self.m1_data)
         self.assertEqual(self.csr1.to_list(), self.m1_data) # CSR should reconstruct to original dense
         self.assertEqual(self.base_complex.to_list(), [[1+1j, 0j], [2+0j, 3-2j]]) # Base should convert 0 to 0j
         self.assertEqual(self.csr_complex.to_list(), [[1+1j, 0j], [2+0j, 3-2j]]) # CSR should also handle 0j


    def test_equality(self):
        base1_copy = BaseMatrix(self.m1_data)
        csr1_copy = CSRMatrix(self.m1_data)
        self.assertEqual(self.base1, base1_copy)
        self.assertEqual(self.csr1, csr1_copy)
        self.assertEqual(self.base1, self.csr1) # Check cross-type equality
        self.assertEqual(self.csr1, self.base1)
        self.assertNotEqual(self.base1, self.base2)
        self.assertNotEqual(self.csr1, self.csr2)
        self.assertNotEqual(self.base1, self.csr2)
        self.assertEqual(self.base_zeros, self.csr_zeros)
        self.assertEqual(self.base_complex, self.csr_complex)

    def test_getitem(self):
        # Single element access
        self.assertEqual(self.base1[0, 0], Decimal(1))
        self.assertEqual(self.csr1[0, 0], Decimal(1))
        self.assertEqual(self.base1[0, 1], Decimal(0))
        self.assertEqual(self.csr1[0, 1], Decimal(0)) # Access zero element
        self.assertEqual(self.base1[2, 2], Decimal(5))
        self.assertEqual(self.csr1[2, 2], Decimal(5))

        # Row access
        self.assertEqual(self.base1[1], [Decimal(0), Decimal(3), Decimal(0)])
        self.assertEqual(self.csr1[1], [Decimal(0), Decimal(3), Decimal(0)])
        # BaseMatrix converts input ints/floats to Decimal/complex
        self.assertEqual(self.base_rect[0], [Decimal('1'), Decimal('2'), Decimal('3')])
        self.assertEqual(self.csr_rect[0], [Decimal('1'), Decimal('2'), Decimal('3')])

        # Complex access
        self.assertEqual(self.base_complex[0, 0], 1+1j)
        self.assertEqual(self.csr_complex[0, 0], 1+1j)
        self.assertEqual(self.base_complex[0, 1], 0j) # Base should return 0j
        self.assertEqual(self.csr_complex[0, 1], 0j) # CSR zero is typed 0j


    def test_negation(self):
        neg_base1 = -self.base1
        neg_csr1 = -self.csr1
        expected_neg_data = [[Decimal(-1), Decimal(0), Decimal(-2)],
                             [Decimal(0), Decimal(-3), Decimal(0)],
                             [Decimal(-4), Decimal(0), Decimal(-5)]]
        expected_neg_base = BaseMatrix(expected_neg_data)

        self.assertEqual(neg_base1, expected_neg_base)
        # Compare neg_csr1 element-wise via equality with expected BaseMatrix
        self.assertEqual(neg_csr1, expected_neg_base)
        self.assertEqual(neg_csr1, neg_base1) # Check cross-type after operation

        # Test complex negation
        neg_complex_csr = -self.csr_complex
        expected_neg_complex = BaseMatrix([[-1-1j, 0], [-2, -3+2j]])
        self.assertEqual(neg_complex_csr, expected_neg_complex)


    def test_addition(self):
        sum_base = self.base1 + self.base2
        sum_csr = self.csr1 + self.csr2
        expected_sum_data = [[Decimal(7), Decimal(1), Decimal(2)],
                             [Decimal(0), Decimal(10), Decimal(2)],
                             [Decimal(12), Decimal(0), Decimal(14)]]
        expected_sum_base = BaseMatrix(expected_sum_data)

        self.assertEqual(sum_base, expected_sum_base)
        self.assertEqual(sum_csr, expected_sum_base) # Compare CSR result to expected dense
        self.assertEqual(sum_csr, sum_base)       # Compare CSR result to Base result

    def test_subtraction(self):
        diff_base = self.base1 - self.base2
        diff_csr = self.csr1 - self.csr2
        expected_diff_data = [[Decimal(-5), Decimal(-1), Decimal(2)],
                              [Decimal(0), Decimal(-4), Decimal(-2)],
                              [Decimal(-4), Decimal(0), Decimal(-4)]]
        expected_diff_base = BaseMatrix(expected_diff_data)

        self.assertEqual(diff_base, expected_diff_base)
        self.assertEqual(diff_csr, expected_diff_base)
        self.assertEqual(diff_csr, diff_base)

    def test_scalar_multiplication(self):
        scalar_dec = Decimal(3)
        scalar_float = 2.0
        scalar_complex = 1+1j

        # --- Decimal Scalar ---
        prod_base_dec = self.base1 * scalar_dec
        prod_csr_dec = self.csr1 * scalar_dec
        prod_base_rmul_dec = scalar_dec * self.base1
        prod_csr_rmul_dec = scalar_dec * self.csr1
        expected_prod_data_dec = [[Decimal(3), 0, Decimal(6)], [0, Decimal(9), 0], [Decimal(12), 0, Decimal(15)]]
        expected_prod_base_dec = BaseMatrix(expected_prod_data_dec)

        self.assertEqual(prod_base_dec, expected_prod_base_dec)
        self.assertEqual(prod_csr_dec, expected_prod_base_dec)
        self.assertEqual(prod_csr_dec, prod_base_dec)
        self.assertEqual(prod_base_rmul_dec, expected_prod_base_dec)
        self.assertEqual(prod_csr_rmul_dec, expected_prod_base_dec)
        self.assertEqual(prod_csr_rmul_dec, prod_csr_dec)

        # --- Float Scalar ---
        prod_base_fl = self.base1 * scalar_float # Base should handle float -> Decimal
        prod_csr_fl = self.csr1 * scalar_float   # CSR should detect float and result in float dtype
        expected_prod_data_fl = [[Decimal('2.0'), Decimal('0.0'), Decimal('4.0')],
                                 [Decimal('0.0'), Decimal('6.0'), Decimal('0.0')],
                                 [Decimal('8.0'), Decimal('0.0'), Decimal('10.0')]]
        expected_prod_base_fl = BaseMatrix(expected_prod_data_fl) # Base result is Decimal

        self.assertEqual(prod_base_fl, expected_prod_base_fl)
        # CSR result should be float, compare element-wise
        self.assertEqual(prod_csr_fl.dtype, float)
        # Use BaseMatrix equality for comparison (handles type diffs)
        self.assertEqual(prod_csr_fl, expected_prod_base_fl)


        # --- Complex Scalar ---
        prod_csr_cx = self.csr1 * scalar_complex   # Result should be complex

        # Expected: [[1*(1+1j), 0, 2*(1+1j)], [0, 3*(1+1j), 0], [4*(1+1j), 0, 5*(1+1j)]]
        #         = [[1+1j, 0, 2+2j], [0, 3+3j, 0], [4+4j, 0, 5+5j]]
        expected_prod_data_cx = [[1+1j, 0j, 2+2j], [0j, 3+3j, 0j], [4+4j, 0j, 5+5j]]
        expected_prod_base_cx = BaseMatrix(expected_prod_data_cx)

        self.assertEqual(prod_csr_cx, expected_prod_base_cx)
        self.assertEqual(prod_csr_cx.dtype, complex)

    def test_matrix_multiplication(self):
        prod_base = self.base1 * self.base2
        prod_csr = self.csr1 * self.csr2
        expected_prod_data = [[Decimal(22), Decimal(1), Decimal(18)],
                              [Decimal(0), Decimal(21), Decimal(6)],
                              [Decimal(64), Decimal(4), Decimal(45)]]
        expected_prod_base = BaseMatrix(expected_prod_data)

        self.assertEqual(prod_base, expected_prod_base)
        self.assertEqual(prod_csr, expected_prod_base)
        self.assertEqual(prod_csr, prod_base)

        # Test multiplication by identity
        prod_base_id = self.base1 * self.base_id
        prod_csr_id = self.csr1 * self.csr_id
        self.assertEqual(prod_base_id, self.base1)
        self.assertEqual(prod_csr_id, self.csr1)
        self.assertEqual(prod_csr_id, self.base1)

        # Test complex multiplication
        prod_complex = self.csr_complex * self.csr_complex
        # Expected: [[1+1j, 0], [2, 3-2j]] * [[1+1j, 0], [2, 3-2j]]
        # Row 0: [(1+1j)*(1+1j) + 0*2, (1+1j)*0 + 0*(3-2j)] = [1+2j-1+0, 0] = [2j, 0j]
        # Row 1: [2*(1+1j) + (3-2j)*2, 2*0 + (3-2j)*(3-2j)] = [2+2j + 6-4j, 0 + 9-12j+4j^2] = [8-2j, 9-12j-4] = [8-2j, 5-12j]
        expected_complex_prod = BaseMatrix([[0+2j, 0j], [8-2j, 5-12j]])
        self.assertEqual(prod_complex, expected_complex_prod)
        self.assertEqual(prod_complex.dtype, complex)

    def test_transpose(self):
        trans_base = self.base1.transpose()
        trans_csr = self.csr1.transpose()
        expected_trans_data = [[Decimal(1), 0, Decimal(4)], [0, Decimal(3), 0], [Decimal(2), 0, Decimal(5)]]
        expected_trans_base = BaseMatrix(expected_trans_data)

        self.assertEqual(trans_base, expected_trans_base)
        self.assertEqual(trans_csr, expected_trans_base)
        self.assertEqual(trans_csr, trans_base)

        trans_rect_base = self.base_rect.transpose()
        trans_rect_csr = self.csr_rect.transpose()
        expected_trans_rect = BaseMatrix([[Decimal('1'),Decimal('4')],[Decimal('2'),Decimal('5')],[Decimal('3'),Decimal('6')]])
        self.assertEqual(trans_rect_base, expected_trans_rect)
        self.assertEqual(trans_rect_csr, expected_trans_rect)
        self.assertEqual(trans_rect_csr.size, (3, 2))

    def test_augment(self):
        aug_base = self.base1.augment(self.base_id)
        aug_csr = self.csr1.augment(self.csr_id) # Augment CSR with CSR
        aug_csr_base = self.csr1.augment(self.base_id) # Augment CSR with Base

        expected_aug_data = [[Decimal(1), 0, Decimal(2), Decimal(1), 0, 0],
                             [0, Decimal(3), 0, 0, Decimal(1), 0],
                             [Decimal(4), 0, Decimal(5), 0, 0, Decimal(1)]]
        expected_aug_base = BaseMatrix(expected_aug_data)

        self.assertEqual(aug_base, expected_aug_base)
        self.assertEqual(aug_csr, expected_aug_base)
        self.assertEqual(aug_csr_base, expected_aug_base)
        self.assertEqual(aug_csr, aug_base)
        self.assertEqual(aug_csr.size, (3, 6))

    def test_determinant(self):
        # Using assertAlmostEqual for potential float/complex inaccuracies
        # det(m1) = -9
        self.assertAlmostEqual(self.base1.determinant(), Decimal(-9))
        self.assertAlmostEqual(self.csr1.determinant(), Decimal(-9))

        # det(m2) = 394
        self.assertAlmostEqual(self.base2.determinant(), Decimal(394))
        self.assertAlmostEqual(self.csr2.determinant(), Decimal(394))

        # det(identity) = 1
        self.assertAlmostEqual(self.base_id.determinant(), Decimal(1))
        self.assertAlmostEqual(self.csr_id.determinant(), Decimal(1))

        # det(zeros) = 0
        self.assertAlmostEqual(self.base_zeros.determinant(), Decimal(0))
        self.assertAlmostEqual(self.csr_zeros.determinant(), Decimal(0))

        # Test complex determinant (Expected: 5 + 1j)
        # Use delta for complex comparison
        delta = 1e-9
        det_complex_base = self.base_complex.determinant()
        det_complex_csr = self.csr_complex.determinant()
        self.assertAlmostEqual(det_complex_base.real, 5, delta=delta)
        self.assertAlmostEqual(det_complex_base.imag, 1, delta=delta)
        self.assertAlmostEqual(det_complex_csr.real, 5, delta=delta)
        self.assertAlmostEqual(det_complex_csr.imag, 1, delta=delta)
        self.assertEqual(self.base_complex.dtype, complex)
        self.assertEqual(self.csr_complex.dtype, complex)

    def test_type_errors_and_value_errors(self):
        # Incompatible sizes for arithmetic/augmentation
        with self.assertRaises(ValueError): self.base1 + self.base_rect
        with self.assertRaises(ValueError): self.csr1 + self.csr_rect
        with self.assertRaises(ValueError): self.base1.augment(self.base_rect)
        with self.assertRaises(ValueError): self.csr1.augment(self.csr_rect)

        # Incompatible dimensions for multiplication
        # FIX: Test a case that should actually fail (3x3 * 2x3)
        with self.assertRaises(ValueError): self.base1 * self.base_rect
        with self.assertRaises(ValueError): self.csr1 * self.csr_rect
        # Test another failing case (2x3 * 2x3)
        with self.assertRaises(ValueError): self.base_rect * self.base_rect

        # Determinant on non-square
        with self.assertRaises(ValueError): self.base_rect.determinant()
        with self.assertRaises(ValueError): self.csr_rect.determinant()

        # Type errors for operations with incompatible types
        with self.assertRaises(TypeError): self.base1 + "string"
        with self.assertRaises(TypeError): self.csr1 - [1, 2, 3]
        with self.assertRaises(TypeError): self.base1 * "string"
        with self.assertRaises(TypeError): "string" * self.csr1
        with self.assertRaises(TypeError): self.base1.augment("string")
        with self.assertRaises(TypeError): self.csr1.augment(None)

        # Initialization errors
        with self.assertRaises(ValueError): BaseMatrix([]) # Empty list
        with self.assertRaises(ValueError): BaseMatrix([[]]) # List with empty row
        with self.assertRaises(ValueError): BaseMatrix([[1, 2], [3]]) # Inconsistent lengths
        with self.assertRaises(ValueError): CSRMatrix([])
        with self.assertRaises(ValueError): CSRMatrix([[]])
        with self.assertRaises(ValueError): CSRMatrix([[1, 2], [3]])


# Run tests from the command line
if __name__ == '__main__':
    unittest.main()