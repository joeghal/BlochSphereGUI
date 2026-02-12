import numpy as np

# define gates
H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)

zero = np.array([1.0+0j, 0.0+0j])

h_zero = H @ zero
x_zero = X @ zero
s_zero = S @ zero

print('H|0> =', np.round(h_zero, 6))
print('X|0> =', np.round(x_zero, 6))
print('S|0> =', np.round(s_zero, 6))

# expected
exp_h = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
assert np.allclose(h_zero, exp_h)
assert np.allclose(x_zero, np.array([0,1]))
assert np.allclose(s_zero, np.array([1,0]))

print('All gate tests passed')
