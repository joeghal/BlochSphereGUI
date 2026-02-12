import numpy as np
import cudaq


# helper that builds a trivial kernel applying a single named gate
# and returns the state vector for |0> after the gate.
def state_after(gate_fn):
    @cudaq.kernel
    def circ():
        q = cudaq.qubit()
        gate_fn(q)

    return np.array(cudaq.get_state(circ), dtype=complex)


def test_h_gate():
    # Hadamard sends |0> -> (|0>+|1>)/√2
    state = state_after(lambda q: cudaq.h(q))
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(state, expected)


def test_x_gate():
    state = state_after(lambda q: cudaq.x(q))
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(state, expected)


def test_s_gate():
    state = state_after(lambda q: cudaq.s(q))
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(state, expected)
