"""State-logic and GUI tests for the Bloch sphere app.

GUI cases need a display (Xvfb is enough). Quantum kernels run on the
CUDA-Q CPU simulator.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")

import cudaq
import tkinter as tk

from bloch_gui.bloch_sphere_gui import (
    BlochSphereGUI,
    bloch_coordinates,
    named_gate_matrix,
    select_cudaq_target,
    state_from_slider_angles,
)


def _phase_fidelity(got, expected):
    got = np.asarray(got, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    got = got / np.linalg.norm(got)
    expected = expected / np.linalg.norm(expected)
    return abs(np.vdot(expected, got))


def _read_entries(gui):
    return np.array(
        [complex(gui.entry_a0.get()), complex(gui.entry_a1.get())],
        dtype=complex,
    )


def _set_entries(gui, a0, a1):
    gui.entry_a0.delete(0, tk.END)
    gui.entry_a0.insert(0, str(a0))
    gui.entry_a1.delete(0, tk.END)
    gui.entry_a1.insert(0, str(a1))


def _clear_entries(gui):
    gui.entry_a0.delete(0, tk.END)
    gui.entry_a1.delete(0, tk.END)


@pytest.fixture
def app():
    root = tk.Tk()
    root.geometry("1100x820+0+0")
    gui = BlochSphereGUI(root)
    root.update_idletasks()
    root.update()
    yield gui, root
    try:
        root.destroy()
    except tk.TclError:
        pass


def test_bloch_coordinates_basis_states():
    assert bloch_coordinates([1, 0]) == pytest.approx((0.0, 0.0, 1.0))
    assert bloch_coordinates([0, 1]) == pytest.approx((0.0, 0.0, -1.0))
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert bloch_coordinates(plus) == pytest.approx((1.0, 0.0, 0.0))
    plus_i = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    assert bloch_coordinates(plus_i) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_slider_angles_match_cpu_simulator():
    cudaq.set_target("qpp-cpu")
    samples = [(0.0, 0.0, 0.0), (0.5, 0.25, 0.0), (1.0, 0.0, 0.5), (0.3, 1.2, 1.7)]
    for nx, ny, nz in samples:
        @cudaq.kernel
        def rotated():
            q = cudaq.qubit()
            rx(nx * np.pi, q)
            ry(ny * np.pi, q)
            rz(nz * np.pi, q)

        got = np.array(cudaq.get_state(rotated), dtype=complex)
        expected = state_from_slider_angles(nx, ny, nz)
        assert _phase_fidelity(got, expected) == pytest.approx(1.0, abs=1e-8)
        assert np.allclose(got, expected, atol=1e-8)


def test_unusable_gpu_request_falls_back_to_qpp_cpu(monkeypatch):
    """A GPU target that cannot start must not escape to the caller."""
    monkeypatch.setenv("CUDAQ_DEFAULT_SIMULATOR", "nvidia")
    name = select_cudaq_target()
    assert name == "qpp-cpu"
    assert cudaq.get_target().name == "qpp-cpu"

    @cudaq.kernel
    def flipped():
        q = cudaq.qubit()
        x(q)

    state = np.array(cudaq.get_state(flipped), dtype=complex)
    assert np.allclose(state, [0, 1])


def test_reported_gpu_target_with_no_gpu_switches_to_cpu(monkeypatch):
    """Startup path: the active target advertises a GPU but none is usable."""
    monkeypatch.delenv("CUDAQ_DEFAULT_SIMULATOR", raising=False)
    cudaq.set_target("qpp-cpu")
    real_get_target = cudaq.get_target
    real_set_target = cudaq.set_target
    calls = []
    switched = {"done": False}

    class FakeGpuTarget:
        name = "nvidia"
        simulator = "custatevec_fp32"

    def get_target():
        if not switched["done"]:
            return FakeGpuTarget()
        return real_get_target()

    def set_target(name, **kwargs):
        calls.append(name)
        switched["done"] = True
        return real_set_target(name, **kwargs)

    monkeypatch.setattr(cudaq, "get_target", get_target)
    monkeypatch.setattr(cudaq, "set_target", set_target)
    monkeypatch.setattr(cudaq, "num_available_gpus", lambda: 0)

    assert select_cudaq_target() == "qpp-cpu"
    assert calls == ["qpp-cpu"]


def test_explicit_cpu_simulator_env_is_honored(monkeypatch):
    monkeypatch.setenv("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu")
    assert select_cudaq_target() == "qpp-cpu"
    assert cudaq.get_target().name == "qpp-cpu"


def test_app_starts_on_cpu_target(app):
    gui, _root = app
    assert gui.cudaq_target == "qpp-cpu"
    assert gui._cudaq_ready
    text = gui.state_text.get("1.0", "end")
    assert "x =" in text and "y =" in text and "z =" in text


def _line_is_on_screen(gui, root, label):
    index = gui.state_text.search(label, "1.0", tk.END)
    assert index, f"{label} missing from state text"
    bbox = gui.state_text.bbox(index)
    assert bbox is not None, f"{label} is clipped inside the text widget"
    _x, y, _w, h = bbox
    top = gui.state_text.winfo_rooty() + y
    bottom = top + h
    root_top = root.winfo_rooty()
    root_bottom = root_top + root.winfo_height()
    assert root_top <= top and bottom <= root_bottom, (
        f"{label} at {top}:{bottom} outside window {root_top}:{root_bottom}"
    )
    canvas = gui.control_canvas
    canvas_top = canvas.winfo_rooty()
    canvas_bottom = canvas_top + canvas.winfo_height()
    assert canvas_top <= top and bottom <= canvas_bottom, (
        f"{label} at {top}:{bottom} outside control canvas {canvas_top}:{canvas_bottom}"
    )


def test_state_information_shows_y_and_z(app):
    gui, root = app
    # The class sets the default window size; don't depend on the fixture.
    root.geometry("900x700+0+0")
    gui.apply_named_gate("H")
    root.update_idletasks()
    root.update()
    text = gui.state_text.get("1.0", "end")
    assert "Bloch Coordinates:" in text
    for label in ("x =", "y =", "z ="):
        _line_is_on_screen(gui, root, label)
    height = int(gui.state_text.cget("height"))
    lines = int(gui.state_text.index("end-1c").split(".")[0])
    assert height >= lines


def test_reset_after_apply_angles_returns_ket0(app):
    gui, root = app
    gui.entry_x.delete(0, tk.END)
    gui.entry_x.insert(0, "1.00")
    gui.entry_y.delete(0, tk.END)
    gui.entry_y.insert(0, "0.00")
    gui.entry_z.delete(0, tk.END)
    gui.entry_z.insert(0, "0.00")
    gui.apply_angles()
    root.update()
    assert gui.entry_a0.get().strip() == ""
    assert gui.entry_a1.get().strip() == ""
    assert gui.last_tip[2] < -0.9

    gui.apply_named_gate("Reset")
    root.update()
    state = _read_entries(gui)
    assert np.allclose(state, [1, 0], atol=1e-6)
    assert gui.last_tip[2] == pytest.approx(1.0, abs=1e-4)
    assert gui._evolution_backend == "cudaq"


def test_reset_with_empty_entries_uses_cudaq(app):
    gui, _root = app
    _clear_entries(gui)
    gui.slider_x.set(0.4)
    gui.slider_y.set(0.2)
    gui.apply_named_gate("Reset")
    state = _read_entries(gui)
    assert np.allclose(state, [1, 0], atol=1e-6)
    assert gui._evolution_backend == "cudaq"


def test_named_gates_with_filled_entries_use_cudaq(app, monkeypatch):
    gui, _root = app
    calls = {"n": 0}
    real_get_state = cudaq.get_state

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real_get_state(*args, **kwargs)

    monkeypatch.setattr(cudaq, "get_state", _counting)

    start = np.array([0.6 + 0.0j, 0.8 + 0.0j])
    for name in ("H", "X", "Y", "Z", "S", "T", "RX90", "RY90", "RZ90"):
        _set_entries(gui, start[0], start[1])
        calls["n"] = 0
        gui.apply_named_gate(name)
        got = _read_entries(gui)
        expected = named_gate_matrix(name) @ start
        assert calls["n"] >= 1, name
        assert gui._evolution_backend == "cudaq", name
        assert _phase_fidelity(got, expected) == pytest.approx(1.0, abs=1e-5)
        assert np.allclose(got, expected, atol=1e-5)


def test_arbitrary_unitary_uses_cudaq(app):
    gui, _root = app
    _set_entries(gui, "0.6", "0.8")
    theta = 0.7
    axis = np.array([1.0, 2.0, -0.5])
    axis = axis / np.linalg.norm(axis)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    n_dot = axis[0] * X + axis[1] * Y + axis[2] * Z
    unitary = np.cos(theta / 2) * np.eye(2, dtype=complex) - 1j * np.sin(theta / 2) * n_dot
    state = np.array([0.6, 0.8], dtype=complex)
    gui.apply_unitary(unitary)
    got = _read_entries(gui)
    assert gui._evolution_backend == "cudaq"
    assert np.allclose(got, unitary @ state, atol=1e-5)


def test_numpy_fallback_when_cudaq_disabled(app):
    gui, _root = app
    gui._cudaq_ready = False
    _set_entries(gui, "1", "0")
    gui.apply_named_gate("X")
    assert gui._evolution_backend == "numpy"
    assert np.allclose(_read_entries(gui), [0, 1], atol=1e-6)
    gui.apply_named_gate("Reset")
    assert gui._evolution_backend == "numpy"
    assert np.allclose(_read_entries(gui), [1, 0], atol=1e-6)


def test_error_labels_clear_after_success(app):
    gui, root = app
    gui.entry_x.delete(0, tk.END)
    gui.entry_x.insert(0, "9")
    gui.apply_angles()
    assert "between 0 and 2" in gui.angles_error_label.cget("text")

    gui.entry_x.delete(0, tk.END)
    gui.entry_x.insert(0, "0.00")
    _set_entries(gui, "nope", "1")
    gui.apply_named_gate("X")
    assert "Invalid" in gui.state_error_label.cget("text")
    assert gui.angles_error_label.cget("text")

    _set_entries(gui, "1", "0")
    gui.apply_named_gate("H")
    assert gui.state_error_label.cget("text") == ""
    assert gui.angles_error_label.cget("text") == ""

    gui.state_error_label.config(text="State vector cannot be zero!")
    gui.angles_error_label.config(text="Invalid entry - use decimal numbers")
    gui.apply_named_gate("Reset")
    assert gui.state_error_label.cget("text") == ""
    assert gui.angles_error_label.cget("text") == ""

    gui.state_error_label.config(text="still here")
    gui.apply_named_gate("Random")
    root.update()
    assert gui.state_error_label.cget("text") == ""
    assert gui._evolution_backend == "cudaq"
    assert np.isclose(np.linalg.norm(_read_entries(gui)), 1.0)

    gui.angles_error_label.config(text="angle error")
    _set_entries(gui, "0", "1")
    gui.apply_state_vector()
    assert gui.angles_error_label.cget("text") == ""
    assert gui.state_error_label.cget("text") == ""


def _trough_point(scale, kind, min_fraction=0.0):
    width = scale.winfo_width()
    height = scale.winfo_height()
    found = None
    for y in range(height):
        for x in range(width):
            if x < width * min_fraction:
                continue
            if scale.identify(x, y) == kind:
                found = (x, y)
                break
        if found:
            break
    return found


def test_slider_trough_click_and_drag(app):
    gui, root = app
    scale = gui.slider_x
    scale.set(0.0)
    root.update_idletasks()
    root.update()

    # Install the same mouse-wheel bindings the pointer uses over the canvas.
    gui.control_canvas.event_generate("<Enter>")
    root.update()

    click = _trough_point(scale, "trough2", min_fraction=0.7)
    assert click is not None
    expected = float(scale.tk.call(scale._w, "get", click[0], click[1]))
    assert expected > 1.0

    scale.event_generate("<Button-1>", x=click[0], y=click[1])
    root.update()
    assert scale.get() == pytest.approx(expected, abs=0.02)
    assert scale.get() > 1.0

    drag_to_x = max(2, click[0] // 5)
    scale.event_generate("<B1-Motion>", x=drag_to_x, y=click[1])
    root.update()
    dragged = float(scale.tk.call(scale._w, "get", drag_to_x, click[1]))
    assert scale.get() == pytest.approx(dragged, abs=0.02)
    assert scale.get() < expected - 0.2

    scale.event_generate("<ButtonRelease-1>", x=drag_to_x, y=click[1])
    root.update()
    try:
        scale.grab_release()
    except tk.TclError:
        pass


def _walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from _walk(child)


def test_custom_axis_dialog_uses_cudaq(app):
    gui, root = app
    _set_entries(gui, "1", "0")
    gui.open_arbitrary_rotation_dialog()
    root.update()
    dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
    menu = next(w for w in _walk(dlg) if w.winfo_class() == "Menubutton")
    dlg.setvar(menu.cget("textvariable"), "Custom")
    entries = [w for w in _walk(dlg) if isinstance(w, tk.Entry)]
    # angle, then nx, ny, nz
    assert len(entries) >= 4
    for entry, value in zip(entries, ("0.50", "0", "1", "0")):
        entry.delete(0, tk.END)
        entry.insert(0, value)
    apply_btn = next(w for w in _walk(dlg) if isinstance(w, tk.Button) and w.cget("text") == "Apply")
    apply_btn.invoke()
    root.update()
    assert gui._evolution_backend == "cudaq"
    # RY(π/2) on |0⟩ is |+⟩, so the Bloch vector points along +x.
    assert gui.last_tip[0] == pytest.approx(1.0, abs=1e-4)
    assert abs(gui.last_tip[2]) < 1e-3

    gui.open_arbitrary_rotation_dialog()
    root.update()
    dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
    menu = next(w for w in _walk(dlg) if w.winfo_class() == "Menubutton")
    dlg.setvar(menu.cget("textvariable"), "Custom")
    entries = [w for w in _walk(dlg) if isinstance(w, tk.Entry)]
    for entry, value in zip(entries, ("1.00", "0", "0", "0")):
        entry.delete(0, tk.END)
        entry.insert(0, value)
    apply_btn = next(w for w in _walk(dlg) if isinstance(w, tk.Button) and w.cget("text") == "Apply")
    apply_btn.invoke()
    root.update()
    assert "zero" in gui.state_error_label.cget("text").lower()
    gui.apply_named_gate("Reset")
    assert gui.state_error_label.cget("text") == ""
    dlg.destroy()


def test_arbitrary_rotation_dialog_uses_cudaq(app):
    gui, root = app
    gui.apply_named_gate("Reset")
    gui.open_arbitrary_rotation_dialog()
    root.update()
    dialogs = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    dlg = dialogs[-1]
    apply_btn = None
    for child in dlg.winfo_children():
        if isinstance(child, tk.Button) and child.cget("text") == "Apply":
            apply_btn = child
    assert apply_btn is not None
    apply_btn.invoke()
    root.update()
    assert gui._evolution_backend == "cudaq"
    # RX(π) on |0⟩ lands on the south pole.
    assert gui.last_tip[2] < -0.9
    assert not dlg.winfo_exists()
