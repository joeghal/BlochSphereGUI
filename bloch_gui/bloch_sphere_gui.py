import os
import tkinter as tk
from tkinter import Scale, Frame, Label
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

try:
    import cudaq
except Exception:  # pragma: no cover - exercised when the package is absent
    cudaq = None


# Name registered with CUDA-Q for arbitrary single-qubit unitaries.
_CUDAQ_CUSTOM_GATE = "bloch_apply_unitary"

_GPU_TARGET_MARKERS = ("nvidia", "custatevec")


def bloch_coordinates(state):
    """Return Bloch (x, y, z) for a single-qubit state vector."""
    state = np.array(state, dtype=complex)
    norm = np.linalg.norm(state)
    if norm < 1e-15:
        return 0.0, 0.0, 0.0
    state = state / norm
    a0 = state[0]
    a1 = state[1]
    x = 2 * np.real(np.conj(a0) * a1)
    y = 2 * np.imag(np.conj(a0) * a1)
    z = np.abs(a0) ** 2 - np.abs(a1) ** 2
    return float(x), float(y), float(z)


def state_from_slider_angles(nx, ny, nz):
    """State produced by RX(nx·π), RY(ny·π), RZ(nz·π) applied to |0⟩.

    Gate order matches the CUDA-Q slider kernel: later gates act on the left,
    so the unitary is RZ @ RY @ RX.
    """
    a = float(nx) * np.pi
    b = float(ny) * np.pi
    c = float(nz) * np.pi
    cA = np.cos(a / 2.0)
    sA = np.sin(a / 2.0)
    RX = np.array([[cA, -1j * sA], [-1j * sA, cA]], dtype=complex)
    cB = np.cos(b / 2.0)
    sB = np.sin(b / 2.0)
    RY = np.array([[cB, -sB], [sB, cB]], dtype=complex)
    RZ = np.array(
        [[np.exp(-1j * c / 2.0), 0], [0, np.exp(1j * c / 2.0)]],
        dtype=complex,
    )
    vec = RZ @ RY @ RX @ np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
    return vec / np.linalg.norm(vec)


def named_gate_matrix(name):
    """NumPy matrix for a named single-qubit gate, or None if not a matrix gate."""
    name = name.upper()
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    RX90 = np.array(
        [
            [np.cos(np.pi / 4), -1j * np.sin(np.pi / 4)],
            [-1j * np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ],
        dtype=complex,
    )
    RY90 = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ],
        dtype=complex,
    )
    RZ90 = np.array(
        [[np.exp(-1j * np.pi / 4), 0], [0, np.exp(1j * np.pi / 4)]],
        dtype=complex,
    )
    return {
        "H": H,
        "X": X,
        "Y": Y,
        "Z": Z,
        "S": S,
        "T": T,
        "RX90": RX90,
        "RY90": RY90,
        "RZ90": RZ90,
    }.get(name)


def _target_needs_gpu(target) -> bool:
    name = str(getattr(target, "name", "") or "").lower()
    simulator = str(getattr(target, "simulator", "") or "").lower()
    blob = f"{name} {simulator}"
    return any(marker in blob for marker in _GPU_TARGET_MARKERS)


def _cudaq_kernel_runs() -> bool:
    """Return True when the active CUDA-Q target can execute a 1-qubit kernel."""
    if cudaq is None:
        return False
    try:
        @cudaq.kernel
        def _ping():
            q = cudaq.qubit()
            h(q)

        cudaq.get_state(_ping)
        return True
    except Exception:
        return False


def select_cudaq_target():
    """Choose a CUDA-Q target that can run kernels.

    An explicit ``CUDAQ_DEFAULT_SIMULATOR`` is tried first. When the active
    target needs a GPU and none is usable (or a kernel fails), fall back to
    the ``qpp-cpu`` simulator. Returns the target name, or None when CUDA-Q
    cannot execute.
    """
    if cudaq is None:
        return None

    requested = os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "").strip()
    if requested:
        try:
            cudaq.set_target(requested)
        except Exception:
            pass

    def _usable(target_name_hint=None):
        try:
            target = cudaq.get_target()
        except Exception:
            return False
        if target_name_hint and getattr(target, "name", None) != target_name_hint:
            return False
        gpus = 0
        try:
            gpus = int(cudaq.num_available_gpus())
        except Exception:
            gpus = 0
        if gpus <= 0 and _target_needs_gpu(target):
            return False
        return _cudaq_kernel_runs()

    if _usable():
        try:
            return cudaq.get_target().name
        except Exception:
            return requested or "qpp-cpu"

    try:
        cudaq.set_target("qpp-cpu")
    except Exception:
        return None
    if _usable("qpp-cpu") or _cudaq_kernel_runs():
        try:
            return cudaq.get_target().name
        except Exception:
            return "qpp-cpu"
    return None


class BlochSphereGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bloch Sphere Rotations - CUDA-Q")
        # Tall enough that the state-information block (including Bloch y and z)
        # sits inside the window instead of being clipped by the control canvas.
        self.root.geometry("900x700")
        
        # Create main frame
        main_frame = Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a scrollable control panel on the left using a Canvas
        control_canvas = tk.Canvas(main_frame, width=360)
        control_canvas.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        self.control_canvas = control_canvas

        # Vertical slider to show there is more content and to control scrolling
        def _on_menu_slider(val):
            try:
                # move to fraction of the true scrollable range
                try:
                    visible = control_canvas.yview()[1] - control_canvas.yview()[0]
                    max_scroll = max(0.0, 1.0 - visible)
                except Exception:
                    max_scroll = 1.0
                control_canvas.yview_moveto((float(val) / 100.0) * max_scroll)
            except Exception:
                pass

        self.menu_slider = Scale(main_frame, from_=0, to=100, orient=tk.VERTICAL, command=_on_menu_slider, length=450)
        self.menu_slider.set(0)
        self.menu_slider.pack(side=tk.LEFT, fill=tk.Y, padx=(0,6))

        control_frame = Frame(control_canvas)
        control_window = control_canvas.create_window((0, 0), window=control_frame, anchor='nw')
        self.control_window = control_window

        # Keep the canvas scrollregion up to date when the inner frame changes.
        # Also stretch the embedded frame to the canvas width so trough clicks
        # land on the sliders instead of the canvas behind them.
        def _on_control_config(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        control_frame.bind("<Configure>", _on_control_config)

        def _on_canvas_configure(event):
            if event.width > 1:
                control_canvas.itemconfig(control_window, width=event.width)
        control_canvas.bind("<Configure>", _on_canvas_configure)
        
        # Mouse wheel support: bind when cursor is over the control canvas
        def _on_mousewheel(event):
            # X11 uses Button-4/5, modern tk on Windows uses delta.
            # Button-1 trough clicks are handled on the scales themselves so
            # this scroll binding cannot swallow a drag.
            try:
                if hasattr(event, 'num') and event.num in (4,):
                    control_canvas.yview_scroll(-1, 'units')
                elif hasattr(event, 'num') and event.num in (5,):
                    control_canvas.yview_scroll(1, 'units')
                else:
                    # event.delta > 0 means up
                    if event.delta > 0:
                        control_canvas.yview_scroll(-1, 'units')
                    else:
                        control_canvas.yview_scroll(1, 'units')
            except Exception:
                pass
            # Keep the slider in sync with true scrollable range
            try:
                v0, v1 = control_canvas.yview()
                visible = v1 - v0
                max_scroll = max(0.0, 1.0 - visible)
                if max_scroll > 0:
                    frac = v0 / max_scroll
                else:
                    frac = 0.0
                self.menu_slider.set(int(round(frac * 100)))
            except Exception:
                pass

        def _bind_mousewheel(_):
            control_canvas.bind_all('<Button-4>', _on_mousewheel)
            control_canvas.bind_all('<Button-5>', _on_mousewheel)
            control_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        def _unbind_mousewheel(_):
            control_canvas.unbind_all('<Button-4>')
            control_canvas.unbind_all('<Button-5>')
            control_canvas.unbind_all('<MouseWheel>')

        control_canvas.bind('<Enter>', _bind_mousewheel)
        control_canvas.bind('<Leave>', _unbind_mousewheel)
        
        # Title
        title_label = Label(control_frame, text="Bloch Sphere Control", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Rotation sliders
        Label(control_frame, text="X-axis Rotation (RX)", font=("Arial", 11)).pack(anchor=tk.W, pady=(15, 5))
        slider_frame_x = Frame(control_frame)
        slider_frame_x.pack(fill=tk.X)
        self.slider_x = Scale(slider_frame_x, from_=0, to=2, resolution=0.01, orient=tk.HORIZONTAL, 
                             command=self.update_state, length=220)
        self.slider_x.set(0)
        self.slider_x.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_x = tk.Entry(slider_frame_x, width=6, font=("Arial", 10))
        self.entry_x.insert(0, "0.00")
        self.entry_x.pack(side=tk.LEFT, padx=(5, 0))
        self.label_x = Label(control_frame, text="nx = 0.00 (0 to 2)", font=("Arial", 9))
        self.label_x.pack(anchor=tk.W)
        
        Label(control_frame, text="Y-axis Rotation (RY)", font=("Arial", 11)).pack(anchor=tk.W, pady=(15, 5))
        slider_frame_y = Frame(control_frame)
        slider_frame_y.pack(fill=tk.X)
        self.slider_y = Scale(slider_frame_y, from_=0, to=2, resolution=0.01, orient=tk.HORIZONTAL, 
                             command=self.update_state, length=220)
        self.slider_y.set(0)
        self.slider_y.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_y = tk.Entry(slider_frame_y, width=6, font=("Arial", 10))
        self.entry_y.insert(0, "0.00")
        self.entry_y.pack(side=tk.LEFT, padx=(5, 0))
        self.label_y = Label(control_frame, text="ny = 0.00 (0 to 2)", font=("Arial", 9))
        self.label_y.pack(anchor=tk.W)
        
        Label(control_frame, text="Z-axis Rotation (RZ)", font=("Arial", 11)).pack(anchor=tk.W, pady=(15, 5))
        slider_frame_z = Frame(control_frame)
        slider_frame_z.pack(fill=tk.X)
        self.slider_z = Scale(slider_frame_z, from_=0, to=2, resolution=0.01, orient=tk.HORIZONTAL, 
                             command=self.update_state, length=220)
        self.slider_z.set(0)
        self.slider_z.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry_z = tk.Entry(slider_frame_z, width=6, font=("Arial", 10))
        self.entry_z.insert(0, "0.00")
        self.entry_z.pack(side=tk.LEFT, padx=(5, 0))
        self.label_z = Label(control_frame, text="nz = 0.00 (0 to 2)", font=("Arial", 9))
        self.label_z.pack(anchor=tk.W)

        # Trough clicks and drags must move the handle. Inside a canvas window
        # item, Tk's default trough behavior only steps by `resolution` (0.01,
        # about one pixel here) and does not drag unless the press starts on
        # the slider element. That reads as a dead slider, especially once the
        # CUDA-Q callback is slow enough to cancel auto-repeat.
        self._enable_slider_interaction(self.slider_x)
        self._enable_slider_interaction(self.slider_y)
        self._enable_slider_interaction(self.slider_z)
        
        # Apply Angles button
        apply_angles_button = tk.Button(control_frame, text="Apply Angles", command=self.apply_angles)
        apply_angles_button.pack(pady=(5, 15))
        
        self.angles_error_label = Label(control_frame, text="", font=("Arial", 9), fg="red")
        self.angles_error_label.pack(anchor=tk.W)
        
        # State information
        info_frame = Frame(control_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        Label(info_frame, text="State Information", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        # Tall enough for the state vector plus Bloch x, y, and z. Height is
        # refit to the text after every update so the last coordinates stay visible.
        self.state_text = tk.Text(info_frame, height=10, width=40, font=("Courier", 9))
        self.state_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # State vector input section
        Label(control_frame, text="Input State Vector", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(15, 5))
        Label(control_frame, text="|ψ⟩ = a₀|0⟩ + a₁|1⟩", font=("Arial", 9)).pack(anchor=tk.W)
        
        Label(control_frame, text="a₀ (complex):", font=("Arial", 9)).pack(anchor=tk.W, pady=(5, 2))
        self.entry_a0 = tk.Entry(control_frame, width=30, font=("Arial", 9))
        self.entry_a0.insert(0, "1.0")
        self.entry_a0.pack(fill=tk.X, pady=(0, 5))
        
        Label(control_frame, text="a₁ (complex):", font=("Arial", 9)).pack(anchor=tk.W, pady=(5, 2))
        self.entry_a1 = tk.Entry(control_frame, width=30, font=("Arial", 9))
        self.entry_a1.insert(0, "0.0")
        self.entry_a1.pack(fill=tk.X, pady=(0, 5))
        
        apply_button = tk.Button(control_frame, text="Apply State Vector", command=self.apply_state_vector)
        apply_button.pack(pady=5)
        
        self.state_error_label = Label(control_frame, text="", font=("Arial", 9), fg="red")
        self.state_error_label.pack(anchor=tk.W)

        # Gate buttons (rows of up to 3 so sphere area stays large)
        Label(control_frame, text="Apply Single-Qubit Gate:", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(10, 4))
        gates = ["H", "X", "Y", "Z", "S", "T", "RX90", "RY90", "RZ90", "Reset", "Random"]
        # create rows of 3 buttons (compact spacing)
        for i in range(0, len(gates), 3):
            row = Frame(control_frame)
            row.pack(fill=tk.X, pady=1)
            for g in gates[i:i+3]:
                btn = tk.Button(row, text=g, width=7, font=("Arial", 9), command=lambda name=g: self.apply_named_gate(name))
                btn.pack(side=tk.LEFT, padx=3)
        # Arbitrary rotation button
        arb_btn = tk.Button(control_frame, text="Arbitrary Rotation...", command=self.open_arbitrary_rotation_dialog)
        arb_btn.pack(pady=(6, 8), anchor=tk.W)
        
        # Right side: Matplotlib figure for Bloch sphere
        canvas_frame = Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])  # Make sphere look like a sphere
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Cache for plot elements
        self.sphere_plot = None
        self.axis_plots = []
        self.state_vector_plot = None
        self.state_tip_plot = None
        self.last_tip = (0.0, 0.0, 1.0)
        self._evolution_backend = None
        self.cudaq_target = select_cudaq_target()
        self._cudaq_ready = self.cudaq_target is not None
        
        # Initial state
        self.update_state()
    
    def _enable_slider_interaction(self, scale):
        """Make a rotation slider jump to trough clicks and follow drags.

        Widget bindings run before the Scale class bindings and before
        ``bind_all`` canvas scrolling. Returning ``"break"`` keeps those
        handlers from stepping by one resolution tick or scrolling the panel
        instead of moving the handle. ``grab_set`` keeps the drag alive if
        the pointer slides off the thin trough.
        """

        def _move(event):
            try:
                raw = scale.tk.call(scale._w, "get", event.x, event.y)
                scale.set(float(raw))
            except tk.TclError:
                return "break"
            except Exception:
                return "break"
            return "break"

        def _press(event):
            try:
                scale.focus_set()
                scale.grab_set()
            except tk.TclError:
                pass
            return _move(event)

        def _release(event):
            try:
                scale.grab_release()
            except tk.TclError:
                pass
            return _move(event)

        scale.bind("<Button-1>", _press)
        scale.bind("<B1-Motion>", _move)
        scale.bind("<ButtonRelease-1>", _release)

    def _set_slider_value(self, scale, value):
        """Set a scale without running its command callback."""
        command = scale.cget("command")
        scale.configure(command="")
        try:
            scale.set(value)
        finally:
            if command:
                scale.configure(command=command)

    def get_state_vector(self, nx, ny, nz):
        """
        Compute the state vector after applying RX, RY, RZ rotations to |0⟩ state.
        Rotations are applied in order: RX(nx*π), RY(ny*π), RZ(nz*π)
        """
        if self._cudaq_ready and cudaq is not None:
            try:
                @cudaq.kernel
                def rotated_state():
                    qubit = cudaq.qubit()
                    # Apply rotations in order: X, Y, Z
                    rx(nx * np.pi, qubit)
                    ry(ny * np.pi, qubit)
                    rz(nz * np.pi, qubit)

                self._evolution_backend = "cudaq"
                return cudaq.get_state(rotated_state)
            except Exception:
                self._cudaq_ready = False
        self._evolution_backend = "numpy"
        return state_from_slider_angles(nx, ny, nz)

    def get_current_state_array(self):
        """Return current state as numpy array [a0, a1].
        Preference: if entries `entry_a0` and `entry_a1` are non-empty and parseable, use them.
        Otherwise compute using sliders via `get_state_vector`.
        """
        # Try parse entries first
        a0_s = self.entry_a0.get().strip()
        a1_s = self.entry_a1.get().strip()
        if a0_s != "" and a1_s != "":
            try:
                a0 = complex(a0_s)
                a1 = complex(a1_s)
                arr = np.array([a0, a1], dtype=complex)
                norm = np.linalg.norm(arr)
                if norm > 1e-8:
                    return arr / norm
            except Exception:
                pass

        # Fallback: compute from sliders
        try:
            nx = self.slider_x.get()
            ny = self.slider_y.get()
            nz = self.slider_z.get()
            state = self.get_state_vector(nx, ny, nz)
            return np.array(state, dtype=complex)
        except Exception:
            # final fallback: |0>
            return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)

    def format_complex(self, z, prec=4):
        """Return a Python-parsable complex string with given precision."""
        z = complex(z)
        re = z.real
        im = z.imag
        if abs(im) < 10**(-prec):
            return f"{re:.{prec}f}"
        sign = '+' if im >= 0 else '-'
        return f"({re:.{prec}f}{sign}{abs(im):.{prec}f}j)"

    def _clear_error_labels(self):
        """Drop red validation messages after a successful operation."""
        self.state_error_label.config(text="")
        self.angles_error_label.config(text="")

    def _set_state_info(self, content):
        """Replace the state-information text and grow it to fit every line.

        The Bloch y and z rows sit at the bottom of the block. A fixed
        six-line widget clips them, so the height tracks the displayed text.
        """
        if not content.endswith("\n"):
            content += "\n"
        self.state_text.delete("1.0", tk.END)
        self.state_text.insert(tk.END, content)
        width = max(1, int(float(self.state_text.cget("width"))))
        visual = 0
        for line in content.splitlines():
            visual += max(1, (len(line) + width - 1) // width)
        last_line = int(float(self.state_text.index("end-1c").split(".")[0]))
        # One spare row keeps the last coordinate clear of the text border.
        self.state_text.configure(height=max(visual, last_line) + 1)
        try:
            self.state_text.see("1.0")
        except tk.TclError:
            pass

    def _state_for_gate(self):
        """Current state for a gate.

        Explicit a0 and a1 entries win when both are filled. Invalid or zero
        entries set the state error label and return None. Empty entries use
        the slider angles (a CUDA-Q kernel when that target is available).
        """
        a0_s = self.entry_a0.get().strip()
        a1_s = self.entry_a1.get().strip()
        if a0_s != "" and a1_s != "":
            try:
                state = np.array([complex(a0_s), complex(a1_s)], dtype=complex)
            except Exception:
                self.state_error_label.config(text="Invalid complex numbers in entries")
                return None
            norm = np.linalg.norm(state)
            if norm < 1e-12:
                self.state_error_label.config(text="State vector cannot be zero")
                return None
            return state / norm
        return self._state_from_sliders()

    def _state_from_sliders(self):
        try:
            state = self.get_state_vector(
                self.slider_x.get(), self.slider_y.get(), self.slider_z.get()
            )
            arr = np.array(state, dtype=complex)
            norm = np.linalg.norm(arr)
            if norm < 1e-12:
                raise ValueError("zero state")
            return arr / norm
        except Exception:
            return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)

    def _haar_unitary(self):
        """Random 2×2 unitary from the Haar measure (via QR)."""
        A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
        Q, R = np.linalg.qr(A)
        diag = np.diag(R)
        lam = np.empty(2, dtype=complex)
        for i, value in enumerate(diag):
            if abs(value) < 1e-12:
                lam[i] = 1.0 + 0j
            else:
                lam[i] = value / abs(value)
        return Q * lam

    def _apply_matrix_numpy(self, state, unitary):
        new_state = np.asarray(unitary, dtype=complex) @ np.asarray(state, dtype=complex)
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm
        return new_state

    def _apply_matrix_cudaq(self, state, unitary):
        """Apply a 2×2 unitary with a CUDA-Q kernel initialized from ``state``."""
        if not self._cudaq_ready or cudaq is None:
            raise RuntimeError("CUDA-Q unavailable")
        matrix = np.asarray(unitary, dtype=complex)
        cudaq.register_operation(_CUDAQ_CUSTOM_GATE, matrix)
        a0 = complex(state[0])
        a1 = complex(state[1])

        @cudaq.kernel
        def kernel():
            q = cudaq.qvector([a0, a1])
            bloch_apply_unitary(q)

        return np.array(cudaq.get_state(kernel), dtype=complex)

    def _apply_reset_cudaq(self, state):
        """Reset via a CUDA-Q kernel and return canonical |0⟩.

        ``reset`` on a state-initialized qubit can keep a global phase.
        The displayed amplitudes are the canonical |0⟩ vector.
        """
        if not self._cudaq_ready or cudaq is None:
            raise RuntimeError("CUDA-Q unavailable")
        a0 = complex(state[0])
        a1 = complex(state[1])

        @cudaq.kernel
        def kernel():
            q = cudaq.qvector([a0, a1])
            reset(q)

        result = np.array(cudaq.get_state(kernel), dtype=complex)
        norm = np.linalg.norm(result)
        if norm < 1e-8:
            raise RuntimeError("CUDA-Q reset returned an empty state")
        result = result / norm
        if abs(result[1]) > 1e-6:
            raise RuntimeError("CUDA-Q reset did not return |0>")
        return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)

    def _apply_named_cudaq(self, state, name):
        """Apply one built-in gate with a CUDA-Q kernel starting from ``state``."""
        if not self._cudaq_ready or cudaq is None:
            raise RuntimeError("CUDA-Q unavailable")
        a0 = complex(state[0])
        a1 = complex(state[1])
        name = name.upper()

        if name == "H":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                h(q)
        elif name == "X":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                x(q)
        elif name == "Y":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                y(q)
        elif name == "Z":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                z(q)
        elif name == "S":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                s(q)
        elif name == "T":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                t(q)
        elif name == "RX90":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                rx(np.pi / 2, q)
        elif name == "RY90":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                ry(np.pi / 2, q)
        elif name == "RZ90":
            @cudaq.kernel
            def kernel():
                q = cudaq.qvector([a0, a1])
                rz(np.pi / 2, q)
        else:
            raise ValueError(f"No CUDA-Q kernel for gate {name}")

        return np.array(cudaq.get_state(kernel), dtype=complex)

    def _apply_named_numpy(self, state, name):
        """NumPy fallback used only when a CUDA-Q kernel cannot run."""
        name = name.upper()
        if name == "RESET":
            return np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
        if name == "RANDOM":
            return self._apply_matrix_numpy(state, self._haar_unitary())
        gate = named_gate_matrix(name)
        if gate is None:
            return None
        return self._apply_matrix_numpy(state, gate)

    def _run_evolution(self, state, cudaq_fn, numpy_fn):
        """Run ``cudaq_fn`` and fall back to ``numpy_fn`` if CUDA-Q cannot."""
        if self._cudaq_ready and cudaq is not None:
            try:
                result = cudaq_fn()
                self._evolution_backend = "cudaq"
                return result
            except Exception:
                pass
        result = numpy_fn()
        if result is None:
            return None
        self._evolution_backend = "numpy"
        return result

    def _finalize_state(self, new_state):
        """Common routine for displaying a new normalized state vector."""
        norm = np.linalg.norm(new_state)
        if norm > 1e-12:
            new_state = new_state / norm

        # Update entries
        self.entry_a0.delete(0, tk.END)
        self.entry_a0.insert(0, self.format_complex(new_state[0], 6))
        self.entry_a1.delete(0, tk.END)
        self.entry_a1.insert(0, self.format_complex(new_state[1], 6))
        self._clear_error_labels()

        # Sync sliders with the new state
        try:
            self.update_sliders_from_state(new_state)
        except Exception:
            pass

        # Update state text and visualization
        x, y, z = self.state_to_bloch(new_state)
        self._set_state_info(
            "State Vector (after gate):\n"
            f"|ψ⟩ = {self.format_complex(new_state[0],4)}|0⟩\n"
            f"    + {self.format_complex(new_state[1],4)}|1⟩\n\n"
            "Bloch Coordinates:\n"
            f"x = {x:.4f}\n"
            f"y = {y:.4f}\n"
            f"z = {z:.4f}\n"
        )

        self.last_tip = (x, y, z)
        self.draw_bloch_sphere(x, y, z)
        try:
            self.animate_tip_highlight(x, y, z)
        except Exception:
            pass

    def apply_unitary(self, G):
        """Apply unitary G (2x2) to the current state and update the UI.

        The state is initialized inside a CUDA-Q kernel (from the amplitude
        entries when they are filled, otherwise from the slider angles) and
        G is applied in that kernel. NumPy is used only when CUDA-Q cannot run.
        """
        state = self._state_for_gate()
        if state is None:
            return
        G = np.asarray(G, dtype=complex)

        def _cudaq_apply():
            return self._apply_matrix_cudaq(state, G)

        def _numpy_apply():
            return self._apply_matrix_numpy(state, G)

        try:
            new_state = self._run_evolution(state, _cudaq_apply, _numpy_apply)
        except Exception as ex:
            self.state_error_label.config(text=str(ex))
            return
        if new_state is None:
            return
        backend = self._evolution_backend
        self._finalize_state(new_state)
        # Slider sync inside finalize also runs a kernel. Keep the backend
        # that actually applied this unitary.
        self._evolution_backend = backend

    def open_arbitrary_rotation_dialog(self):
        """Open a dialog letting the user choose an axis (X/Y/Z/Custom) and angle (multiples of π)."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Arbitrary Rotation")

        Label(dlg, text="Axis:").grid(row=0, column=0, sticky=tk.W, padx=6, pady=6)
        axis_var = tk.StringVar(dlg)
        axis_var.set("X")
        axis_menu = tk.OptionMenu(dlg, axis_var, "X", "Y", "Z", "Custom")
        axis_menu.grid(row=0, column=1, sticky=tk.W, padx=6, pady=6)

        Label(dlg, text="Angle (multiples of π):").grid(row=1, column=0, sticky=tk.W, padx=6, pady=6)
        angle_entry = tk.Entry(dlg)
        angle_entry.insert(0, "1.00")
        angle_entry.grid(row=1, column=1, sticky=tk.W, padx=6, pady=6)

        # Custom axis inputs
        custom_frame = Frame(dlg)
        custom_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=6, pady=6)
        Label(custom_frame, text="Custom axis (nx, ny, nz):").pack(side=tk.LEFT)
        nx_e = tk.Entry(custom_frame, width=6)
        ny_e = tk.Entry(custom_frame, width=6)
        nz_e = tk.Entry(custom_frame, width=6)
        nx_e.pack(side=tk.LEFT, padx=(6,2))
        ny_e.pack(side=tk.LEFT, padx=2)
        nz_e.pack(side=tk.LEFT, padx=2)

        def apply_dialog():
            try:
                angle_mult = float(angle_entry.get())
                theta = angle_mult * np.pi
                axis = axis_var.get()
                # Pauli matrices
                X = np.array([[0, 1], [1, 0]], dtype=complex)
                Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
                Z = np.array([[1, 0], [0, -1]], dtype=complex)

                if axis in ("X", "Y", "Z"):
                    if axis == "X":
                        n = np.array([1.0, 0.0, 0.0])
                    elif axis == "Y":
                        n = np.array([0.0, 1.0, 0.0])
                    else:
                        n = np.array([0.0, 0.0, 1.0])
                else:
                    # parse custom
                    nxv = float(nx_e.get() or 0.0)
                    nyv = float(ny_e.get() or 0.0)
                    nzv = float(nz_e.get() or 0.0)
                    n = np.array([nxv, nyv, nzv], dtype=float)
                    norm = np.linalg.norm(n)
                    if norm < 1e-8:
                        self.state_error_label.config(text="Custom axis cannot be zero")
                        return
                    n = n / norm

                # Build rotation: U = cos(theta/2) I - i sin(theta/2) (n·σ)
                n_x, n_y, n_z = n
                n_dot_sigma = n_x * X + n_y * Y + n_z * Z
                U = np.cos(theta/2.0) * np.eye(2, dtype=complex) - 1j * np.sin(theta/2.0) * n_dot_sigma
                self.apply_unitary(U)
                dlg.destroy()
            except Exception as ex:
                self.state_error_label.config(text=str(ex))

        apply_btn = tk.Button(dlg, text="Apply", command=apply_dialog)
        apply_btn.grid(row=3, column=0, columnspan=2, pady=(8,10))

    def apply_named_gate(self, name: str):
        """Apply a named single-qubit gate and update the UI.

        The current state (amplitude entries when both are filled, otherwise
        the slider angles) is loaded into a CUDA-Q kernel and the gate runs
        there. Reset always returns |0⟩ and refills the amplitude boxes.
        NumPy is used only when CUDA-Q is unavailable or the kernel fails.
        """
        name = name.upper()
        if name not in ("RESET", "RANDOM") and named_gate_matrix(name) is None:
            return

        state = self._state_for_gate()
        if state is None:
            return

        def _cudaq_apply():
            if name == "RESET":
                return self._apply_reset_cudaq(state)
            if name == "RANDOM":
                return self._apply_matrix_cudaq(state, self._haar_unitary())
            return self._apply_named_cudaq(state, name)

        def _numpy_apply():
            return self._apply_named_numpy(state, name)

        try:
            new_state = self._run_evolution(state, _cudaq_apply, _numpy_apply)
        except Exception as ex:
            self.state_error_label.config(text=str(ex))
            return
        if new_state is None:
            return
        backend = self._evolution_backend
        self._finalize_state(new_state)
        self._evolution_backend = backend
    
    def state_to_bloch(self, state):
        """
        Convert a single-qubit state to Bloch sphere coordinates.
        state is a numpy array [a0, a1] representing a0|0⟩ + a1|1⟩
        """
        return bloch_coordinates(state)

    def update_sliders_from_state(self, state):
        """Estimate RX/RY/RZ angles (in slider units where value*pi is angle)
        that produce `state` when applied in order RX then RY then RZ to |0>.

        Approach:
        - Construct a SU(2) unitary U whose first column is `state`.
        - Convert U to a unit quaternion and then to a 3x3 rotation matrix R.
        - Extract Z-Y-X Euler angles (gamma, beta, alpha) from R where
          R = Rz(gamma) * Ry(beta) * Rx(alpha).
        - Map alpha,beta,gamma into slider multiples of pi in [0,2).
        """
        try:
            s = np.array(state, dtype=complex)
            norm = np.linalg.norm(s)
            if norm < 1e-12:
                return
            s = s / norm

            a0 = s[0]
            a1 = s[1]

            # Build an SU(2) matrix U whose first column is the state
            U = np.array([[a0, -np.conj(a1)], [a1, np.conj(a0)]], dtype=complex)

            # Extract quaternion components (w,x,y,z) from U using SU(2) -> quaternion mapping
            # For U = [[alpha, -beta*],[beta, alpha*]] with alpha=a0, beta=a1:
            # w = Re(alpha), z = Im(alpha), y = -Re(beta), x = Im(beta)
            alpha = a0
            beta = a1
            w = float(np.real(alpha))
            z = float(np.imag(alpha))
            y = -float(np.real(beta))
            x = float(np.imag(beta))

            # Normalize quaternion to avoid numerical issues
            qnorm = np.sqrt(w*w + x*x + y*y + z*z)
            if qnorm < 1e-12:
                return
            w, x, y, z = w/qnorm, x/qnorm, y/qnorm, z/qnorm

            # Convert quaternion to rotation matrix R (SO(3))
            R = np.zeros((3,3), dtype=float)
            R[0,0] = 1 - 2*(y*y + z*z)
            R[0,1] = 2*(x*y - w*z)
            R[0,2] = 2*(x*z + w*y)
            R[1,0] = 2*(x*y + w*z)
            R[1,1] = 1 - 2*(x*x + z*z)
            R[1,2] = 2*(y*z - w*x)
            R[2,0] = 2*(x*z - w*y)
            R[2,1] = 2*(y*z + w*x)
            R[2,2] = 1 - 2*(x*x + y*y)

            # Extract Z-Y-X Euler angles (gamma, beta, alpha) from R
            # beta = asin(-R[2,0])
            # alpha = atan2(R[2,1], R[2,2])
            # gamma = atan2(R[1,0], R[0,0])
            # Handle numerical edge cases
            r20 = np.clip(R[2,0], -1.0, 1.0)
            beta = np.arcsin(-r20)
            cosb = np.cos(beta)
            if abs(cosb) > 1e-6:
                alpha = np.arctan2(R[2,1], R[2,2])
                gamma = np.arctan2(R[1,0], R[0,0])
            else:
                # Gimbal lock; set alpha = 0 and compute gamma differently
                alpha = 0.0
                gamma = np.arctan2(-R[0,1], R[1,1])

            # Map to [0, 2*pi)
            alpha = float(alpha % (2*np.pi))
            beta = float(beta % (2*np.pi))
            gamma = float(gamma % (2*np.pi))

            nx_val = float(alpha / np.pi)
            ny_val = float(beta / np.pi)
            nz_val = float(gamma / np.pi)

            # Bring into [0,2)
            nx_val = (nx_val + 2.0) % 2.0
            ny_val = (ny_val + 2.0) % 2.0
            nz_val = (nz_val + 2.0) % 2.0

            # Verify analytic guess by generating the state from these angles
            def gen_state_from_angles(nx_v, ny_v, nz_v):
                # angles in radians
                a = nx_v * np.pi
                b = ny_v * np.pi
                c = nz_v * np.pi
                # RX(a)
                cA = np.cos(a/2.0)
                sA = np.sin(a/2.0)
                RX = np.array([[cA, -1j*sA], [-1j*sA, cA]], dtype=complex)
                # RY(b)
                cB = np.cos(b/2.0)
                sB = np.sin(b/2.0)
                RY = np.array([[cB, -sB], [sB, cB]], dtype=complex)
                # RZ(c)
                RZ = np.array([[np.exp(-1j*c/2.0), 0], [0, np.exp(1j*c/2.0)]], dtype=complex)
                U = RZ @ RY @ RX
                vec = U @ np.array([1.0 + 0j, 0.0 + 0j], dtype=complex)
                # normalize
                return vec / np.linalg.norm(vec)

            def fidelity(s1, s2):
                return abs(np.vdot(s1, s2))

            cand = gen_state_from_angles(nx_val, ny_val, nz_val)
            fid = fidelity(s, cand)

            if fid < 0.999999:
                # perform a small local search (hill-climb) around the guess to maximize fidelity
                def find_angles_for_state(target, init_guess=(nx_val, ny_val, nz_val)):
                    tx = float(init_guess[0])
                    ty = float(init_guess[1])
                    tz = float(init_guess[2])
                    best = (tx, ty, tz)
                    best_f = fidelity(target, gen_state_from_angles(*best))
                    steps = [0.5, 0.12, 0.03, 0.01]
                    for step in steps:
                        improved = True
                        while improved:
                            improved = False
                            for dx in (-1, 0, 1):
                                for dy in (-1, 0, 1):
                                    for dz in (-1, 0, 1):
                                        cand_angles = (best[0] + dx * step, best[1] + dy * step, best[2] + dz * step)
                                        # wrap into [0,2)
                                        cand_angles = tuple(((np.array(cand_angles) + 2.0) % 2.0).tolist())
                                        f = fidelity(target, gen_state_from_angles(*cand_angles))
                                        if f > best_f + 1e-9:
                                            best_f = f
                                            best = cand_angles
                                            improved = True
                        # continue to next finer step
                    return best, best_f

                try:
                    (nx_val, ny_val, nz_val), best_f = find_angles_for_state(s, (nx_val, ny_val, nz_val))
                except Exception:
                    pass

            # Apply to sliders and entry fields. Silence the scale command so
            # the idle redraw does not replace the state text the caller just
            # computed (slider.set schedules that callback, and the rebuilt
            # angles can differ by a global phase).
            try:
                self._set_slider_value(self.slider_x, nx_val)
                self._set_slider_value(self.slider_y, ny_val)
                self._set_slider_value(self.slider_z, nz_val)
            except Exception:
                pass

            self.entry_x.delete(0, tk.END)
            self.entry_x.insert(0, f"{nx_val:.2f}")
            self.entry_y.delete(0, tk.END)
            self.entry_y.insert(0, f"{ny_val:.2f}")
            self.entry_z.delete(0, tk.END)
            self.entry_z.insert(0, f"{nz_val:.2f}")

            self.label_x.config(text=f"nx = {nx_val:.2f} (0 to 2)")
            self.label_y.config(text=f"ny = {ny_val:.2f} (0 to 2)")
            self.label_z.config(text=f"nz = {nz_val:.2f} (0 to 2)")
        except Exception:
            # best-effort; do not crash GUI
            return
    
    def draw_bloch_sphere(self, x, y, z):
        """Draw the Bloch sphere with the state vector"""
        # Only initialize sphere once
        if self.sphere_plot is None:
            # Draw the Bloch sphere with higher resolution for better appearance
            u = np.linspace(0, 2 * np.pi, 80)
            v = np.linspace(0, np.pi, 80)
            sphere_x = np.outer(np.cos(u), np.sin(v))
            sphere_y = np.outer(np.sin(u), np.sin(v))
            sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Plot with both facecolor and edgecolor for better 3D appearance
            self.sphere_plot = self.ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                                                     alpha=0.25, color='cyan', 
                                                     edgecolor='gray', linewidth=0.3)
            
            # Draw axes
            axis_length = 1.3
            self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
            self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
            self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
            
            # Set limits and labels
            self.ax.set_xlim([-1.1, 1.1])
            self.ax.set_ylim([-1.1, 1.1])
            self.ax.set_zlim([-1.1, 1.1])
            self.ax.set_box_aspect([1,1,1])
            
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Bloch Sphere')
        
        # Update the state vector
        if self.state_vector_plot is not None:
            self.state_vector_plot.remove()
        
        # Remove previous tip marker if present
        if self.state_tip_plot is not None:
            try:
                self.state_tip_plot.remove()
            except Exception:
                pass
            self.state_tip_plot = None

        magnitude = np.sqrt(x**2 + y**2 + z**2)
        if magnitude > 0.01:  # Only draw if non-negligible
            self.state_vector_plot = self.ax.quiver(0, 0, 0, x, y, z, color='black', arrow_length_ratio=0.15, linewidth=3, label='State')

            # Determine if the tip is facing the camera. Use axes elevation/azim to compute view vector.
            try:
                elev_rad = np.deg2rad(self.ax.elev)
                azim_rad = np.deg2rad(self.ax.azim)
                view_vec = np.array([
                    np.cos(elev_rad) * np.cos(azim_rad),
                    np.cos(elev_rad) * np.sin(azim_rad),
                    np.sin(elev_rad)
                ])
                tip_vec = np.array([x, y, z])
                facing = float(np.dot(tip_vec, view_vec)) > 0
            except Exception:
                # If any issue computing view, assume facing
                facing = True

            tip_alpha = 1.0 if facing else 0.25
            # Plot a small dot at the tip of the state vector. Use slight edge for visibility.
            self.state_tip_plot = self.ax.scatter([x], [y], [z], color='red', s=60, alpha=tip_alpha, edgecolors='k', linewidths=0.4)

        # Adjust axis tick density based on canvas size to avoid overlap when small
        try:
            widget = self.canvas.get_tk_widget()
            w = widget.winfo_width() if widget.winfo_width() > 0 else None
            h = widget.winfo_height() if widget.winfo_height() > 0 else None
        except Exception:
            w = h = None

        # choose ticks: fewer ticks for small displays
        try:
            if (w is not None and min(w, h) < 420) or (self.fig.bbox.width < 500):
                ticks = [-1.0, 0.0, 1.0]
            else:
                ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
            self.ax.set_xticks(ticks)
            self.ax.set_yticks(ticks)
            self.ax.set_zticks(ticks)
        except Exception:
            pass

        self.canvas.draw_idle()
    
    def update_state(self, value=None):
        """Update the quantum state and visualization"""
        # Get slider values
        nx = self.slider_x.get()
        ny = self.slider_y.get()
        nz = self.slider_z.get()
        
        # Update entry fields
        self.entry_x.delete(0, tk.END)
        self.entry_x.insert(0, f"{nx:.2f}")
        self.entry_y.delete(0, tk.END)
        self.entry_y.insert(0, f"{ny:.2f}")
        self.entry_z.delete(0, tk.END)
        self.entry_z.insert(0, f"{nz:.2f}")
        
        # Update labels
        self.label_x.config(text=f"nx = {nx:.2f} (0 to 2)")
        self.label_y.config(text=f"ny = {ny:.2f} (0 to 2)")
        self.label_z.config(text=f"nz = {nz:.2f} (0 to 2)")
        
        # Compute state vector
        try:
            state = self.get_state_vector(nx, ny, nz)
            state_array = np.array(state)
            
            # Extract Bloch coordinates
            x, y, z = self.state_to_bloch(state_array)
            # remember last tip for other actions
            self.last_tip = (x, y, z)

            self._set_state_info(
                "State Vector:\n"
                f"|ψ⟩ = {self.format_complex(state_array[0],4)}|0⟩\n"
                f"    + {self.format_complex(state_array[1],4)}|1⟩\n\n"
                "Bloch Coordinates:\n"
                f"x = {x:.4f}\n"
                f"y = {y:.4f}\n"
                f"z = {z:.4f}\n"
            )
            
            # Draw Bloch sphere
            self.draw_bloch_sphere(x, y, z)
            self._clear_error_labels()
            
            return
            
        except Exception as e:
            self._set_state_info(f"Error: {str(e)}")
    
    def apply_angles(self):
        """Apply user-entered rotation angles"""
        try:
            # Parse entries
            nx = float(self.entry_x.get())
            ny = float(self.entry_y.get())
            nz = float(self.entry_z.get())
            
            # Validate ranges
            if not (0 <= nx <= 2):
                self.angles_error_label.config(text="X angle must be between 0 and 2")
                return
            if not (0 <= ny <= 2):
                self.angles_error_label.config(text="Y angle must be between 0 and 2")
                return
            if not (0 <= nz <= 2):
                self.angles_error_label.config(text="Z angle must be between 0 and 2")
                return
            
            # Update sliders
            self.slider_x.set(nx)
            self.slider_y.set(ny)
            self.slider_z.set(nz)
            
            # Clear state vector inputs (since we're applying rotations, not direct state)
            self.entry_a0.delete(0, tk.END)
            self.entry_a0.insert(0, "")
            self.entry_a1.delete(0, tk.END)
            self.entry_a1.insert(0, "")
            
            self._clear_error_labels()
            
            # Trigger state update
            self.update_state()
            # animate tip
            try:
                self.animate_tip_highlight(*self.last_tip)
            except Exception:
                pass
            
        except ValueError:
            self.angles_error_label.config(text="Invalid entry - use decimal numbers")
    
    def apply_state_vector(self):
        """Apply user-entered state vector"""
        try:
            a0_str = self.entry_a0.get().strip()
            a1_str = self.entry_a1.get().strip()
            
            # Parse complex numbers
            a0 = complex(a0_str)
            a1 = complex(a1_str)
            
            state_array = np.array([a0, a1])
            
            # Check normalization
            norm = np.linalg.norm(state_array)
            if norm < 0.001:
                self.state_error_label.config(text="State vector cannot be zero!")
                return
            
            # Normalize state
            state_array = state_array / norm
            
            # Update entry fields with normalized values
            self.entry_a0.delete(0, tk.END)
            self.entry_a0.insert(0, self.format_complex(state_array[0], 6))
            self.entry_a1.delete(0, tk.END)
            self.entry_a1.insert(0, self.format_complex(state_array[1], 6))
            
            # Sync sliders with the input state
            try:
                self.update_sliders_from_state(state_array)
            except Exception:
                pass
            
            # Extract Bloch coordinates
            x, y, z = self.state_to_bloch(state_array)
            # remember last tip
            self.last_tip = (x, y, z)

            self._set_state_info(
                "State Vector (from input):\n"
                f"|ψ⟩ = {self.format_complex(state_array[0],4)}|0⟩\n"
                f"    + {self.format_complex(state_array[1],4)}|1⟩\n\n"
                "Bloch Coordinates:\n"
                f"x = {x:.4f}\n"
                f"y = {y:.4f}\n"
                f"z = {z:.4f}\n"
            )
            
            # Draw Bloch sphere
            self.draw_bloch_sphere(x, y, z)
            # animate tip highlight so user notices new position
            try:
                self.animate_tip_highlight(x, y, z)
            except Exception:
                pass
            self._clear_error_labels()
            
        except ValueError as e:
            self.state_error_label.config(text=f"Invalid complex number format")
        except Exception as e:
            self.state_error_label.config(text=f"Error: {str(e)}")

    def animate_tip_highlight(self, x, y, z, duration=350, frames=6):
        """Briefly pulse the tip marker to highlight it."""
        if x is None or y is None or z is None:
            return

        # remove existing tip if present
        try:
            if self.state_tip_plot is not None:
                self.state_tip_plot.remove()
                self.state_tip_plot = None
        except Exception:
            pass

        base = 60
        grow = np.linspace(base, base * 2.0, frames // 2)
        shrink = np.linspace(base * 2.0, base, frames - frames // 2)
        sizes = np.concatenate([grow, shrink])
        interval = max(1, int(duration / max(1, frames)))

        def step(i):
            if i >= len(sizes):
                return
            try:
                if self.state_tip_plot is not None:
                    self.state_tip_plot.remove()
            except Exception:
                pass

            # recompute facing for alpha
            try:
                elev_rad = np.deg2rad(self.ax.elev)
                azim_rad = np.deg2rad(self.ax.azim)
                view_vec = np.array([
                    np.cos(elev_rad) * np.cos(azim_rad),
                    np.cos(elev_rad) * np.sin(azim_rad),
                    np.sin(elev_rad)
                ])
                tip_vec = np.array([x, y, z])
                facing = float(np.dot(tip_vec, view_vec)) > 0
            except Exception:
                facing = True
            alpha = 1.0 if facing else 0.25

            self.state_tip_plot = self.ax.scatter([x], [y], [z], color='red', s=float(sizes[i]), alpha=alpha, edgecolors='k', linewidths=0.4)
            self.canvas.draw_idle()
            self.root.after(interval, lambda: step(i + 1))

        step(0)


if __name__ == "__main__":
    root = tk.Tk()
    gui = BlochSphereGUI(root)
    root.mainloop()
