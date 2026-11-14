# Bloch Sphere GUI (CUDA-Q)

A simple interactive application for visualizing single-qubit states on
the Bloch sphere.  Users can perform RX/RY/RZ rotations with sliders,
enter arbitrary state vectors, and apply common quantum gates.  The
interface is built with `tkinter` and `matplotlib` while the underlying
quantum evolution is computed using [CUDA-Q](https://github.com/NVIDIA/cuda-quantum).

![Animation placeholder](demo.gif)

## Features

* Three sliders for X‑, Y‑, and Z‑axis rotations (multiples of π).
* Text entries accompany each slider and accept direct input.
* Optional complex state-vector entry (normalization is handled
  automatically).
* Quick buttons for standard gates (H, X, Y, Z, S, T, ±90° rotations,
  reset and random unitary).
* Arbitrary rotation dialog with custom axis.
* Scrollable control panel with mouse-wheel support. Rotation sliders
  jump to a click on the trough and follow a drag.
* Bloch sphere rendered in 3‑D with tip animation and responsive
  tick marks.
* State information (amplitudes and Bloch x, y, z) stays fully visible.
* Quantum evolution runs through a CUDA‑Q kernel. The kernel is
  initialized from the current state, whether that came from the
  sliders or from the a₀/a₁ boxes. Reset always returns |0⟩ and
  refills those boxes. NumPy is used only when CUDA‑Q cannot run.

## Installation

```sh
# create and activate a virtual environment (Linux/macOS)
python -m venv venv
source venv/bin/activate

# install dependencies
pip install -r bloch_sphere_gui_requirements.txt
```

`tkinter` comes with most Python installations; if you see errors
running the GUI you may need to install your OS package (e.g.
`sudo apt install python3-tk`).

## Usage

```sh
python main.py
```

On a machine without a usable GPU, CUDA-Q is pointed at the CPU
simulator (`qpp-cpu`). The app does this itself: if the active target
cannot execute a kernel (for example a GPU target when
`cudaq.num_available_gpus()` is 0), it switches to `qpp-cpu`. You can
force that target before startup:

```sh
CUDAQ_DEFAULT_SIMULATOR=qpp-cpu python main.py
```

The application window should appear.  Adjust sliders, type state vectors,
and click gate buttons as desired.  Exit the application by closing the
window.

Clicking or dragging the rotation-slider trough moves the handle to the
pointer. Gate, Apply Angles, Apply State Vector, Reset, and Random clear
any red validation message left by an earlier bad entry.

## Development & Packaging

This repo is structured as a small Python package.  The module code
resides in `bloch_gui/bloch_sphere_gui.py`, and `main.py` simply
imports the `BlochSphereGUI` class and runs it.  This makes it easy to
write tests, use the GUI logic from other scripts, or convert the
package into a distributable project later.

State-logic tests live in `tests/test_bloch_gui.py`. They drive the
CUDA-Q CPU simulator and, for the widget checks, an X display (Xvfb is
enough):

```sh
CUDAQ_DEFAULT_SIMULATOR=qpp-cpu python -m pytest
# or, without a desktop:
CUDAQ_DEFAULT_SIMULATOR=qpp-cpu xvfb-run -a python -m pytest
```

## Credits

* **Author**: Joe Ghal – developer and maintainer of this demo.
* **Libraries**: numpy, matplotlib, tkinter, CUDA-Q.

## License

See `LICENSE` (same as the CUDA-Q repository) for details.
