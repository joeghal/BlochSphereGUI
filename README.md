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
* Scrollable control panel with mouse-wheel support.
* Bloch sphere rendered in 3‑D with tip animation and responsive
  tick marks.
* Every quantum operation runs through a CUDA‑Q kernel for pedagogical
  value.

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

The application window should appear.  Adjust sliders, type state vectors,
and click gate buttons as desired.  Exit the application by closing the
window.

## Development & Packaging

This repo is structured as a small Python package.  The module code
resides in `bloch_gui/bloch_sphere_gui.py`, and `main.py` simply
imports the `BlochSphereGUI` class and runs it.  This makes it easy to
write tests, use the GUI logic from other scripts, or convert the
package into a distributable project later.

To run the unit tests (already containing a few sanity checks):

```sh
pytest
```

## Credits

* **Author**: Joe Ghal – developer and maintainer of this demo.
* **Libraries**: numpy, matplotlib, tkinter, CUDA-Q.

## License

See `LICENSE` (same as the CUDA-Q repository) for details.
