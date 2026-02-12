"""Entry point for the Bloch sphere GUI application."""

import tkinter as tk
from bloch_gui import BlochSphereGUI


def main():
    root = tk.Tk()
    gui = BlochSphereGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
