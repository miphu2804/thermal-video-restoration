"""Tk display state shared by interactive inference demos."""

from thermal_restore.detection.live import fit_to_window, ppm_bytes


class LiveWindow:
    """Own a Tk window and its pause/close state, without owning a model."""

    def __init__(self):
        try:
            import tkinter as tk
        except ImportError as error:
            raise RuntimeError(
                "Tk is unavailable; install Python with Tcl/Tk or use --no-display"
            ) from error
        self.tk = tk
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError(
                "Cannot open a Tk window; use --no-display in a headless session"
            ) from error
        self.root.title("Thermal detector | Space: pause | Q / Esc: quit")
        self.root.configure(background="black")
        self.root.minsize(160, 120)
        self.label = tk.Label(self.root, background="black")
        self.label.pack(fill=tk.BOTH, expand=True)
        self.sized = False
        self.paused = False
        self.closed = False
        self.photo = None
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        for key in ("<Escape>", "q", "Q"):
            self.root.bind(key, lambda event: self.close())
        self.root.bind("<space>", self.toggle_pause)

    def toggle_pause(self, event=None):
        self.paused = not self.paused

    def pump(self):
        if not self.closed:
            self.root.update_idletasks()
            self.root.update()

    def show(self, image):
        """Show a BGR frame scaled to the current window, keeping aspect ratio.

        The first frame sets the window size; afterwards the geometry is fixed
        so resizing or maximizing the window rescales the video instead of the
        image size driving the window.
        """
        if self.closed:
            return
        if self.sized:
            image = fit_to_window(
                image, self.label.winfo_width(), self.label.winfo_height()
            )
        self.photo = self.tk.PhotoImage(data=ppm_bytes(image), format="PPM")
        self.label.configure(image=self.photo)
        self.pump()
        if not self.sized and not self.closed:
            self.root.geometry(f"{self.root.winfo_width()}x{self.root.winfo_height()}")
            self.sized = True

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
