import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Reconstruction GUI")

        # Initialize variables
        self.video_path = None
        self.camera_matrix = np.array([[1000, 0, 640],
                                      [0, 1000, 360],
                                      [0, 0, 1]])
        self.max_blips = 100
        self.error_threshold = 1.0
        self.video_cap = None

        # GUI Elements
        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack(pady=10)

        self.track_button = tk.Button(root, text="Track Blips", command=self.track_blips)
        self.track_button.pack(pady=10)

        self.refine_button = tk.Button(root, text="Refine and Retrack", command=self.refine_and_retrack)
        self.refine_button.pack(pady=10)

        self.export_button = tk.Button(root, text="Export 3D Data", command=self.export_data)
        self.export_button.pack(pady=10)

        # Video display area
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        # 3D Visualization window
        self.fig, self.ax = self.create_3d_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_video(self):
        filetypes = [("Video Files", "*.mp4;*.avi")]
        self.video_path = filedialog.askopenfilename(filetypes=filetypes)
        if self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
            messagebox.showinfo("Info", "Video loaded successfully.")
            self.show_frame()  # Start displaying video
        else:
            messagebox.showwarning("Warning", "No video selected.")

    def show_frame(self):
        if self.video_cap is not None:
            ret, frame = self.video_cap.read()
            if ret:
                # Convert the frame to RGB (Tkinter uses RGB, while OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect blips and draw them on the frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blips = self.detect_blips(gray_frame, max_blips=self.max_blips)
                if blips is not None:
                    for blip in blips:
                        x, y = blip.ravel()
                        cv2.circle(frame_rgb, (x, y), 5, (0, 255, 0), -1)

                # Convert OpenCV image to Tkinter-compatible format
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                # Display the image in the video_label
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

                # Update frame after 30ms
                self.video_label.after(30, self.show_frame)

    def detect_blips(self, image, max_blips=100):
        feature_params = dict(maxCorners=max_blips,
                              qualityLevel=0.01,
                              minDistance=10,
                              blockSize=7)
        blips = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
        return np.int32(blips) if blips is not None else None

    def track_blips(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first.")
            return
        # Start tracking the blips in video frames
        messagebox.showinfo("Info", "Blips tracked.")
        self.show_frame()

    def refine_and_retrack(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        self.max_blips = simpledialog.askinteger("Input", "Max Blips:", initialvalue=self.max_blips)
        if self.max_blips is None:
            return

        self.error_threshold = simpledialog.askfloat("Input", "Error Threshold:", initialvalue=self.error_threshold)
        if self.error_threshold is None:
            return

        # Call your refinement function here
        # Example: refine_and_retrack_blips(self.video_path, self.max_blips, self.error_threshold)

        messagebox.showinfo("Info", "Blips refined and retracked.")

    def export_data(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        filetypes = [("USD Files", "*.usd"), ("FBX Files", "*.fbx")]
        filepath = filedialog.asksaveasfilename(defaultextension=".usd", filetypes=filetypes)
        if not filepath:
            return

        # Call your export function here
        # Example: export_3d_data(filepath)

        messagebox.showinfo("Info", "3D data exported.")

    def create_3d_figure(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D Reconstruction View')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return fig, ax

    def update_3d_view(self, points_3d):
        self.ax.clear()
        self.ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
        self.ax.set_title('3D Reconstruction View')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.canvas.draw()

# Create the main window and application
root = tk.Tk()
app = App(root)
root.mainloop()
