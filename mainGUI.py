import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
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
        self.tracking = False
        self.current_frame = 0
        self.end_frame = None

        # Create a PanedWindow
        self.paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Create the left and right panes
        self.left_pane = tk.Frame(self.paned_window, width=300, relief=tk.SUNKEN)
        self.right_pane = tk.Frame(self.paned_window, relief=tk.SUNKEN)

        self.paned_window.add(self.left_pane)
        self.paned_window.add(self.right_pane)

        # Create GUI elements in the left pane
        self.load_button = tk.Button(self.left_pane, text="Load Video", command=self.load_video)
        self.load_button.pack(pady=10)

        self.track_button = tk.Button(self.left_pane, text="Track Blips", command=self.start_tracking)
        self.track_button.pack(pady=10)

        self.stop_button = tk.Button(self.left_pane, text="Stop", command=self.stop_tracking)
        self.stop_button.pack(pady=10)

        self.refine_button = tk.Button(self.left_pane, text="Refine and Retrack", command=self.refine_and_retrack)
        self.refine_button.pack(pady=10)

        self.export_button = tk.Button(self.left_pane, text="Export 3D Data", command=self.export_data)
        self.export_button.pack(pady=10)

        # Create sliders in the left pane
        self.size_slider = tk.Scale(self.left_pane, from_=50, to_=100, orient=tk.HORIZONTAL, label="Window Size (%)", command=self.update_window_size)
        self.size_slider.set(50)  # Default to 50%
        self.size_slider.pack(pady=10)

        # Parameter sliders in the left pane
        self.max_blips_slider = tk.Scale(self.left_pane, from_=10, to_=500, orient=tk.HORIZONTAL, label="Max Blips")
        self.max_blips_slider.set(self.max_blips)
        self.max_blips_slider.pack(pady=5)

        self.quality_level_slider = tk.Scale(self.left_pane, from_=1, to_=100, orient=tk.HORIZONTAL, label="Quality Level (%)")
        self.quality_level_slider.set(1)  # Default to 1% (0.01)
        self.quality_level_slider.pack(pady=5)

        self.min_distance_slider = tk.Scale(self.left_pane, from_=1, to_=50, orient=tk.HORIZONTAL, label="Min Distance")
        self.min_distance_slider.set(10)  # Default to 10
        self.min_distance_slider.pack(pady=5)

        self.block_size_slider = tk.Scale(self.left_pane, from_=3, to_=30, orient=tk.HORIZONTAL, label="Block Size")
        self.block_size_slider.set(7)  # Default to 7
        self.block_size_slider.pack(pady=5)

        # Video display area in the right pane
        self.video_label = tk.Label(self.right_pane)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 3D Visualization window in the right pane
        self.fig, self.ax = self.create_3d_figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_pane)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Timeline bar at the bottom
        self.timeline_frame = tk.Frame(root)
        self.timeline_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.timeline = tk.Scale(self.timeline_frame, from_=0, to_=100, orient=tk.HORIZONTAL, label="Timeline", command=self.update_frame_from_timeline)
        self.timeline.pack(fill=tk.X, padx=10, pady=5)

        self.update_window_size()  # Initial adjustment of window size

    def load_video(self):
        filetypes = [("Video Files", "*.mp4;*.avi")]
        self.video_path = filedialog.askopenfilename(filetypes=filetypes)
        if self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
            self.end_frame = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            self.timeline.config(to=self.end_frame)  # Update timeline range
            self.timeline.set(0)  # Set timeline to the first frame
            messagebox.showinfo("Info", "Video loaded successfully.")
        else:
            messagebox.showwarning("Warning", "No video selected.")

    def start_tracking(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        self.tracking = True
        self.current_frame = 0
        self.show_frame()

    def stop_tracking(self):
        self.tracking = False

    def show_frame(self):
        if self.tracking and self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video_cap.read()
            if ret:
                # Convert the frame to RGB (Tkinter uses RGB, while OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect blips and draw them on the frame
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blips = self.detect_blips(gray_frame)
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
                self.current_frame += 1
                if self.current_frame <= self.end_frame:
                    self.timeline.set(self.current_frame)
                    self.video_label.after(30, self.show_frame)
                else:
                    self.stop_tracking()
            else:
                # Stop tracking if video ends
                self.stop_tracking()

    def update_frame_from_timeline(self, value):
        self.current_frame = int(value)
        if not self.tracking:
            self.show_frame()  # Show the selected frame if not currently tracking

    def detect_blips(self, image):
        max_blips = self.max_blips_slider.get()
        quality_level = self.quality_level_slider.get() / 100.0  # Convert from slider percentage to decimal
        min_distance = self.min_distance_slider.get()
        block_size = self.block_size_slider.get()

        feature_params = dict(maxCorners=max_blips,
                              qualityLevel=quality_level,
                              minDistance=min_distance,
                              blockSize=block_size)
        blips = cv2.goodFeaturesToTrack(image, mask=None, **feature_params)
        return np.int32(blips) if blips is not None else None

    def refine_and_retrack(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first.")
            return

        # Re-read parameters from sliders
        self.max_blips = self.max_blips_slider.get()
        self.error_threshold = self.error_threshold_slider.get() / 100.0  # Convert from slider percentage to decimal

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

    def update_window_size(self, event=None):
        size_percentage = self.size_slider.get() / 100.0
        window_width = int(self.root.winfo_screenwidth() * size_percentage)
        window_height = int(self.root.winfo_screenheight() * size_percentage)

        # Update the size of the left and right panes
        self.left_pane.config(width=window_width // 2, height=window_height)
        self.right_pane.config(width=window_width // 2, height=window_height)

        # Update the size of video label and canvas
        self.video_label.config(width=window_width // 2, height=window_height)
        self.canvas.get_tk_widget().config(width=window_width // 2, height=window_height)

        # Update timeline bar size
        self.timeline_frame.config(width=window_width)

# Create the main window and application
root = tk.Tk()
app = App(root)
root.mainloop()
