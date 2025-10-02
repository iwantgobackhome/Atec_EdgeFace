#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Face Recognition System with GUI
실시간 얼굴 인식 시스템 (GUI 버전)

Features:
- Tkinter-based GUI interface
- Real-time camera feed display
- Reference image management (Add/Remove)
- Detector selection dropdown
- Display: FPS, Person ID, Similarity score
"""

import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import threading
import time
from typing import Optional

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

# Import core system
from face_recognition_system import FaceRecognitionSystem


class FaceRecognitionGUI:
    """GUI for Face Recognition System"""

    def __init__(self, root: tk.Tk):
        """
        Initialize GUI

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Real-time Face Recognition System")
        self.root.geometry("1400x800")

        # System variables
        self.system: Optional[FaceRecognitionSystem] = None
        self.camera_running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame = None

        # Configuration
        self.detector_var = tk.StringVar(value='mtcnn')
        self.device_var = tk.StringVar(value='cuda')
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.camera_id_var = tk.IntVar(value=0)

        # Model paths
        self.model_path = 'checkpoints/edgeface_xs_gamma_06.pt'
        self.model_name = 'edgeface_xs_gamma_06'

        # Build UI
        self.build_ui()

        # Initialize system
        self.initialize_system()

    def build_ui(self):
        """Build user interface"""

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # ===== Left Panel: Controls =====
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        row = 0

        # Detector Selection
        ttk.Label(control_frame, text="Face Detector:").grid(row=row, column=0, sticky=tk.W, pady=5)
        detector_combo = ttk.Combobox(control_frame, textvariable=self.detector_var, state='readonly', width=20)
        detector_combo['values'] = ('mtcnn', 'yunet', 'yolov5_face', 'yolov8')
        detector_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        detector_combo.bind('<<ComboboxSelected>>', self.on_detector_changed)
        row += 1

        # Device Selection
        ttk.Label(control_frame, text="Device:").grid(row=row, column=0, sticky=tk.W, pady=5)
        device_combo = ttk.Combobox(control_frame, textvariable=self.device_var, state='readonly', width=20)
        device_combo['values'] = ('cuda', 'cpu')
        device_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        device_combo.bind('<<ComboboxSelected>>', self.on_device_changed)
        row += 1

        # Similarity Threshold
        ttk.Label(control_frame, text="Similarity Threshold:").grid(row=row, column=0, sticky=tk.W, pady=5)
        threshold_spinbox = ttk.Spinbox(control_frame, from_=0.0, to=1.0, increment=0.05,
                                       textvariable=self.threshold_var, width=20)
        threshold_spinbox.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Camera ID
        ttk.Label(control_frame, text="Camera ID:").grid(row=row, column=0, sticky=tk.W, pady=5)
        camera_spinbox = ttk.Spinbox(control_frame, from_=0, to=5, increment=1,
                                     textvariable=self.camera_id_var, width=20)
        camera_spinbox.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                               sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Camera Control Buttons
        ttk.Label(control_frame, text="Camera Control:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.start_btn = ttk.Button(control_frame, text="▶ Start Camera", command=self.start_camera)
        self.start_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.stop_btn = ttk.Button(control_frame, text="⬛ Stop Camera", command=self.stop_camera, state='disabled')
        self.stop_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=row, column=0, columnspan=2,
                                                               sticky=(tk.W, tk.E), pady=10)
        row += 1

        # Reference Management
        ttk.Label(control_frame, text="Reference Management:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        self.capture_btn = ttk.Button(control_frame, text="📸 Capture from Camera",
                                       command=self.capture_reference, state='disabled')
        self.capture_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="➕ Add from File", command=self.add_reference).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="➖ Remove Reference", command=self.remove_reference).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Reference List
        ttk.Label(control_frame, text="Registered Persons:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1

        self.ref_listbox = tk.Listbox(control_frame, height=10, width=30)
        self.ref_listbox.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=self.ref_listbox.yview)
        scrollbar.grid(row=row, column=2, sticky=(tk.N, tk.S))
        self.ref_listbox.config(yscrollcommand=scrollbar.set)
        row += 1

        # Configure control frame grid weights
        control_frame.columnconfigure(1, weight=1)
        control_frame.rowconfigure(row-1, weight=1)

        # ===== Top Right: Video Display =====
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # ===== Bottom Right: Status =====
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))

        self.status_text = tk.Text(status_frame, height=8, wrap=tk.WORD, state='disabled')
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def log_status(self, message: str):
        """Log message to status panel"""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')

    def initialize_system(self):
        """Initialize face recognition system"""
        try:
            self.log_status("🚀 Initializing Face Recognition System...")

            self.system = FaceRecognitionSystem(
                detector_method=self.detector_var.get(),
                edgeface_model_path=self.model_path,
                edgeface_model_name=self.model_name,
                device=self.device_var.get(),
                similarity_threshold=self.threshold_var.get()
            )

            self.log_status("✅ System initialized successfully")
            self.update_reference_list()

        except Exception as e:
            self.log_status(f"❌ Initialization failed: {e}")
            messagebox.showerror("Error", f"Failed to initialize system: {e}")

    def on_detector_changed(self, event=None):
        """Handle detector change"""
        if self.camera_running:
            messagebox.showwarning("Warning", "Stop camera before changing detector")
            return
        self.log_status(f"🔄 Changing detector to: {self.detector_var.get()}")
        self.initialize_system()

    def on_device_changed(self, event=None):
        """Handle device change"""
        if self.camera_running:
            messagebox.showwarning("Warning", "Stop camera before changing device")
            return
        self.log_status(f"🔄 Changing device to: {self.device_var.get()}")
        self.initialize_system()

    def start_camera(self):
        """Start camera feed"""
        if self.camera_running:
            return

        camera_id = self.camera_id_var.get()
        self.log_status(f"📹 Opening camera {camera_id}...")

        try:
            # Use V4L2 backend with proper settings for Linux
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

            if self.cap.isOpened():
                # Set MJPEG codec for better performance and compatibility
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

                # Test if we can actually read a frame (with timeout)
                self.log_status("📸 Testing frame capture...")
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    # Get actual codec being used
                    fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                    self.log_status(f"✅ Camera opened successfully (V4L2, Codec: {codec})")
                    self.log_status(f"📐 Resolution: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    self.cap.release()
                    self.cap = None
                    self.log_status("❌ Cannot read frames from camera")
            else:
                self.cap.release()
                self.cap = None

        except Exception as e:
            self.log_status(f"❌ Error opening camera: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None

        if self.cap is None or not self.cap.isOpened():
            self.log_status(f"❌ Cannot open camera {camera_id}")
            self.log_status(f"💡 Try: 1) Check camera permissions, 2) Try different Camera ID, 3) Check if camera is used by another app")
            messagebox.showerror("Error", f"Cannot open camera {camera_id}\n\nTroubleshooting:\n- Check camera permissions\n- Try different Camera ID (0, 1, 2...)\n- Close other apps using camera\n- Run: v4l2-ctl --list-devices")
            return

        self.camera_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.capture_btn.config(state='normal')

        # Update threshold
        self.system.similarity_threshold = self.threshold_var.get()

        self.log_status("✅ Camera started")

        # Start video thread
        self.video_thread = threading.Thread(target=self.update_video, daemon=True)
        self.video_thread.start()

    def stop_camera(self):
        """Stop camera feed"""
        if not self.camera_running:
            return

        self.camera_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.capture_btn.config(state='disabled')

        if self.cap:
            self.cap.release()

        self.log_status("⬛ Camera stopped")

        # Clear video display
        self.video_label.config(image='')

    def update_video(self):
        """Update video feed (runs in separate thread)"""
        consecutive_failures = 0
        max_failures = 30  # Stop after 30 consecutive failures (~1 second)

        while self.camera_running:
            if self.cap and self.cap.isOpened():
                # Use grab/retrieve for better performance and lower latency
                # grab() captures frame into buffer without decoding
                if not self.cap.grab():
                    consecutive_failures += 1
                    self.log_status(f"⚠️ Failed to grab frame ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        self.log_status("❌ Camera disconnected or too many failures")
                        self.camera_running = False
                        break
                    time.sleep(0.01)
                    continue

                # retrieve() decodes the grabbed frame
                ret, frame = self.cap.retrieve()

                if ret and frame is not None:
                    consecutive_failures = 0  # Reset failure counter

                    # Store current frame for capture
                    self.current_frame = frame.copy()

                    # Process frame
                    annotated_frame, _ = self.system.process_frame(frame)

                    # Convert to RGB for display
                    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Resize to fit display (max 800x600)
                    h, w = display_frame.shape[:2]
                    max_w, max_h = 800, 600
                    if w > max_w or h > max_h:
                        scale = min(max_w/w, max_h/h)
                        new_w, new_h = int(w*scale), int(h*scale)
                        display_frame = cv2.resize(display_frame, (new_w, new_h))

                    # Convert to PhotoImage
                    img = Image.fromarray(display_frame)
                    imgtk = ImageTk.PhotoImage(image=img)

                    # Update label
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                else:
                    consecutive_failures += 1
                    self.log_status(f"⚠️ Failed to retrieve frame ({consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        self.log_status("❌ Camera disconnected or too many failures")
                        self.camera_running = False
                        break
            else:
                self.log_status("❌ Camera is not opened")
                break

            time.sleep(0.01)  # Small delay

    def capture_reference(self):
        """Capture current frame and add as reference"""
        if not self.camera_running or self.current_frame is None:
            messagebox.showwarning("Warning", "Camera must be running to capture")
            return

        # Ask for person ID first
        person_id = simpledialog.askstring("Person ID", "Enter person name/ID:")

        if not person_id:
            return

        try:
            # Get current frame
            frame = self.current_frame.copy()

            # Convert to PIL for detector
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Detect and align face
            self.log_status(f"📸 Capturing face for {person_id}...")
            aligned_face = self.system.detector.align(pil_img)

            if aligned_face is None:
                self.log_status(f"❌ No face detected in current frame")
                messagebox.showerror("Error", "No face detected! Please position your face in the camera.")
                return

            # Convert PIL to numpy BGR
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            # Extract embedding
            embedding = self.system.recognizer.extract_embedding(face_np)

            # Add to database
            self.system.ref_db.add_person(person_id, embedding)

            self.log_status(f"✅ Successfully captured and added {person_id}")
            self.update_reference_list()
            messagebox.showinfo("Success", f"Captured and added {person_id} to reference database")

            # Save captured image for reference (optional)
            os.makedirs("captured_references", exist_ok=True)
            save_path = f"captured_references/{person_id}.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR))
            self.log_status(f"💾 Saved reference image to {save_path}")

        except Exception as e:
            self.log_status(f"❌ Error capturing reference: {e}")
            messagebox.showerror("Error", f"Failed to capture reference: {e}")

    def add_reference(self):
        """Add reference person from image file"""
        # Select image file
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # Ask for person ID
        person_id = simpledialog.askstring("Person ID", "Enter person name/ID:")

        if not person_id:
            return

        # Add to system
        self.log_status(f"➕ Adding reference: {person_id} from {os.path.basename(file_path)}")

        success = self.system.add_reference_from_image(file_path, person_id)

        if success:
            self.log_status(f"✅ Successfully added {person_id}")
            self.update_reference_list()
            messagebox.showinfo("Success", f"Added {person_id} to reference database")
        else:
            self.log_status(f"❌ Failed to add {person_id}")
            messagebox.showerror("Error", f"Failed to add {person_id}")

    def remove_reference(self):
        """Remove reference person"""
        selection = self.ref_listbox.curselection()

        if not selection:
            messagebox.showwarning("Warning", "Please select a person to remove")
            return

        person_id = self.ref_listbox.get(selection[0])

        # Confirm removal
        if messagebox.askyesno("Confirm", f"Remove {person_id} from database?"):
            self.system.ref_db.remove_person(person_id)
            self.log_status(f"➖ Removed {person_id}")
            self.update_reference_list()

    def update_reference_list(self):
        """Update reference list display"""
        self.ref_listbox.delete(0, tk.END)
        persons = self.system.ref_db.get_all_persons()
        for person in sorted(persons):
            self.ref_listbox.insert(tk.END, person)

    def on_closing(self):
        """Handle window close"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use('clam')

    app = FaceRecognitionGUI(root)

    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()


if __name__ == '__main__':
    main()
