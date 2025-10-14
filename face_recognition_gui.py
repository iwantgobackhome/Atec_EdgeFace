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
        self.current_landmarks = None

        # Multi-angle capture state
        self.capture_mode = False
        self.capture_person_id = None
        self.required_angles = ['front', 'left', 'right', 'up', 'down']
        self.captured_angles = set()
        self.angle_slots = {}

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

        self.capture_btn = ttk.Button(control_frame, text="📸 Capture Multi-Angle",
                                       command=self.start_multi_angle_capture, state='disabled')
        self.capture_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        self.cancel_capture_btn = ttk.Button(control_frame, text="❌ Cancel Capture",
                                              command=self.cancel_capture, state='disabled')
        self.cancel_capture_btn.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="➕ Add from File", command=self.add_reference).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="➖ Remove Person", command=self.remove_reference).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        ttk.Button(control_frame, text="🗑️ Manage Angles", command=self.manage_angles).grid(
            row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # Reference List
        ttk.Label(control_frame, text="Registered Persons:", font=('', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        row += 1

        self.ref_listbox = tk.Listbox(control_frame, height=10, width=30)
        self.ref_listbox.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.ref_listbox.bind('<<ListboxSelect>>', self.on_person_selected)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=self.ref_listbox.yview)
        scrollbar.grid(row=row, column=2, sticky=(tk.N, tk.S))
        self.ref_listbox.config(yscrollcommand=scrollbar.set)
        row += 1

        # Show angles for selected person
        self.angles_label = ttk.Label(control_frame, text="Captured angles: -", font=('', 9))
        self.angles_label.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
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
                    annotated_frame, detections = self.system.process_frame(frame)

                    # Handle multi-angle capture mode
                    if self.capture_mode and len(detections) > 0:
                        det = detections[0]  # Use first detected face
                        if det['landmarks'] is not None:
                            self.current_landmarks = det['landmarks']
                            self.auto_capture_angle(det['landmarks'])

                    # Draw angle capture overlay if in capture mode
                    if self.capture_mode:
                        annotated_frame = self.draw_angle_overlay(annotated_frame)

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

    def draw_angle_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw angle capture progress overlay on frame"""
        h, w = frame.shape[:2]

        # Create semi-transparent overlay panel on the left side
        overlay = frame.copy()
        panel_width = 200
        panel_height = 250
        panel_x = 10
        panel_y = 60

        # Draw semi-transparent background
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "Capture Progress", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw angle indicators with visual icons
        y_offset = panel_y + 50
        angle_positions = {
            'front': ('Front', (panel_width // 2, 0)),
            'left': ('Left', (20, panel_height // 2 - 40)),
            'right': ('Right', (panel_width - 60, panel_height // 2 - 40)),
            'up': ('Up', (panel_width // 2, -20)),
            'down': ('Down', (panel_width // 2, panel_height - 60))
        }

        # Draw center face icon
        center_x = panel_x + panel_width // 2
        center_y = panel_y + panel_height // 2
        cv2.circle(frame, (center_x, center_y), 25, (100, 100, 100), 2)

        # Draw angle indicators
        for i, angle in enumerate(self.required_angles):
            y = y_offset + i * 35

            # Status symbol
            if angle in self.captured_angles:
                symbol = "✓"
                color = (0, 255, 0)
            else:
                symbol = "○"
                color = (150, 150, 150)

            # Draw angle name and status
            text = f"{angle.capitalize()}: {symbol}"
            cv2.putText(frame, text, (panel_x + 15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Progress bar at bottom
        progress = len(self.captured_angles) / len(self.required_angles)
        bar_width = panel_width - 20
        bar_height = 15
        bar_x = panel_x + 10
        bar_y = panel_y + panel_height - 30

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)

        # Progress bar
        progress_width = int(bar_width * progress)
        if progress_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + progress_width, bar_y + bar_height),
                         (0, 255, 0), -1)

        # Progress text
        progress_text = f"{len(self.captured_angles)}/{len(self.required_angles)}"
        cv2.putText(frame, progress_text, (bar_x + bar_width // 2 - 20, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def start_multi_angle_capture(self):
        """Start multi-angle capture mode"""
        if not self.camera_running or self.current_frame is None:
            messagebox.showwarning("Warning", "Camera must be running to capture")
            return

        # Ask for person ID first
        person_id = simpledialog.askstring("Person ID", "Enter person name/ID:")

        if not person_id:
            return

        # Initialize capture mode
        self.capture_mode = True
        self.capture_person_id = person_id
        self.captured_angles = set()
        self.angle_slots = {}

        # Update UI
        self.capture_btn.config(state='disabled')
        self.cancel_capture_btn.config(state='normal')

        self.log_status(f"🎯 Multi-angle capture started for {person_id}")
        self.log_status(f"👉 Please rotate your face to capture: Front, Left, Right, Up, Down")

    def cancel_capture(self):
        """Cancel multi-angle capture mode"""
        self.capture_mode = False
        self.capture_person_id = None
        self.captured_angles = set()
        self.angle_slots = {}

        # Update UI
        self.capture_btn.config(state='normal')
        self.cancel_capture_btn.config(state='disabled')

        self.log_status("❌ Multi-angle capture cancelled")

    def auto_capture_angle(self, landmarks: np.ndarray):
        """Auto-capture when face is at specific angle"""
        from face_recognition_system import FaceAngleCalculator

        # Calculate face angles
        yaw, pitch, roll = FaceAngleCalculator.calculate_head_pose(landmarks)
        angle_category = FaceAngleCalculator.get_angle_category(yaw, pitch)

        # Check if this angle is needed and not yet captured
        if angle_category in self.required_angles and angle_category not in self.captured_angles:
            # Capture this angle
            try:
                frame = self.current_frame.copy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Detect and align face
                aligned_face = self.system.detector.align(pil_img)

                if aligned_face is not None:
                    # Convert PIL to numpy BGR
                    face_np = np.array(aligned_face)
                    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                    # Extract embedding
                    embedding = self.system.recognizer.extract_embedding(face_np)

                    # Add to database with angle
                    self.system.ref_db.add_person(self.capture_person_id, embedding, angle_category)

                    # Mark as captured
                    self.captured_angles.add(angle_category)
                    self.angle_slots[angle_category] = aligned_face

                    # Log capture
                    self.log_status(f"✅ Captured {angle_category} angle (Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°)")

                    # Save image
                    os.makedirs(f"captured_references/{self.capture_person_id}", exist_ok=True)
                    save_path = f"captured_references/{self.capture_person_id}/{angle_category}.jpg"
                    cv2.imwrite(save_path, cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR))

                    # Check if all angles captured
                    if len(self.captured_angles) == len(self.required_angles):
                        self.log_status(f"🎉 All angles captured for {self.capture_person_id}!")
                        self.update_reference_list()
                        messagebox.showinfo("Success", f"All angles captured for {self.capture_person_id}!")
                        self.cancel_capture()

            except Exception as e:
                self.log_status(f"❌ Error capturing {angle_category} angle: {e}")

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

    def on_person_selected(self, event=None):
        """Handle person selection in listbox"""
        selection = self.ref_listbox.curselection()
        if not selection:
            self.angles_label.config(text="Captured angles: -")
            return

        person_id = self.ref_listbox.get(selection[0])
        angles = self.system.ref_db.get_person_angles(person_id)

        if angles:
            angle_text = ", ".join(sorted(angles))
            self.angles_label.config(text=f"Captured angles: {angle_text}")
        else:
            self.angles_label.config(text="Captured angles: None")

    def manage_angles(self):
        """Open angle management dialog"""
        selection = self.ref_listbox.curselection()

        if not selection:
            messagebox.showwarning("Warning", "Please select a person first")
            return

        person_id = self.ref_listbox.get(selection[0])
        angles = self.system.ref_db.get_person_angles(person_id)

        if not angles:
            messagebox.showinfo("Info", f"{person_id} has no captured angles")
            return

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Manage Angles - {person_id}")
        dialog.geometry("400x300")

        ttk.Label(dialog, text=f"Captured angles for {person_id}:",
                 font=('', 10, 'bold')).pack(pady=10)

        # Listbox with angles
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        angle_listbox = tk.Listbox(frame, height=8)
        angle_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=angle_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        angle_listbox.config(yscrollcommand=scrollbar.set)

        for angle in sorted(angles):
            angle_listbox.insert(tk.END, angle)

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        def delete_selected_angle():
            sel = angle_listbox.curselection()
            if not sel:
                messagebox.showwarning("Warning", "Please select an angle to delete")
                return

            angle = angle_listbox.get(sel[0])

            if messagebox.askyesno("Confirm", f"Delete {angle} angle for {person_id}?"):
                self.system.ref_db.remove_person_angle(person_id, angle)
                self.log_status(f"🗑️ Deleted {angle} angle for {person_id}")
                angle_listbox.delete(sel[0])

                # Update main UI
                self.on_person_selected()

                # If no angles left, close dialog
                if angle_listbox.size() == 0:
                    messagebox.showinfo("Info", f"All angles deleted for {person_id}")
                    dialog.destroy()
                    self.update_reference_list()

        ttk.Button(btn_frame, text="Delete Selected Angle", command=delete_selected_angle).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Close", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def update_reference_list(self):
        """Update reference list display"""
        self.ref_listbox.delete(0, tk.END)
        persons = self.system.ref_db.get_all_persons()
        for person in sorted(persons):
            self.ref_listbox.insert(tk.END, person)

        # Update angles display if a person is selected
        self.on_person_selected()

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
