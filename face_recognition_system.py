#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Face Recognition System using EdgeFace
실시간 얼굴 인식 시스템

Features:
- Real-time face detection and recognition
- Multiple detector support (MTCNN, YuNet, YOLO, etc.)
- Reference image management
- Display: FPS, Person ID, Similarity score
"""

import os
import sys
import cv2
import numpy as np
import torch
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

from backbones import get_model
from face_alignment.unified_detector import UnifiedFaceDetector


class EdgeFaceRecognizer:
    """EdgeFace based face recognition module"""

    def __init__(self, model_path: str, model_name: str = 'edgeface_xs_gamma_06', device: str = 'cuda'):
        """
        Initialize EdgeFace recognizer

        Args:
            model_path: Path to EdgeFace model checkpoint
            model_name: Model architecture name
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load EdgeFace model
        self.model = get_model(model_name, fp16=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ EdgeFace model loaded: {model_name} on {self.device}")

    @torch.no_grad()
    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from aligned face image

        Args:
            face_img: Aligned face image (112x112x3)

        Returns:
            Face embedding vector (512-d)
        """
        # Preprocess
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112))

        # Convert BGR to RGB
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Transpose to CHW
        img = np.transpose(img, (2, 0, 1))

        # Convert to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        # Normalize
        img_tensor.div_(255).sub_(0.5).div_(0.5)

        # Extract embedding
        embedding = self.model(img_tensor).cpu().numpy().flatten()

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(emb1, emb2))


class ReferenceDatabase:
    """Reference face database management"""

    def __init__(self, db_path: str = 'reference_db.pkl'):
        """
        Initialize reference database

        Args:
            db_path: Path to save/load database
        """
        self.db_path = db_path
        self.db: Dict[str, np.ndarray] = {}
        self.load()

    def add_person(self, person_id: str, embedding: np.ndarray):
        """Add or update a person's embedding"""
        self.db[person_id] = embedding
        self.save()

    def remove_person(self, person_id: str):
        """Remove a person from database"""
        if person_id in self.db:
            del self.db[person_id]
            self.save()

    def get_all_persons(self) -> List[str]:
        """Get list of all person IDs"""
        return list(self.db.keys())

    def find_match(self, query_embedding: np.ndarray, threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Find best matching person

        Args:
            query_embedding: Query face embedding
            threshold: Minimum similarity threshold

        Returns:
            (person_id, similarity) or (None, 0.0) if no match
        """
        if not self.db:
            return None, 0.0

        best_match = None
        best_similarity = threshold

        for person_id, ref_embedding in self.db.items():
            similarity = EdgeFaceRecognizer.cosine_similarity(query_embedding, ref_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_id

        return best_match, best_similarity

    def save(self):
        """Save database to disk"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.db, f)

    def load(self):
        """Load database from disk"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.db = pickle.load(f)
            print(f"✅ Loaded {len(self.db)} persons from database")
        else:
            print("📂 New database created")


class FaceRecognitionSystem:
    """Main real-time face recognition system"""

    def __init__(
        self,
        detector_method: str = 'mtcnn',
        edgeface_model_path: str = 'checkpoints/edgeface_xs_gamma_06.pt',
        edgeface_model_name: str = 'edgeface_xs_gamma_06',
        device: str = 'cuda',
        similarity_threshold: float = 0.5
    ):
        """
        Initialize face recognition system

        Args:
            detector_method: Face detection method ('mtcnn', 'yunet', 'yolov5_face', 'yolov8')
            edgeface_model_path: Path to EdgeFace model
            edgeface_model_name: EdgeFace model architecture name
            device: 'cuda' or 'cpu'
            similarity_threshold: Minimum similarity for recognition
        """
        print("🚀 Initializing Face Recognition System...")

        # Device setup
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Initialize face detector
        print(f"📷 Initializing face detector: {detector_method}")
        self.detector = UnifiedFaceDetector(detector_method, device=self.device)

        # Initialize EdgeFace recognizer
        print(f"🧠 Initializing EdgeFace recognizer...")
        self.recognizer = EdgeFaceRecognizer(edgeface_model_path, edgeface_model_name, self.device)

        # Initialize reference database
        print(f"💾 Initializing reference database...")
        self.ref_db = ReferenceDatabase()

        # Settings
        self.similarity_threshold = similarity_threshold

        # FPS calculation
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()

        print("✅ System initialized successfully!")

    def add_reference_from_image(self, image_path: str, person_id: str) -> bool:
        """
        Add reference person from image file

        Args:
            image_path: Path to reference image
            person_id: Person identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image
            img = Image.open(image_path)

            # Detect and align face
            aligned_face = self.detector.align(img)

            if aligned_face is None:
                print(f"❌ No face detected in {image_path}")
                return False

            # Convert PIL to numpy
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_np)

            # Add to database
            self.ref_db.add_person(person_id, embedding)

            print(f"✅ Added {person_id} to reference database")
            return True

        except Exception as e:
            print(f"❌ Error adding reference: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame

        Args:
            frame: Input frame (BGR)

        Returns:
            (annotated_frame, detections)
            detections: List of {'person_id', 'similarity', 'bbox', 'landmarks'}
        """
        # Convert to PIL for detector
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        # Detect faces
        bboxes, landmarks = self.detector.detect_faces(pil_img)

        detections = []

        if bboxes is not None and len(bboxes) > 0:
            # Extract embeddings for all detected faces
            face_embeddings = []
            for bbox, lm in zip(bboxes, landmarks):
                # Align face
                aligned_face = self.detector.align_face(pil_img, lm)

                if aligned_face is not None:
                    # Convert to numpy BGR
                    face_np = np.array(aligned_face)
                    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                    # Extract embedding
                    embedding = self.recognizer.extract_embedding(face_np)
                    face_embeddings.append({
                        'embedding': embedding,
                        'bbox': bbox,
                        'landmarks': lm
                    })
                else:
                    face_embeddings.append(None)

            # Perform matching with assignment tracking
            detections = self.assign_identities(face_embeddings)

        # Update FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        # Annotate frame
        annotated_frame = self.draw_results(frame.copy(), detections)

        return annotated_frame, detections

    def assign_identities(self, face_embeddings: List[Dict]) -> List[Dict]:
        """
        Assign identities to detected faces using greedy assignment

        Args:
            face_embeddings: List of face embedding data

        Returns:
            List of detections with assigned identities
        """
        if not self.ref_db.db:
            # No references, mark all as unknown
            return [{
                'person_id': 'Unknown',
                'similarity': 0.0,
                'bbox': fe['bbox'],
                'landmarks': fe['landmarks']
            } for fe in face_embeddings if fe is not None]

        # Calculate similarity matrix: [num_faces x num_references]
        candidates = []
        for i, fe in enumerate(face_embeddings):
            if fe is None:
                continue

            for person_id, ref_embedding in self.ref_db.db.items():
                similarity = EdgeFaceRecognizer.cosine_similarity(fe['embedding'], ref_embedding)
                if similarity >= self.similarity_threshold:
                    candidates.append({
                        'face_idx': i,
                        'person_id': person_id,
                        'similarity': similarity,
                        'bbox': fe['bbox'],
                        'landmarks': fe['landmarks']
                    })

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x['similarity'], reverse=True)

        # Greedy assignment: assign each face to best match, avoiding duplicates
        assigned_faces = set()
        assigned_persons = set()
        detections = []

        for candidate in candidates:
            face_idx = candidate['face_idx']
            person_id = candidate['person_id']

            # Skip if face or person already assigned
            if face_idx in assigned_faces or person_id in assigned_persons:
                continue

            # Assign this match
            detections.append({
                'person_id': person_id,
                'similarity': candidate['similarity'],
                'bbox': candidate['bbox'],
                'landmarks': candidate['landmarks']
            })

            assigned_faces.add(face_idx)
            assigned_persons.add(person_id)

        # Add unmatched faces as Unknown
        for i, fe in enumerate(face_embeddings):
            if fe is not None and i not in assigned_faces:
                detections.append({
                    'person_id': 'Unknown',
                    'similarity': 0.0,
                    'bbox': fe['bbox'],
                    'landmarks': fe['landmarks']
                })

        return detections

    def draw_results(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on frame"""

        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw detections
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox[:4])

            # Draw bounding box
            color = (0, 255, 0) if det['person_id'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw person ID and similarity
            label = f"{det['person_id']}: {det['similarity']:.2f}"

            # Background for text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)

            # Text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw landmarks
            landmarks = det['landmarks']
            if landmarks is not None:
                for i in range(0, len(landmarks), 2):
                    x, y = int(landmarks[i]), int(landmarks[i+1])
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

        return frame

    def add_reference_from_frame(self, frame: np.ndarray, person_id: str) -> bool:
        """
        Add reference person from camera frame

        Args:
            frame: Camera frame (BGR)
            person_id: Person identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to PIL for detector
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Detect and align face
            aligned_face = self.detector.align(pil_img)

            if aligned_face is None:
                print(f"❌ No face detected in frame")
                return False

            # Convert PIL to numpy
            face_np = np.array(aligned_face)
            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_np)

            # Add to database
            self.ref_db.add_person(person_id, embedding)

            # Save captured image
            os.makedirs("captured_references", exist_ok=True)
            save_path = f"captured_references/{person_id}.jpg"
            cv2.imwrite(save_path, cv2.cvtColor(np.array(aligned_face), cv2.COLOR_RGB2BGR))

            print(f"✅ Added {person_id} to reference database")
            print(f"💾 Saved reference image to {save_path}")
            return True

        except Exception as e:
            print(f"❌ Error adding reference: {e}")
            return False

    def run_camera(self, camera_id: int = 0, enable_capture: bool = True):
        """
        Run face recognition on camera feed

        Args:
            camera_id: Camera device ID
            enable_capture: Enable 'c' key to capture reference
        """
        print(f"📹 Opening camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return

        print("✅ Camera opened successfully")
        print("Controls:")
        print("  'q' - Quit")
        if enable_capture:
            print("  'c' - Capture current frame as reference")
        print(f"📊 Registered persons: {self.ref_db.get_all_persons()}")

        current_frame = None

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("❌ Failed to grab frame")
                    break

                current_frame = frame.copy()

                # Process frame
                annotated_frame, detections = self.process_frame(frame)

                # Display
                cv2.imshow('Face Recognition System', annotated_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF

                # Quit on 'q'
                if key == ord('q'):
                    break

                # Capture on 'c'
                elif key == ord('c') and enable_capture:
                    print("\n📸 Capture mode activated")
                    person_id = input("Enter person name/ID (or press Enter to cancel): ").strip()

                    if person_id:
                        success = self.add_reference_from_frame(current_frame, person_id)
                        if success:
                            print(f"✅ {person_id} added successfully!")
                            print(f"📊 Registered persons: {self.ref_db.get_all_persons()}")
                        else:
                            print(f"❌ Failed to add {person_id}")
                    else:
                        print("⏭️  Capture cancelled")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Camera closed")


def main():
    """Main function for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Face Recognition System')
    parser.add_argument('--detector', type=str, default='mtcnn',
                       choices=['mtcnn', 'yunet', 'yolov5_face', 'yolov8'],
                       help='Face detection method')
    parser.add_argument('--model', type=str, default='checkpoints/edgeface_xs_gamma_06.pt',
                       help='EdgeFace model path')
    parser.add_argument('--model-name', type=str, default='edgeface_xs_gamma_06',
                       help='EdgeFace model architecture name')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for recognition')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--add-ref', type=str, nargs=2, metavar=('IMAGE', 'ID'),
                       help='Add reference image: --add-ref path/to/image.jpg person_name')

    args = parser.parse_args()

    # Initialize system
    system = FaceRecognitionSystem(
        detector_method=args.detector,
        edgeface_model_path=args.model,
        edgeface_model_name=args.model_name,
        device=args.device,
        similarity_threshold=args.threshold
    )

    # Add reference if specified
    if args.add_ref:
        image_path, person_id = args.add_ref
        system.add_reference_from_image(image_path, person_id)

    # Run camera
    system.run_camera(args.camera)


if __name__ == '__main__':
    main()
