import cv2 as cv
import numpy as np
from PIL import Image
import os
import sys
from typing import Tuple, List, Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Mediapipe not installed. Install with: pip install mediapipe")

sys.path.insert(0, os.path.dirname(__file__))
try:
    from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
except ImportError:
    try:
        from align_trans import get_reference_facial_points, warp_and_crop_face
    except ImportError:
        print("Warning: align_trans module not found. MediaPipe alignment may not work properly.")
        get_reference_facial_points = None
        warp_and_crop_face = None


class MediaPipeDetector:
    """MediaPipe detector for face detection and landmark extraction"""
    
    def __init__(self, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        """
        MediaPipe detector for face alignment
        Args:
            device: Device to use ('cpu' only for MediaPipe)
            crop_size: Output crop size for aligned faces
        """
        self.device = device
        self.crop_size = crop_size
        
        if not MEDIAPIPE_AVAILABLE:
            self.face_detection = None
            self.face_mesh = None
            print("MediaPipe not available - detector disabled")
            return
        
        # Initialize MediaPipe solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Create detection and mesh models
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            refine_landmarks=True,  # Get additional eye/lip landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Get reference points for alignment
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))
        
        # MediaPipe face mesh landmark indices for 5-point landmarks
        self.FACE_LANDMARK_INDICES = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose_tip': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151],
            'mouth_left': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308],
            'mouth_right': [291, 303, 267, 269, 270, 267, 271, 272, 271, 302]
        }

    def _mediapipe_to_5points(self, landmarks, image_width: int, image_height: int) -> np.ndarray:
        """Convert MediaPipe 468 landmarks to 5-point face landmarks"""
        if landmarks is None:
            return None
        
        try:
            # Convert normalized coordinates to pixel coordinates
            landmark_points = []
            for lm in landmarks.landmark:
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)
                landmark_points.append([x, y])
            
            landmark_points = np.array(landmark_points)
            
            # Extract 5-point landmarks
            # Left eye center (average of eye landmarks)
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_eye_center = landmark_points[left_eye_indices].mean(axis=0)
            
            # Right eye center
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_eye_center = landmark_points[right_eye_indices].mean(axis=0)
            
            # Nose tip
            nose_tip = landmark_points[1]  # Nose tip landmark
            
            # Mouth corners
            mouth_left = landmark_points[61]   # Left mouth corner
            mouth_right = landmark_points[291] # Right mouth corner
            
            # Create 5-point landmarks array
            face_5points = np.array([
                left_eye_center,
                right_eye_center,
                nose_tip,
                mouth_left,
                mouth_right
            ])
            
            return face_5points.flatten()  # [x1,y1,x2,y2,...,x5,y5]
            
        except Exception as e:
            print(f"Failed to convert MediaPipe landmarks: {e}")
            return None

    def detect_faces(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces and extract landmarks using MediaPipe
        Returns:
            bboxes: Array of shape [N, 5] with [x1, y1, x2, y2, conf]
            landmarks: Array of shape [N, 10] with [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
        """
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            return np.array([]), np.array([])
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_rgb = cv.cvtColor(img_array, cv.COLOR_RGB2BGR) if len(img_array.shape) == 3 else img_array
        img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        
        height, width = img_rgb.shape[:2]
        
        try:
            # Detect faces
            detection_results = self.face_detection.process(img_rgb)
            
            # Extract landmarks
            mesh_results = self.face_mesh.process(img_rgb)
            
            bboxes = []
            landmarks = []
            
            if detection_results.detections and mesh_results.multi_face_landmarks:
                # Process each detected face
                num_faces = min(len(detection_results.detections), len(mesh_results.multi_face_landmarks))
                
                for i in range(num_faces):
                    detection = detection_results.detections[i]
                    face_landmarks = mesh_results.multi_face_landmarks[i]
                    
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * width)
                    y1 = int(bbox.ymin * height)
                    x2 = int((bbox.xmin + bbox.width) * width)
                    y2 = int((bbox.ymin + bbox.height) * height)
                    conf = detection.score[0] if detection.score else 0.9
                    
                    bboxes.append([x1, y1, x2, y2, conf])
                    
                    # Extract 5-point landmarks
                    face_5points = self._mediapipe_to_5points(face_landmarks, width, height)
                    if face_5points is not None:
                        landmarks.append(face_5points)
                    else:
                        # Create default landmarks from bounding box
                        default_landmarks = self._create_default_landmarks_from_bbox(x1, y1, x2, y2)
                        landmarks.append(default_landmarks)
            
            return np.array(bboxes), np.array(landmarks)
            
        except Exception as e:
            print(f"MediaPipe detection failed: {e}")
            return np.array([]), np.array([])

    def _create_default_landmarks_from_bbox(self, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Create approximate 5-point landmarks from bounding box"""
        width = x2 - x1
        height = y2 - y1
        
        # Approximate landmark positions relative to bounding box
        landmarks = np.array([
            [x1 + 0.3 * width, y1 + 0.4 * height],  # Left eye
            [x1 + 0.7 * width, y1 + 0.4 * height],  # Right eye
            [x1 + 0.5 * width, y1 + 0.6 * height],  # Nose
            [x1 + 0.35 * width, y1 + 0.8 * height], # Left mouth
            [x1 + 0.65 * width, y1 + 0.8 * height]  # Right mouth
        ])
        
        return landmarks.flatten()

    def align(self, image: Image.Image) -> Optional[Image.Image]:
        """Align the first detected face"""
        try:
            bboxes, landmarks = self.detect_faces(image)
            
            if len(landmarks) == 0:
                return None
            
            # Use first detection
            landmark = landmarks[0]
            
            # Convert landmarks to 5-point format expected by warp_and_crop_face
            facial5points = [[landmark[i*2], landmark[i*2+1]] for i in range(5)]
            
            # Apply alignment
            warped_face = warp_and_crop_face(
                np.array(image), 
                facial5points, 
                self.reference_points, 
                crop_size=self.crop_size
            )
            
            return Image.fromarray(warped_face)
            
        except Exception as e:
            print(f"MediaPipe alignment failed: {e}")
            return None

    def align_multi(self, image: Image.Image, limit: Optional[int] = None) -> Tuple[np.ndarray, List[Image.Image]]:
        """Align multiple faces in the image"""
        try:
            bboxes, landmarks = self.detect_faces(image)
            
            if limit:
                bboxes = bboxes[:limit]
                landmarks = landmarks[:limit]
            
            aligned_faces = []
            for landmark in landmarks:
                try:
                    # Convert landmarks to 5-point format
                    facial5points = [[landmark[i*2], landmark[i*2+1]] for i in range(5)]
                    
                    # Apply alignment
                    warped_face = warp_and_crop_face(
                        np.array(image), 
                        facial5points, 
                        self.reference_points, 
                        crop_size=self.crop_size
                    )
                    
                    aligned_faces.append(Image.fromarray(warped_face))
                except Exception as e:
                    print(f"Failed to align face: {e}")
                    continue
            
            return bboxes, aligned_faces
            
        except Exception as e:
            print(f"MediaPipe multi-alignment failed: {e}")
            return np.array([]), []


# Alternative MediaPipe implementation using only face detection
class MediaPipeFaceDetection:
    """Simplified MediaPipe using only face detection (not face mesh)"""
    
    def __init__(self, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        self.device = device
        self.crop_size = crop_size
        
        if not MEDIAPIPE_AVAILABLE:
            self.face_detection = None
            return
        
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))

    def detect_faces(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Detect faces and create approximate landmarks"""
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            return np.array([]), np.array([])
        
        img_array = np.array(image)
        img_rgb = cv.cvtColor(img_array, cv.COLOR_RGB2BGR) if len(img_array.shape) == 3 else img_array
        img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
        
        height, width = img_rgb.shape[:2]
        
        try:
            results = self.face_detection.process(img_rgb)
            
            bboxes = []
            landmarks = []
            
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * width)
                    y1 = int(bbox.ymin * height)
                    x2 = int((bbox.xmin + bbox.width) * width)
                    y2 = int((bbox.ymin + bbox.height) * height)
                    conf = detection.score[0] if detection.score else 0.9
                    
                    bboxes.append([x1, y1, x2, y2, conf])
                    
                    # Create approximate landmarks from bbox
                    face_width = x2 - x1
                    face_height = y2 - y1
                    
                    # Approximate landmark positions
                    face_landmarks = np.array([
                        [x1 + 0.3 * face_width, y1 + 0.4 * face_height],  # Left eye
                        [x1 + 0.7 * face_width, y1 + 0.4 * face_height],  # Right eye
                        [x1 + 0.5 * face_width, y1 + 0.6 * face_height],  # Nose
                        [x1 + 0.35 * face_width, y1 + 0.8 * face_height], # Left mouth
                        [x1 + 0.65 * face_width, y1 + 0.8 * face_height]  # Right mouth
                    ])
                    
                    landmarks.append(face_landmarks.flatten())
            
            return np.array(bboxes), np.array(landmarks)
            
        except Exception as e:
            print(f"MediaPipe face detection failed: {e}")
            return np.array([]), np.array([])

    def align(self, image: Image.Image) -> Optional[Image.Image]:
        """Align using approximate landmarks"""
        try:
            bboxes, landmarks = self.detect_faces(image)
            
            if len(landmarks) == 0:
                return None
            
            landmark = landmarks[0]
            facial5points = [[landmark[i*2], landmark[i*2+1]] for i in range(5)]
            
            warped_face = warp_and_crop_face(
                np.array(image), 
                facial5points, 
                self.reference_points, 
                crop_size=self.crop_size
            )
            
            return Image.fromarray(warped_face)
            
        except Exception as e:
            print(f"MediaPipe alignment failed: {e}")
            return None

    def align_multi(self, image: Image.Image, limit: Optional[int] = None) -> Tuple[np.ndarray, List[Image.Image]]:
        """Align multiple faces using approximate landmarks"""
        try:
            bboxes, landmarks = self.detect_faces(image)
            
            if limit:
                bboxes = bboxes[:limit]
                landmarks = landmarks[:limit]
            
            aligned_faces = []
            for landmark in landmarks:
                try:
                    facial5points = [[landmark[i*2], landmark[i*2+1]] for i in range(5)]
                    
                    warped_face = warp_and_crop_face(
                        np.array(image), 
                        facial5points, 
                        self.reference_points, 
                        crop_size=self.crop_size
                    )
                    
                    aligned_faces.append(Image.fromarray(warped_face))
                except Exception as e:
                    print(f"Failed to align face: {e}")
                    continue
            
            return bboxes, aligned_faces
            
        except Exception as e:
            print(f"MediaPipe multi-alignment failed: {e}")
            return np.array([]), []