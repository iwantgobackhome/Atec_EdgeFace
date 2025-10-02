import cv2 as cv
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import sys
from typing import Tuple, List, Optional
import requests

sys.path.insert(0, os.path.dirname(__file__))
from align_trans import get_reference_facial_points, warp_and_crop_face


class RTMPoseDetector:
    """RTMPose detector for face keypoint extraction and alignment"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        """
        RTMPose detector for face alignment
        Args:
            model_path: Path to RTMPose ONNX model
            device: Device to use ('cpu' or 'cuda')
            crop_size: Output crop size for aligned faces
        """
        self.device = device
        self.crop_size = crop_size
        
        if model_path is None:
            model_path = self._get_model_path()
        
        if not os.path.exists(model_path):
            print(f"RTMPose model not found: {model_path}")
            print("Please download RTMPose model manually")
            self.session = None
        else:
            # Initialize ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            
            # Get model input info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
        # Get reference points for alignment
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))
        
        # Face detection for preprocessing (using OpenCV's DNN)
        self._init_face_detector()

    def _get_model_path(self) -> str:
        """Get RTMPose model path"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if model exists in Downloads directory first
        downloads_path = "/mnt/c/Users/Admin/Downloads"
        potential_models = [
            "rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.onnx",
            "rtmw-x_simcc-cocktail14_pt-ucoco_270e-256x192-13a2546d_20231208.onnx",
            "rtmpose_face.onnx"
        ]
        
        for model_name in potential_models:
            full_path = os.path.join(downloads_path, model_name)
            if os.path.exists(full_path):
                return full_path
                
        # If not found, return expected path in models directory
        return os.path.join(model_dir, 'rtmpose_face.onnx')

    def _init_face_detector(self):
        """Initialize face detector for preprocessing"""
        try:
            # Try to use YuNet if available
            yunet_path = os.path.join(os.path.dirname(__file__), 'models', 'face_detection_yunet_2023mar.onnx')
            if os.path.exists(yunet_path):
                self.face_detector = cv.FaceDetectorYN.create(
                    model=yunet_path, config="", input_size=(320, 320),
                    score_threshold=0.6, nms_threshold=0.3, top_k=5000,
                    backend_id=cv.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv.dnn.DNN_TARGET_CPU
                )
            else:
                # Fallback to OpenCV Haar Cascade
                cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv.CascadeClassifier(cascade_path)
                
        except Exception as e:
            print(f"Failed to initialize face detector: {e}")
            self.face_detector = None

    def _detect_face_bbox(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box for RTMPose preprocessing"""
        if self.face_detector is None:
            return None
            
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
        
        try:
            # Try YuNet first
            if hasattr(self.face_detector, 'detect'):
                self.face_detector.setInputSize((img_array.shape[1], img_array.shape[0]))
                _, faces = self.face_detector.detect(img_array)
                
                if faces is not None and len(faces) > 0:
                    x, y, w, h = faces[0][:4].astype(int)
                    return (x, y, x + w, y + h)
            else:
                # Fallback to Haar Cascade
                gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    return (x, y, x + w, y + h)
                    
        except Exception as e:
            print(f"Face detection failed: {e}")
            
        return None

    def _preprocess_for_rtmpose(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Preprocess image for RTMPose inference"""
        x1, y1, x2, y2 = bbox
        
        # Crop face region with some padding
        padding = max((x2 - x1), (y2 - y1)) * 0.3
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(image.width, int(x2 + padding))
        y2 = min(image.height, int(y2 + padding))
        
        # Crop and resize
        face_crop = image.crop((x1, y1, x2, y2))
        
        # Resize to model input size (typically 256x192 or 384x288)
        if self.session is not None:
            input_height = self.input_shape[2] if len(self.input_shape) > 2 else 256
            input_width = self.input_shape[3] if len(self.input_shape) > 3 else 192
        else:
            input_height, input_width = 256, 192
            
        face_resized = face_crop.resize((input_width, input_height))
        
        # Convert to numpy and normalize
        img_array = np.array(face_resized, dtype=np.float32)
        img_array = img_array / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Transpose to NCHW and add batch dimension
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def _extract_face_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract 5-point face landmarks from RTMPose keypoints
        RTMPose typically outputs 133 keypoints including face landmarks
        Face keypoint indices (approximate):
        - Left eye: around index 33-42
        - Right eye: around index 362-371  
        - Nose tip: around index 1
        - Mouth corners: around indices 61, 291
        """
        if len(keypoints) < 133:
            # Fallback: create approximate landmarks from bounding box
            return self._create_default_landmarks()
        
        try:
            # Extract approximate face landmarks (indices may vary by model)
            left_eye = keypoints[36:37].mean(axis=0)  # Left eye center
            right_eye = keypoints[45:46].mean(axis=0)  # Right eye center
            nose = keypoints[30]  # Nose tip
            mouth_left = keypoints[48]  # Left mouth corner
            mouth_right = keypoints[54]  # Right mouth corner
            
            # Create 5-point landmarks
            face_landmarks = np.array([
                left_eye, right_eye, nose, mouth_left, mouth_right
            ])
            
            return face_landmarks.flatten()  # [x1,y1,x2,y2,...,x5,y5]
            
        except Exception as e:
            print(f"Failed to extract face keypoints: {e}")
            return self._create_default_landmarks()

    def _create_default_landmarks(self) -> np.ndarray:
        """Create default 5-point landmarks when keypoint extraction fails"""
        # Return standard face landmark positions (normalized 0-1)
        default_landmarks = np.array([
            [0.3, 0.4],   # Left eye
            [0.7, 0.4],   # Right eye  
            [0.5, 0.6],   # Nose
            [0.35, 0.8],  # Left mouth
            [0.65, 0.8]   # Right mouth
        ])
        return default_landmarks.flatten()

    def detect_faces(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces and extract keypoints
        Returns:
            bboxes: Array of shape [N, 5] with [x1, y1, x2, y2, conf]
            landmarks: Array of shape [N, 10] with [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
        """
        if self.session is None:
            print("RTMPose model not available")
            return np.array([]), np.array([])
        
        # Detect face bounding box
        bbox = self._detect_face_bbox(image)
        if bbox is None:
            return np.array([]), np.array([])
        
        try:
            # Preprocess for RTMPose
            img_input = self._preprocess_for_rtmpose(image, bbox)
            
            # Run RTMPose inference
            outputs = self.session.run(None, {self.input_name: img_input})
            
            # Extract keypoints from outputs
            keypoints = outputs[0][0]  # Remove batch dimension
            
            # Extract face landmarks
            face_landmarks = self._extract_face_keypoints(keypoints)
            
            # Scale landmarks back to original image coordinates
            x1, y1, x2, y2 = bbox
            face_landmarks_scaled = face_landmarks.copy()
            face_landmarks_scaled[::2] = face_landmarks_scaled[::2] * (x2 - x1) + x1  # x coordinates
            face_landmarks_scaled[1::2] = face_landmarks_scaled[1::2] * (y2 - y1) + y1  # y coordinates
            
            # Create bounding box with confidence
            bbox_with_conf = np.array([[x1, y1, x2, y2, 0.9]])
            landmarks_array = np.array([face_landmarks_scaled])
            
            return bbox_with_conf, landmarks_array
            
        except Exception as e:
            print(f"RTMPose inference failed: {e}")
            return np.array([]), np.array([])

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
            print(f"RTMPose alignment failed: {e}")
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
            print(f"RTMPose multi-alignment failed: {e}")
            return np.array([]), []