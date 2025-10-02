"""
Unified Face Alignment Detector
Provides a common interface for all face detection and alignment methods
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Union

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import all detector classes
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("MTCNN not available")

try:
    from yunet import YuNetDetector
    YUNET_AVAILABLE = True
except ImportError:
    YUNET_AVAILABLE = False
    print("YuNet not available")

try:
    from retinaface_onnx import RetinaFaceDetector
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False
    print("RetinaFace not available")

try:
    from rtmpose_detector import RTMPoseDetector
    RTMPOSE_AVAILABLE = True
except ImportError:
    RTMPOSE_AVAILABLE = False
    print("RTMPose not available")

try:
    from mediapipe_detector import MediaPipeDetector, MediaPipeFaceDetection
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available")

try:
    from yolo_detector import YOLOFaceDetector, YOLOUltralyticsDetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO not available")


class UnifiedFaceDetector:
    """
    Unified interface for multiple face detection and alignment methods
    
    Supported methods:
    - MTCNN: Multi-task CNN (PyTorch)
    - YuNet: OpenCV's fast face detection
    - RetinaFace: ONNX-based RetinaFace
    - RTMPose: Face keypoint detection
    - MediaPipe: Google's face detection/mesh
    - YOLO: YOLO-based face detection
    """
    
    AVAILABLE_METHODS = {
        'mtcnn': MTCNN_AVAILABLE,
        'yunet': YUNET_AVAILABLE,
        'retinaface': RETINAFACE_AVAILABLE,
        'rtmpose': RTMPOSE_AVAILABLE,
        'mediapipe': MEDIAPIPE_AVAILABLE,
        'mediapipe_simple': MEDIAPIPE_AVAILABLE,
        'yolov5_face': YOLO_AVAILABLE,  # YOLOv5-Face (face-specific ONNX)
        'yolov8': YOLO_AVAILABLE         # YOLOv8 (general object detection)
    }
    
    def __init__(self, method: str, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112), **kwargs):
        """
        Initialize unified face detector
        
        Args:
            method: Detection method to use
            device: Device to run on ('cpu' or 'cuda')
            crop_size: Output crop size for aligned faces
            **kwargs: Method-specific parameters
        """
        self.method = method.lower()
        self.device = device
        self.crop_size = crop_size
        self.detector = None
        self.available = False
        
        if self.method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.AVAILABLE_METHODS.keys())}")
        
        if not self.AVAILABLE_METHODS[self.method]:
            raise ImportError(f"Method {method} is not available (missing dependencies)")
        
        # Initialize the specific detector
        self._init_detector(**kwargs)
    
    def _init_detector(self, **kwargs):
        """Initialize the specific detector based on method"""
        try:
            if self.method == 'mtcnn':
                self.detector = MTCNN(device=self.device, crop_size=self.crop_size)
                self.available = True
                
            elif self.method == 'yunet':
                # Set default YuNet model path if not provided
                default_yunet_path = os.path.join(os.path.dirname(__file__), 'models', 'face_detection_yunet_2023mar.onnx')
                model_path = kwargs.get('model_path', default_yunet_path)
                self.detector = YuNetDetector(model_path=model_path, device=self.device, crop_size=self.crop_size)
                self.available = True
                
            elif self.method == 'retinaface':
                model_path = kwargs.get('model_path', None)
                self.detector = RetinaFaceDetector(model_path=model_path, device=self.device, crop_size=self.crop_size)
                self.available = True
                
            elif self.method == 'rtmpose':
                model_path = kwargs.get('model_path', None)
                self.detector = RTMPoseDetector(model_path=model_path, device=self.device, crop_size=self.crop_size)
                self.available = True
                
            elif self.method == 'mediapipe':
                self.detector = MediaPipeDetector(device=self.device, crop_size=self.crop_size)
                self.available = True
                
            elif self.method == 'mediapipe_simple':
                self.detector = MediaPipeFaceDetection(device=self.device, crop_size=self.crop_size)
                self.available = True
                
            elif self.method == 'yolov5_face':
                model_path = kwargs.get('model_path', None)
                self.detector = YOLOFaceDetector(model_path=model_path, device=self.device, crop_size=self.crop_size)
                self.available = True

            elif self.method == 'yolov8':
                default_model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolov8n.pt')
                model_path = kwargs.get('model_path', default_model_path)
                self.detector = YOLOUltralyticsDetector(model_path=model_path, device=self.device, crop_size=self.crop_size)
                self.available = True
                
        except Exception as e:
            print(f"Failed to initialize {self.method}: {e}")
            self.available = False
    
    def detect_faces(self, image: Union[str, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces and landmarks
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            bboxes: Array of shape [N, 5] with [x1, y1, x2, y2, conf]
            landmarks: Array of shape [N, 10] with [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
        """
        if not self.available:
            return np.array([]), np.array([])
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if hasattr(self.detector, 'detect_faces'):
            # Handle MTCNN's specific parameter requirements
            if self.method == 'mtcnn':
                try:
                    return self.detector.detect_faces(
                        image, 
                        self.detector.min_face_size, 
                        self.detector.thresholds, 
                        self.detector.nms_thresholds, 
                        self.detector.factor
                    )
                except Exception as e:
                    print(f"MTCNN detect_faces error: {e}")
                    return np.array([]), np.array([])
            elif self.method == 'yunet':
                # YuNet returns faces array, convert to bboxes and landmarks
                try:
                    faces = self.detector.detect_faces(image)
                    if faces is None or len(faces) == 0:
                        return np.array([]), np.array([])
                    
                    # Convert YuNet format to unified format
                    bboxes = []
                    landmarks = []
                    for face in faces:
                        # face format: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]
                        if len(face) < 15:  # Skip invalid faces
                            continue
                            
                        x, y, w, h = face[0], face[1], face[2], face[3]
                        conf = face[14] if len(face) > 14 else 0.9
                        bboxes.append([x, y, x+w, y+h, conf])
                        
                        # Extract landmarks (5 points)
                        lmks = []
                        for i in range(5):
                            lmks.extend([face[4 + i*2], face[5 + i*2]])
                        landmarks.append(lmks)
                    
                    return np.array(bboxes), np.array(landmarks)
                except Exception as e:
                    print(f"YuNet detect_faces error: {e}")
                    return np.array([]), np.array([])
            else:
                return self.detector.detect_faces(image)
        else:
            print(f"Method {self.method} does not support detect_faces")
            return np.array([]), np.array([])
    
    def align(self, image: Union[str, Image.Image]) -> Optional[Image.Image]:
        """
        Align the first detected face
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Aligned face image or None if no face detected
        """
        if not self.available:
            return None
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        return self.detector.align(image)
    
    def align_multi(self, image: Union[str, Image.Image], limit: Optional[int] = None) -> Tuple[np.ndarray, List[Image.Image]]:
        """
        Align multiple faces in image
        
        Args:
            image: PIL Image or path to image
            limit: Maximum number of faces to align
            
        Returns:
            bboxes: Array of detected face bounding boxes
            aligned_faces: List of aligned face images
        """
        if not self.available:
            return np.array([]), []
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        return self.detector.align_multi(image, limit=limit)
    
    def benchmark_speed(self, image: Union[str, Image.Image], num_runs: int = 10) -> dict:
        """
        Benchmark detection and alignment speed
        
        Args:
            image: Test image
            num_runs: Number of test runs
            
        Returns:
            Dictionary with timing statistics
        """
        if not self.available:
            return {'error': f'Method {self.method} not available'}
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Benchmark detection
        detect_times = []
        detect_successes = 0
        
        for i in range(num_runs):
            try:
                start_time = time.time()
                bboxes, landmarks = self.detect_faces(image)
                end_time = time.time()
                detect_times.append(end_time - start_time)
                detect_successes += 1
            except Exception as e:
                print(f"Detection run {i+1} failed for {self.method}: {e}")
                continue
        
        # Benchmark alignment
        align_times = []
        align_successes = 0
        
        for i in range(num_runs):
            try:
                start_time = time.time()
                aligned_face = self.align(image)
                end_time = time.time()
                align_times.append(end_time - start_time)
                align_successes += 1
            except Exception as e:
                print(f"Alignment run {i+1} failed for {self.method}: {e}")
                continue
        
        # Calculate safe statistics
        def safe_stats(times_list):
            if len(times_list) == 0:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'mean': np.mean(times_list),
                'std': np.std(times_list),
                'min': np.min(times_list),
                'max': np.max(times_list)
            }
        
        detect_stats = safe_stats(detect_times)
        align_stats = safe_stats(align_times)
        
        return {
            'method': self.method,
            'device': self.device,
            'detection_time': detect_stats,
            'alignment_time': align_stats,
            'total_time': {
                'mean': detect_stats['mean'] + align_stats['mean'],
                'std': np.sqrt(detect_stats['std']**2 + align_stats['std']**2)
            },
            'success_rates': {
                'detection': detect_successes / num_runs,
                'alignment': align_successes / num_runs
            }
        }
    
    @classmethod
    def list_available_methods(cls) -> List[str]:
        """List all available detection methods"""
        return [method for method, available in cls.AVAILABLE_METHODS.items() if available]
    
    @classmethod
    def check_dependencies(cls) -> dict:
        """Check which methods have their dependencies satisfied"""
        return cls.AVAILABLE_METHODS.copy()
    
    def __str__(self) -> str:
        return f"UnifiedFaceDetector(method={self.method}, device={self.device}, available={self.available})"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_detector_ensemble(methods: List[str], device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)) -> dict:
    """
    Create multiple detectors for comparison
    
    Args:
        methods: List of detection methods to create
        device: Device to use
        crop_size: Output crop size
        
    Returns:
        Dictionary mapping method names to detector instances
    """
    detectors = {}
    
    for method in methods:
        try:
            detector = UnifiedFaceDetector(method, device=device, crop_size=crop_size)
            if detector.available:
                detectors[method] = detector
            else:
                print(f"Failed to create detector for {method}")
        except Exception as e:
            print(f"Error creating {method} detector: {e}")
    
    return detectors


def benchmark_all_methods(image: Union[str, Image.Image], methods: Optional[List[str]] = None, 
                         device: str = 'cpu', num_runs: int = 5) -> dict:
    """
    Benchmark all available methods
    
    Args:
        image: Test image
        methods: List of methods to benchmark (None for all available)
        device: Device to use
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results for each method
    """
    if methods is None:
        methods = UnifiedFaceDetector.list_available_methods()
    
    results = {}
    
    for method in methods:
        try:
            detector = UnifiedFaceDetector(method, device=device)
            if detector.available:
                print(f"Benchmarking {method}...")
                results[method] = detector.benchmark_speed(image, num_runs=num_runs)
            else:
                results[method] = {'error': f'Method {method} not available'}
        except Exception as e:
            results[method] = {'error': str(e)}
    
    return results