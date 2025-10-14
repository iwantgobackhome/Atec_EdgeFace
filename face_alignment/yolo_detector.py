import cv2 as cv
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import sys
from typing import Tuple, List, Optional
import requests

sys.path.insert(0, os.path.dirname(__file__))
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face


class YOLOFaceDetector:
    """YOLO-based face detector for face detection and alignment"""
    
    def __init__(self, model_path: str = None, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        """
        YOLO Face detector
        Args:
            model_path: Path to YOLO face detection model
            device: Device to use ('cpu' or 'cuda')
            crop_size: Output crop size for aligned faces
        """
        self.device = device
        self.crop_size = crop_size
        
        if model_path is None:
            model_path = self._get_model_path()

        self.model_path = model_path  # Store for potential CPU fallback

        if not os.path.exists(model_path):
            print(f"YOLO model not found: {model_path}")
            self.session = None
        else:
            try:
                # Initialize ONNX Runtime session with GPU support
                available_providers = ort.get_available_providers()

                if device == 'cuda':
                    if 'CUDAExecutionProvider' in available_providers:
                        # Use CUDA with fallback on error
                        cuda_options = {
                            'device_id': 0,
                            'arena_extend_strategy': 'kNextPowerOfTwo',
                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                            'do_copy_in_default_stream': True,
                        }
                        providers = [
                            ('CUDAExecutionProvider', cuda_options),
                            'CPUExecutionProvider'
                        ]
                        print(f"YOLOFaceDetector: Attempting CUDA GPU (with CPU fallback)")
                    else:
                        providers = ['CPUExecutionProvider']
                        print(f"⚠️ YOLOFaceDetector: CUDA requested but not available, using CPU")
                        print(f"   Available providers: {available_providers}")
                else:
                    providers = ['CPUExecutionProvider']

                try:
                    self.session = ort.InferenceSession(model_path, providers=providers)
                except Exception as cuda_error:
                    if device == 'cuda':
                        print(f"⚠️ CUDA initialization failed: {str(cuda_error)[:100]}")
                        print(f"   Falling back to CPU execution")
                        providers = ['CPUExecutionProvider']
                        self.session = ort.InferenceSession(model_path, providers=providers)
                    else:
                        raise

                # Verify which provider is actually being used
                actual_providers = self.session.get_providers()
                print(f"YOLOFaceDetector: Active providers: {actual_providers}")

                # Get model input info
                self.input_name = self.session.get_inputs()[0].name
                self.input_shape = self.session.get_inputs()[0].shape
                self.input_height = self.input_shape[2] if len(self.input_shape) > 2 else 640
                self.input_width = self.input_shape[3] if len(self.input_shape) > 3 else 640

            except Exception as e:
                print(f"Failed to initialize YOLO model: {e}")
                self.session = None
        
        # Get reference points for alignment
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))
        
        # Detection parameters
        self.conf_threshold = 0.7  # 더 높은 confidence threshold
        self.nms_threshold = 0.4

    def _get_model_path(self) -> str:
        """Get YOLO model path - check for common YOLO face models"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Check common YOLO face model names
        model_names = [
            'yolov5_face.onnx',
            'yolov8_face.onnx',
            'yolo_face.onnx',
            'yolov5s-face.onnx',
            'yolov8n-face.onnx'
        ]
        
        for model_name in model_names:
            model_path = os.path.join(model_dir, model_name)
            if os.path.exists(model_path):
                return model_path
        
        # Return default path
        return os.path.join(model_dir, 'yolov5_face.onnx')

    def _preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for YOLO inference"""
        orig_size = image.size  # (width, height)
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
        
        # Resize to model input size while maintaining aspect ratio
        scale = min(self.input_width / orig_size[0], self.input_height / orig_size[1])
        new_width = int(orig_size[0] * scale)
        new_height = int(orig_size[1] * scale)
        
        img_resized = cv.resize(img_array, (new_width, new_height))
        
        # Pad to input size
        pad_width = (self.input_width - new_width) // 2
        pad_height = (self.input_height - new_height) // 2
        
        img_padded = cv.copyMakeBorder(
            img_resized,
            pad_height, self.input_height - new_height - pad_height,
            pad_width, self.input_width - new_width - pad_width,
            cv.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        # Convert BGR to RGB and normalize
        img_rgb = cv.cvtColor(img_padded, cv.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to NCHW and add batch dimension
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input, scale, orig_size, (pad_width, pad_height)

    def _postprocess_outputs(self, outputs: List[np.ndarray], scale: float, orig_size: Tuple[int, int], padding: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process YOLO outputs to get bboxes"""
        if self.session is None or len(outputs) == 0:
            return np.array([]), np.array([])
        
        # YOLOv8 pose output: (1, 20, 8400) -> (20, 8400)
        predictions = outputs[0]  # Shape: (20, 8400)
        
        # Transpose to (8400, 20) for easier processing
        predictions = predictions.T  # Shape: (8400, 20)
        
        pad_width, pad_height = padding
        
        boxes = []
        scores = []
        keypoints_list = []
        
        # Process predictions
        for pred in predictions:
            if len(pred) >= 5:  # x, y, w, h, conf, keypoints...
                x_center, y_center, width, height, confidence = pred[:5]
                
                if confidence > self.conf_threshold:
                    # Convert from model space to original image space
                    # Remove padding first, then scale
                    x_center = (x_center - pad_width) / scale
                    y_center = (y_center - pad_height) / scale
                    width = width / scale
                    height = height / scale
                    
                    # Convert to x1, y1, x2, y2 format
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, orig_size[0]))
                    y1 = max(0, min(y1, orig_size[1]))
                    x2 = max(0, min(x2, orig_size[0]))
                    y2 = max(0, min(y2, orig_size[1]))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Convert to Python floats to ensure consistent types
                    boxes.append([float(x1), float(y1), float(x2), float(y2), float(confidence)])
                    scores.append(float(confidence))
                    
                    # Extract keypoints if available (pose model has 15 keypoints after confidence)
                    kpts = []
                    if len(pred) >= 20:  # 5 + 15 keypoints
                        try:
                            for i in range(5):  # 5 face keypoints
                                kx = pred[5 + i*3]     # x coordinate
                                ky = pred[5 + i*3 + 1] # y coordinate
                                # kv = pred[5 + i*3 + 2] # visibility (not used)

                                # Transform keypoints
                                kx = (kx - pad_width) / scale
                                ky = (ky - pad_height) / scale

                                # Convert to Python floats
                                kpts.extend([float(kx), float(ky)])
                        except (IndexError, ValueError):
                            # Fallback if keypoint extraction fails
                            kpts = self._create_landmarks_from_bbox(x1, y1, x2, y2)
                    else:
                        # Fallback: create approximate keypoints from bbox
                        kpts = self._create_landmarks_from_bbox(x1, y1, x2, y2)
                    
                    # Ensure all keypoint arrays have exactly 10 elements (5 points * 2 coordinates)
                    if len(kpts) != 10:
                        kpts = self._create_landmarks_from_bbox(x1, y1, x2, y2)
                    
                    keypoints_list.append(kpts)
        
        if len(boxes) == 0:
            return np.array([]), np.array([])

        # Apply NMS BEFORE converting to numpy arrays
        try:
            # Convert boxes to proper format for NMS
            boxes_for_nms = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in boxes]  # x, y, w, h
            indices = cv.dnn.NMSBoxes(
                boxes_for_nms,
                [float(s) for s in scores],
                self.conf_threshold,
                self.nms_threshold
            )

            if len(indices) > 0:
                indices = indices.flatten()
                # Filter boxes and keypoints using indices
                boxes = [boxes[i] for i in indices]
                keypoints_list = [keypoints_list[i] for i in indices]
        except Exception as e:
            print(f"NMS failed: {e}")
            # Continue without NMS if it fails

        # Convert boxes to numpy array
        try:
            boxes = np.array(boxes, dtype=np.float32)
            # Ensure shape is (N, 5)
            if boxes.ndim == 1:
                boxes = boxes.reshape(1, -1)
            elif boxes.ndim == 3:  # (N, 5, 1) -> (N, 5)
                boxes = boxes.squeeze(axis=-1)
        except ValueError as e:
            print(f"YOLO detection failed: Box conversion error: {e}")
            return np.array([]), np.array([])

        # Convert keypoints to numpy array with proper shape validation
        if len(keypoints_list) > 0:
            try:
                # Ensure all keypoints are lists/arrays of exactly 10 elements
                validated_keypoints = []
                for i, kpts in enumerate(keypoints_list):
                    if isinstance(kpts, np.ndarray):
                        kpts = kpts.flatten().tolist()

                    if len(kpts) != 10:
                        # Recreate from bbox if wrong length
                        if i < len(boxes):
                            bbox = boxes[i]
                            kpts = self._create_landmarks_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3])

                    validated_keypoints.append(kpts)

                keypoints_array = np.array(validated_keypoints, dtype=np.float32)

                # Ensure shape is (N, 10)
                if keypoints_array.ndim == 1:
                    keypoints_array = keypoints_array.reshape(1, -1)
                elif keypoints_array.ndim == 3:
                    keypoints_array = keypoints_array.squeeze(axis=-1)

            except (ValueError, TypeError) as e:
                print(f"YOLO detection failed: Keypoint conversion error: {e}")
                # Create fallback keypoints from bboxes
                fallback_keypoints = []
                for bbox in boxes:
                    if len(bbox) >= 4:
                        fallback_kpts = self._create_landmarks_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
                        fallback_keypoints.append(fallback_kpts)

                if fallback_keypoints:
                    keypoints_array = np.array(fallback_keypoints, dtype=np.float32)
                else:
                    keypoints_array = np.array([])
        else:
            keypoints_array = np.array([])
        
        return boxes, keypoints_array

    def _create_landmarks_from_bbox(self, x1: float, y1: float, x2: float, y2: float) -> List[float]:
        """Create approximate 5-point landmarks from bounding box"""
        width = x2 - x1
        height = y2 - y1
        
        # Approximate landmark positions relative to bounding box (return as flat list)
        landmarks = [
            x1 + 0.3 * width, y1 + 0.4 * height,  # Left eye
            x1 + 0.7 * width, y1 + 0.4 * height,  # Right eye
            x1 + 0.5 * width, y1 + 0.6 * height,  # Nose
            x1 + 0.35 * width, y1 + 0.8 * height, # Left mouth
            x1 + 0.65 * width, y1 + 0.8 * height  # Right mouth
        ]
        
        return landmarks

    def detect_faces(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces and create approximate landmarks
        Returns:
            bboxes: Array of shape [N, 5] with [x1, y1, x2, y2, conf]
            landmarks: Array of shape [N, 10] with [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
        """
        if self.session is None:
            print("YOLO model not available")
            return np.array([]), np.array([])

        try:
            # Preprocess
            img_input, scale, orig_size, padding = self._preprocess_image(image)

            # Run inference
            outputs = self.session.run(None, {self.input_name: img_input})

            # Postprocess
            bboxes, landmarks = self._postprocess_outputs(outputs, scale, orig_size, padding)

            return bboxes, landmarks

        except Exception as e:
            error_msg = str(e)
            # Check if it's a CUDA compute capability error
            if 'cudaErrorNoKernelImageForDevice' in error_msg or 'no kernel image is available' in error_msg:
                if not hasattr(self, '_cuda_fallback_warned'):
                    print(f"⚠️ CUDA execution failed (compute capability incompatibility)")
                    print(f"   Reinitializing with CPU execution provider...")
                    self._cuda_fallback_warned = True

                    # Reinitialize with CPU
                    try:
                        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                        print(f"   Successfully switched to CPU execution")

                        # Retry with CPU
                        outputs = self.session.run(None, {self.input_name: img_input})
                        bboxes, landmarks = self._postprocess_outputs(outputs, scale, orig_size, padding)
                        return bboxes, landmarks
                    except Exception as cpu_error:
                        print(f"   CPU fallback also failed: {cpu_error}")
                        return np.array([]), np.array([])
            else:
                print(f"YOLO detection failed: {e}")
            return np.array([]), np.array([])

    def align(self, image: Image.Image) -> Optional[Image.Image]:
        """Align the first detected face using approximate landmarks"""
        try:
            bboxes, landmarks = self.detect_faces(image)
            
            if len(landmarks) == 0:
                return None
            
            # Use first detection
            landmark = landmarks[0]
            
            # Flatten landmark if it has extra dimensions
            if landmark.ndim > 1:
                landmark = landmark.flatten()
            
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
            print(f"YOLO alignment failed: {e}")
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
                    # Flatten landmark if it has extra dimensions
                    if landmark.ndim > 1:
                        landmark = landmark.flatten()
                    
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
            print(f"YOLO multi-alignment failed: {e}")
            return np.array([]), []


# Alternative YOLO implementation using Ultralytics
class YOLOUltralyticsDetector:
    """YOLO detector using Ultralytics YOLO library"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        """
        Ultralytics YOLO detector
        Args:
            model_path: Path to YOLO model or model name
            device: Device to use
            crop_size: Output crop size
        """
        self.device = device
        self.crop_size = crop_size
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(device)
            ULTRALYTICS_AVAILABLE = True
        except ImportError:
            print("Ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
            ULTRALYTICS_AVAILABLE = False
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None
            ULTRALYTICS_AVAILABLE = False
        
        # Get reference points for alignment
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))

    def detect_faces(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Detect faces using Ultralytics YOLO"""
        if self.model is None:
            return np.array([]), np.array([])
        
        try:
            # Run detection
            results = self.model(image, verbose=False)
            
            bboxes = []
            landmarks = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confs = result.boxes.conf.cpu().numpy()  # confidence scores
                    
                    for box, conf in zip(boxes, confs):
                        if conf > 0.5:  # confidence threshold
                            x1, y1, x2, y2 = box
                            bboxes.append([x1, y1, x2, y2, conf])
                            
                            # Create approximate landmarks
                            landmark = self._create_landmarks_from_bbox(x1, y1, x2, y2)
                            landmarks.append(landmark)
            
            return np.array(bboxes), np.array(landmarks)
            
        except Exception as e:
            print(f"Ultralytics YOLO detection failed: {e}")
            return np.array([]), np.array([])

    def _create_landmarks_from_bbox(self, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
        """Create approximate landmarks from bbox"""
        width = x2 - x1
        height = y2 - y1
        
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
            print(f"YOLO Ultralytics alignment failed: {e}")
            return None

    def align_multi(self, image: Image.Image, limit: Optional[int] = None) -> Tuple[np.ndarray, List[Image.Image]]:
        """Align multiple faces"""
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
            print(f"YOLO Ultralytics multi-alignment failed: {e}")
            return np.array([]), []