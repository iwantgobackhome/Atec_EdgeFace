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


class RetinaFaceDetector:
    def __init__(self, model_path: str = None, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        """
        RetinaFace ONNX detector for face detection and 5-point landmark extraction
        Args:
            model_path: Path to ONNX model file
            device: Device to use ('cpu' or 'cuda')
            crop_size: Output crop size for aligned faces
        """
        self.device = device
        self.crop_size = crop_size
        
        if model_path is None:
            model_path = self._download_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RetinaFace ONNX model not found: {model_path}")
        
        # Initialize ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = (640, 640)  # Standard RetinaFace input size
        
        # Get reference points for alignment
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))
        
        # Detection parameters
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

    def _download_model(self) -> str:
        """Download RetinaFace ONNX model if not exists"""
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'retinaface_r50_v1.onnx')
        
        if not os.path.exists(model_path):
            print("Downloading RetinaFace ONNX model...")
            # Download from a reliable source
            url = "https://github.com/onnx/models/raw/main/vision/body_analysis/retinaface/model/retinaface-r50.onnx"
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Model downloaded: {model_path}")
            except Exception as e:
                print(f"Failed to download model: {e}")
                print("Please manually download RetinaFace ONNX model to:", model_path)
                raise
        
        return model_path

    def _preprocess_image(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for RetinaFace inference"""
        orig_size = image.size  # (width, height)
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv.cvtColor(img_array, cv.COLOR_RGB2BGR)
        
        # Resize to model input size
        img_resized = cv.resize(img_array, self.input_size)
        
        # Calculate scale factor
        scale = min(self.input_size[0] / orig_size[0], self.input_size[1] / orig_size[1])
        
        # Normalize
        img_normalized = img_resized.astype(np.float32)
        img_normalized = (img_normalized - 127.5) / 128.0
        
        # Add batch dimension and transpose to NCHW
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input, scale, orig_size

    def _postprocess_outputs(self, outputs: List[np.ndarray], scale: float, orig_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Post-process RetinaFace outputs to get bboxes and landmarks"""
        # This is a simplified implementation - actual RetinaFace has multiple anchor scales
        # For a complete implementation, you'd need to handle all anchor levels
        
        # Extract predictions (assuming single output format)
        predictions = outputs[0][0]  # Remove batch dimension
        
        boxes = []
        landmarks = []
        scores = []
        
        for pred in predictions:
            if len(pred) >= 15:  # bbox(4) + score(1) + landmarks(10)
                x1, y1, x2, y2, conf = pred[:5]
                
                if conf > self.conf_threshold:
                    # Scale back to original image size
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale
                    
                    # Extract landmarks (5 points * 2 coordinates)
                    lmk = pred[5:15].reshape(5, 2)
                    lmk[:, 0] = lmk[:, 0] / scale  # x coordinates
                    lmk[:, 1] = lmk[:, 1] / scale  # y coordinates
                    
                    boxes.append([x1, y1, x2, y2, conf])
                    landmarks.append(lmk.flatten())  # Flatten to [x1,y1,x2,y2,...,x5,y5]
                    scores.append(conf)
        
        if len(boxes) == 0:
            return np.array([]), np.array([])
        
        boxes = np.array(boxes)
        landmarks = np.array(landmarks)
        
        # Apply NMS
        indices = cv.dnn.NMSBoxes(
            boxes[:, :4].tolist(),
            scores,
            self.conf_threshold,
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = boxes[indices]
            landmarks = landmarks[indices]
        
        return boxes, landmarks

    def detect_faces(self, image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faces and landmarks in image
        Returns:
            bboxes: Array of shape [N, 5] with [x1, y1, x2, y2, conf]
            landmarks: Array of shape [N, 10] with [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]
        """
        # Preprocess
        img_input, scale, orig_size = self._preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_input})
        
        # Postprocess
        bboxes, landmarks = self._postprocess_outputs(outputs, scale, orig_size)
        
        return bboxes, landmarks

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
            print(f"RetinaFace alignment failed: {e}")
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
            print(f"RetinaFace multi-alignment failed: {e}")
            return np.array([]), []


# Alternative simplified RetinaFace using OpenCV DNN
class RetinaFaceOpenCV:
    """Simplified RetinaFace using OpenCV DNN backend"""
    
    def __init__(self, device: str = 'cpu', crop_size: Tuple[int, int] = (112, 112)):
        self.device = device
        self.crop_size = crop_size
        self.reference_points = get_reference_facial_points(default_square=(crop_size[0] == crop_size[1]))
        
        # For now, we'll use a placeholder - actual RetinaFace model would be loaded here
        print("RetinaFaceOpenCV: Using placeholder implementation")
        print("For full RetinaFace support, use RetinaFaceDetector with ONNX model")

    def align(self, image: Image.Image) -> Optional[Image.Image]:
        """Placeholder alignment method"""
        print("RetinaFaceOpenCV: Placeholder alignment - returning None")
        return None

    def align_multi(self, image: Image.Image, limit: Optional[int] = None) -> Tuple[np.ndarray, List[Image.Image]]:
        """Placeholder multi-alignment method"""
        print("RetinaFaceOpenCV: Placeholder multi-alignment - returning empty results")
        return np.array([]), []