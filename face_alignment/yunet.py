try:
    import cv2 as cv
except ImportError:
    print("OpenCV가 설치되지 않았습니다. pip install opencv-python을 실행하세요.")
    cv = None

try:
    import numpy as np
except ImportError:
    print("NumPy가 설치되지 않았습니다. pip install numpy를 실행하세요.")
    np = None

try:
    from PIL import Image
except ImportError:
    print("PIL이 설치되지 않았습니다. pip install Pillow를 실행하세요.")
    Image = None

import sys
import os

# MTCNN의 align_trans 모듈을 재사용
sys.path.insert(0, os.path.dirname(__file__))
try:
    from mtcnn_pytorch.src.align_trans import warp_and_crop_face
except ImportError:
    print("MTCNN align_trans 모듈을 가져올 수 없습니다.")
    warp_and_crop_face = None


class YuNetDetector:
    def __init__(self, model_path, device='cpu', crop_size=(112, 112)):
        """
        YuNet face detector with alignment capability
        
        Args:
            model_path: Path to YuNet ONNX model
            device: 'cpu' or 'cuda' (only cpu supported for YuNet)
            crop_size: Output face crop size (width, height)
        """
        # 필수 라이브러리 확인
        if cv is None:
            raise ImportError("OpenCV가 설치되지 않았습니다.")
        if np is None:
            raise ImportError("NumPy가 설치되지 않았습니다.")
        if Image is None:
            raise ImportError("PIL이 설치되지 않았습니다.")
        if warp_and_crop_face is None:
            raise ImportError("MTCNN align_trans 모듈을 사용할 수 없습니다.")
            
        self.device = device
        self.crop_size = crop_size
        self.model_path = model_path
        self._cuda_tried = False
        self._using_cuda = False

        # Determine backend and target based on device
        # Note: OpenCV DNN CUDA support requires OpenCV built with CUDA
        # Force CPU for YuNet since OpenCV CUDA DNN is often not available
        if device == 'cuda' or device.startswith('cuda:'):
            # Try CUDA first, but will fallback to CPU on first error
            backend_id = cv.dnn.DNN_BACKEND_CUDA
            target_id = cv.dnn.DNN_TARGET_CUDA
            self._using_cuda = True
        else:
            backend_id = cv.dnn.DNN_BACKEND_OPENCV
            target_id = cv.dnn.DNN_TARGET_CPU
            self._using_cuda = False

        self.backend_id = backend_id
        self.target_id = target_id

        # YuNet detector configuration
        try:
            self.detector = cv.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(320, 320),  # Default input size
                score_threshold=0.8,  # Increased from 0.6 to 0.8 for higher quality detections
                nms_threshold=0.3,
                top_k=5000,
                backend_id=self.backend_id,
                target_id=self.target_id
            )
            if self._using_cuda:
                print(f"YuNet: Attempting CUDA GPU acceleration")
            else:
                print(f"YuNet: Using CPU")
        except Exception as e:
            # If CUDA backend failed, try CPU fallback
            if self._using_cuda:
                print(f"YuNet: CUDA initialization failed, retrying with CPU...")
                self.backend_id = cv.dnn.DNN_BACKEND_OPENCV
                self.target_id = cv.dnn.DNN_TARGET_CPU
                self._using_cuda = False
                try:
                    self.detector = cv.FaceDetectorYN.create(
                        model=model_path,
                        config="",
                        input_size=(320, 320),
                        score_threshold=0.8,
                        nms_threshold=0.3,
                        top_k=5000,
                        backend_id=self.backend_id,
                        target_id=self.target_id
                    )
                    print(f"YuNet: Successfully initialized with CPU backend")
                except Exception as e2:
                    raise RuntimeError(f"YuNet detector 초기화 실패 (CUDA and CPU): {e2}")
            else:
                raise RuntimeError(f"YuNet detector 초기화 실패: {e}")

    def _reinit_with_cpu(self):
        """Reinitialize detector with CPU backend (fallback from CUDA)"""
        if not self._using_cuda or self._cuda_tried:
            return False

        print(f"YuNet: CUDA execution failed, switching to CPU backend...")
        self.backend_id = cv.dnn.DNN_BACKEND_OPENCV
        self.target_id = cv.dnn.DNN_TARGET_CPU
        self._using_cuda = False
        self._cuda_tried = True

        try:
            self.detector = cv.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=0.8,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=self.backend_id,
                target_id=self.target_id
            )
            print(f"YuNet: Successfully switched to CPU backend")
            return True
        except Exception as e:
            print(f"YuNet: CPU fallback also failed: {e}")
            return False
        
    def detect_faces(self, image):
        """
        Detect faces and landmarks using YuNet
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            faces: list of detected faces with landmarks
                   each face: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]
        """
        if isinstance(image, Image.Image):
            cv_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        else:
            cv_image = image
            
        # Store original image dimensions
        orig_h, orig_w = cv_image.shape[:2]
        
        # Calculate optimal input size (multiples of 32 for better performance)
        optimal_size = min(max(orig_w, orig_h), 640)  # Cap at 640 for performance
        optimal_size = ((optimal_size + 31) // 32) * 32  # Round to nearest 32
        
        # Maintain aspect ratio for input size
        if orig_w > orig_h:
            input_w = optimal_size
            input_h = int(orig_h * optimal_size / orig_w)
            input_h = ((input_h + 31) // 32) * 32
        else:
            input_h = optimal_size
            input_w = int(orig_w * optimal_size / orig_h)
            input_w = ((input_w + 31) // 32) * 32
        
        # Resize image to match the input size expected by detector
        resized_image = cv.resize(cv_image, (input_w, input_h))

        # Detect faces on resized image with CUDA fallback
        try:
            # Set the input size for the detector
            self.detector.setInputSize((input_w, input_h))
            _, faces = self.detector.detect(resized_image)
        except cv.error as e:
            # Check if this is a CUDA backend error
            if 'DNN_BACKEND_CUDA' in str(e) and self._using_cuda:
                # Try to reinitialize with CPU
                if self._reinit_with_cpu():
                    # Retry detection with CPU backend
                    self.detector.setInputSize((input_w, input_h))
                    _, faces = self.detector.detect(resized_image)
                else:
                    raise e
            else:
                raise e
        
        # Scale back the coordinates to original image size
        if faces is not None and len(faces) > 0:
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h
            
            for i in range(len(faces)):
                # Scale bounding box coordinates
                faces[i][0] *= scale_x  # x
                faces[i][1] *= scale_y  # y
                faces[i][2] *= scale_x  # width
                faces[i][3] *= scale_y  # height
                
                # Scale landmark coordinates (5 points * 2 coordinates each)
                for j in range(4, 14):  # landmarks are at indices 4-13
                    if j % 2 == 0:  # even indices are x coordinates
                        faces[i][j] *= scale_x
                    else:  # odd indices are y coordinates
                        faces[i][j] *= scale_y
        
        return faces if faces is not None else []
    
    def align(self, img, return_landmarks=False):
        """
        Detect and align face using YuNet (compatible with MTCNN interface)
        
        Args:
            img: PIL Image
            return_landmarks: whether to return landmarks
            
        Returns:
            aligned_face: PIL Image of aligned face or None if no face detected
            landmarks: facial landmarks if return_landmarks=True
        """
        faces = self.detect_faces(img)
        
        if len(faces) == 0:
            return None if not return_landmarks else (None, None)
            
        # Select the best face (highest confidence) instead of averaging
        best_face = max(faces, key=lambda x: x[-1])  # Select face with highest confidence score
        
        # Extract landmarks from the best face
        landmarks = []
        for i in range(5):
            x = best_face[4 + i * 2]
            y = best_face[4 + i * 2 + 1]
            landmarks.append([x, y])
        
        best_landmarks = np.array(landmarks, dtype=np.float32)
        
        # Apply enhanced landmark stabilization
        facial_pts = self._stabilize_landmarks(best_landmarks)
        
        # Align face using the same transformation as MTCNN
        try:
            # Use YuNet-optimized reference points
            from mtcnn_pytorch.src.align_trans import get_reference_facial_points
            ref_pts = get_reference_facial_points(
                output_size=self.crop_size,
                inner_padding_factor=0,
                outer_padding=(0, 0),
                default_square=(self.crop_size[0] == self.crop_size[1])
            )
            
            aligned_face = warp_and_crop_face(
                np.array(img), 
                facial_pts, 
                reference_pts=ref_pts,
                crop_size=self.crop_size,
                align_type='similarity'
            )
        except Exception as e:
            try:
                # Fallback: use affine transformation (more flexible)
                aligned_face = warp_and_crop_face(
                    np.array(img), 
                    facial_pts, 
                    reference_pts=ref_pts,
                    crop_size=self.crop_size,
                    align_type='affine'
                )
            except Exception as e2:
                # Final fallback: use basic crop around detected face
                face = faces[0]
                x, y, w, h = face[:4]
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Add some padding
                padding = int(min(w, h) * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img.size[0], x + w + padding)
                y2 = min(img.size[1], y + h + padding)
                
                cropped = img.crop((x1, y1, x2, y2))
                aligned_face = np.array(cropped.resize(self.crop_size))
        
        aligned_face_pil = Image.fromarray(aligned_face)
        
        if return_landmarks:
            return aligned_face_pil, facial_pts.tolist()
        else:
            return aligned_face_pil
    
    def _stabilize_landmarks(self, landmarks):
        """
        Apply enhanced landmark stabilization for better alignment quality
        
        Args:
            landmarks: numpy array of shape (5, 2)
            
        Returns:
            stabilized_landmarks: numpy array of shape (5, 2)
        """
        stabilized = landmarks.copy()
        
        # 1. Basic sanity check: ensure left eye is to the left of right eye
        if stabilized[0, 0] > stabilized[1, 0]:  # left_eye_x > right_eye_x
            stabilized[[0, 1]] = stabilized[[1, 0]]
        
        # 2. Validate landmark positions are within reasonable bounds
        # Calculate face center and size for reference
        face_center_x = np.mean(stabilized[:, 0])
        face_center_y = np.mean(stabilized[:, 1])
        
        # Calculate approximate face size from eye distance
        eye_distance = np.linalg.norm(stabilized[1] - stabilized[0])
        face_size = eye_distance * 3.0  # Approximate face size
        
        # 3. Validate each landmark is within reasonable distance from face center
        for i in range(5):
            dist_from_center = np.linalg.norm(stabilized[i] - [face_center_x, face_center_y])
            if dist_from_center > face_size:
                # If landmark is too far, move it closer to expected position
                direction = (stabilized[i] - [face_center_x, face_center_y]) / dist_from_center
                stabilized[i] = [face_center_x, face_center_y] + direction * (face_size * 0.8)
        
        # 4. Ensure nose is roughly centered between eyes horizontally
        expected_nose_x = (stabilized[0, 0] + stabilized[1, 0]) / 2
        nose_x_diff = abs(stabilized[2, 0] - expected_nose_x)
        if nose_x_diff > eye_distance * 0.3:  # If nose is too far off-center
            stabilized[2, 0] = expected_nose_x + np.sign(stabilized[2, 0] - expected_nose_x) * eye_distance * 0.3
        
        # 5. Ensure mouth landmarks are below nose
        nose_y = stabilized[2, 1]
        for i in [3, 4]:  # mouth corners
            if stabilized[i, 1] < nose_y:
                stabilized[i, 1] = nose_y + eye_distance * 0.3
        
        # 6. Ensure symmetric mouth position
        mouth_center_x = (stabilized[3, 0] + stabilized[4, 0]) / 2
        expected_mouth_x = expected_nose_x
        mouth_offset = mouth_center_x - expected_mouth_x
        if abs(mouth_offset) > eye_distance * 0.2:
            correction = np.sign(mouth_offset) * eye_distance * 0.2 - mouth_offset
            stabilized[3, 0] += correction / 2
            stabilized[4, 0] += correction / 2
        
        return stabilized
    
    def align_multi(self, img, limit=None):
        """
        Detect and align multiple faces (compatible with MTCNN interface)
        
        Args:
            img: PIL Image
            limit: maximum number of faces to process
            
        Returns:
            bboxes: list of bounding boxes
            faces: list of aligned face PIL Images
        """
        detected_faces = self.detect_faces(img)
        
        if len(detected_faces) == 0:
            return [], []
            
        if limit is not None:
            detected_faces = detected_faces[:limit]
        
        bboxes = []
        aligned_faces = []
        
        for face in detected_faces:
            # Extract bounding box
            x, y, w, h = face[:4]
            bbox = [x, y, x + w, y + h, face[-1]]  # [x1, y1, x2, y2, confidence]
            bboxes.append(bbox)
            
            # Extract landmarks
            landmarks = []
            for i in range(5):
                x_pt = face[4 + i * 2]
                y_pt = face[4 + i * 2 + 1]
                landmarks.append([x_pt, y_pt])
            
            # Align face
            facial_pts = np.array(landmarks, dtype=np.float32)
            try:
                aligned_face = warp_and_crop_face(
                    np.array(img), 
                    facial_pts, 
                    reference_pts=None,
                    crop_size=self.crop_size,
                    align_type='similarity'
                )
            except Exception as e:
                try:
                    # Fallback with explicit reference points
                    from mtcnn_pytorch.src.align_trans import get_reference_facial_points
                    ref_pts = get_reference_facial_points(
                        output_size=self.crop_size,
                        inner_padding_factor=0,
                        outer_padding=(0, 0),
                        default_square=True
                    )
                    aligned_face = warp_and_crop_face(
                        np.array(img), 
                        facial_pts, 
                        reference_pts=ref_pts,
                        crop_size=self.crop_size,
                        align_type='similarity'
                    )
                except Exception as e2:
                    # Skip this face if alignment fails
                    continue
            
            aligned_faces.append(Image.fromarray(aligned_face))
        
        return bboxes, aligned_faces


def get_aligned_face_yunet(image_path, model_path, rgb_pil_image=None):
    """
    Convenience function to get aligned face using YuNet (similar to MTCNN version)
    
    Args:
        image_path: path to image file
        model_path: path to YuNet ONNX model
        rgb_pil_image: PIL image (if provided, image_path is ignored)
        
    Returns:
        aligned_face: PIL Image of aligned face or None if failed
    """
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    
    try:
        detector = YuNetDetector(model_path, crop_size=(112, 112))
        aligned_face = detector.align(img)
        return aligned_face
    except Exception as e:
        print('Face detection failed due to error:')
        print(e)
        import traceback
        traceback.print_exc()
        return None