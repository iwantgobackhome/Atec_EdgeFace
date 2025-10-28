"""
YuNet Face Detector with DeepX NPU Support
DeepX NPU를 사용한 YuNet 얼굴 검출기
"""

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

try:
    from dx_engine import InferenceEngine
    DXENGINE_AVAILABLE = True
except ImportError:
    print("dx_engine을 가져올 수 없습니다. DeepX NPU SDK가 설치되어 있는지 확인하세요.")
    DXENGINE_AVAILABLE = False
    InferenceEngine = None

import sys
import os

# MTCNN의 align_trans 모듈을 재사용
sys.path.insert(0, os.path.dirname(__file__))
try:
    from mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points
except ImportError:
    print("MTCNN align_trans 모듈을 가져올 수 없습니다.")
    warp_and_crop_face = None
    get_reference_facial_points = None


class YuNetNPUDetector:
    """YuNet face detector using DeepX NPU"""

    def __init__(self, model_path, device='npu', crop_size=(112, 112)):
        """
        Initialize YuNet detector with DeepX NPU

        Args:
            model_path: Path to YuNet DXNN model (e.g., face_detection_yunet_2023mar.dxnn)
            device: 'npu' (only npu is supported)
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
        if not DXENGINE_AVAILABLE:
            raise ImportError("dx_engine을 사용할 수 없습니다. DeepX NPU SDK를 설치하세요.")

        self.device = device
        self.crop_size = crop_size
        self.model_path = model_path

        # Input size for YuNet (from calibration config)
        self.input_size = (320, 320)

        # Detection thresholds
        self.score_threshold = 0.8
        self.nms_threshold = 0.3

        # Initialize DeepX NPU Inference Engine
        try:
            print(f"YuNet NPU: Loading model from {model_path}...")
            self.inference_engine = InferenceEngine(model_path)
            print(f"YuNet NPU: Model loaded successfully")
            print(f"YuNet NPU: Input size: {self.inference_engine.input_size()}")
            print(f"YuNet NPU: Output dtype: {self.inference_engine.output_dtype()}")
        except Exception as e:
            raise RuntimeError(f"YuNet NPU detector 초기화 실패: {e}")

    def _preprocess_image(self, cv_image):
        """
        Preprocess image for YuNet NPU inference

        Args:
            cv_image: OpenCV BGR image

        Returns:
            preprocessed: Preprocessed image tensor ready for NPU
            scale_x, scale_y: Scale factors for coordinate conversion
        """
        orig_h, orig_w = cv_image.shape[:2]

        # Resize to input size (320x320)
        resized = cv.resize(cv_image, self.input_size)

        # Convert BGR to RGB (as per calibration config)
        # Note: YuNet calibration config expects RGB input
        rgb_image = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

        # Transpose to CHW format (as per calibration config)
        # Shape: (H, W, C) -> (C, H, W)
        chw_image = np.transpose(rgb_image, (2, 0, 1))

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        # Note: dx_engine expects this format
        input_tensor = np.expand_dims(chw_image, axis=0)

        # Calculate scale factors for coordinate conversion
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]

        return input_tensor.astype(np.uint8), scale_x, scale_y

    def _decode_outputs(self, outputs, scale_x, scale_y):
        """
        Decode YuNet NPU outputs to face detections

        Args:
            outputs: List of output tensors from NPU
            scale_x, scale_y: Scale factors for coordinate conversion

        Returns:
            faces: List of detected faces [x, y, w, h, x1, y1, ..., x5, y5, confidence]
        """
        faces = []

        # YuNet typically outputs a single tensor with shape (N, 15) or (1, N, 15)
        # where N is the number of detections
        # Each detection: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, conf]

        if len(outputs) == 0:
            return faces

        # Get the main output tensor
        output = outputs[0]

        # Handle different output shapes
        if len(output.shape) == 3:
            # Shape: (1, N, 15) -> squeeze to (N, 15)
            output = output.squeeze(0)
        elif len(output.shape) == 1:
            # Shape: (15,) -> single detection, reshape to (1, 15)
            if len(output) >= 15:
                output = output.reshape(1, -1)
            else:
                return faces

        # output should now be (N, 15) where N is number of detections
        if len(output.shape) != 2:
            print(f"[WARNING] Unexpected YuNet output shape: {output.shape}")
            return faces

        if output.shape[1] < 15:
            print(f"[WARNING] YuNet output has fewer than 15 values per detection: {output.shape}")
            return faces

        # Filter detections by confidence threshold
        for detection in output:
            if len(detection) < 15:
                continue

            confidence = float(detection[14])

            # Apply confidence threshold
            if confidence < self.score_threshold:
                continue

            # Extract coordinates (already in input image coordinates: 320x320)
            x = float(detection[0])
            y = float(detection[1])
            w = float(detection[2])
            h = float(detection[3])

            # Extract landmarks (5 points)
            landmarks = []
            for i in range(5):
                lm_x = float(detection[4 + i * 2])
                lm_y = float(detection[5 + i * 2])
                landmarks.extend([lm_x, lm_y])

            # Scale coordinates back to original image size
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y

            landmarks_scaled = []
            for i in range(0, 10, 2):
                landmarks_scaled.append(landmarks[i] * scale_x)      # x
                landmarks_scaled.append(landmarks[i+1] * scale_y)    # y

            # Build face detection result
            # Format: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]
            face = [x_scaled, y_scaled, w_scaled, h_scaled]
            face.extend(landmarks_scaled)
            face.append(confidence)

            faces.append(face)

        # Apply NMS (Non-Maximum Suppression) if multiple detections
        if len(faces) > 1:
            faces = self._apply_nms(faces, self.nms_threshold)

        return faces

    def _apply_nms(self, faces, nms_threshold):
        """
        Apply Non-Maximum Suppression to remove overlapping detections

        Args:
            faces: List of face detections
            nms_threshold: IoU threshold for NMS

        Returns:
            Filtered list of faces
        """
        if len(faces) == 0:
            return faces

        # Convert to numpy array for easier processing
        faces_array = np.array(faces)

        # Extract bounding boxes and scores
        x = faces_array[:, 0]
        y = faces_array[:, 1]
        w = faces_array[:, 2]
        h = faces_array[:, 3]
        scores = faces_array[:, 14]

        # Calculate areas
        x2 = x + w
        y2 = y + h
        areas = w * h

        # Sort by confidence (descending)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]

        return [faces[i] for i in keep]

    def detect_faces(self, image):
        """
        Detect faces and landmarks using YuNet NPU

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

        # Preprocess image
        input_tensor, scale_x, scale_y = self._preprocess_image(cv_image)

        # Run inference on NPU
        try:
            outputs = self.inference_engine.Run(input_tensor)
        except Exception as e:
            print(f"YuNet NPU inference error: {e}")
            return []

        # Decode outputs to face detections
        faces = self._decode_outputs(outputs, scale_x, scale_y)

        return faces

    def align(self, img, return_landmarks=False):
        """
        Detect and align face using YuNet NPU (compatible with MTCNN interface)

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

        # Select the best face (highest confidence)
        best_face = max(faces, key=lambda x: x[-1])

        # Extract landmarks from the best face
        landmarks = []
        for i in range(5):
            x = best_face[4 + i * 2]
            y = best_face[4 + i * 2 + 1]
            landmarks.append([x, y])

        best_landmarks = np.array(landmarks, dtype=np.float32)

        # Apply landmark stabilization
        facial_pts = self._stabilize_landmarks(best_landmarks)

        # Align face using the same transformation as MTCNN
        try:
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
                # Fallback: use affine transformation
                aligned_face = warp_and_crop_face(
                    np.array(img),
                    facial_pts,
                    reference_pts=ref_pts,
                    crop_size=self.crop_size,
                    align_type='affine'
                )
            except Exception as e2:
                # Final fallback: use basic crop
                face = best_face
                x, y, w, h = face[:4]
                x, y, w, h = int(x), int(y), int(w), int(h)

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
        Apply landmark stabilization for better alignment quality
        (Same as yunet.py implementation)
        """
        stabilized = landmarks.copy()

        # 1. Basic sanity check: ensure left eye is to the left of right eye
        if stabilized[0, 0] > stabilized[1, 0]:
            stabilized[[0, 1]] = stabilized[[1, 0]]

        # 2. Validate landmark positions
        face_center_x = np.mean(stabilized[:, 0])
        face_center_y = np.mean(stabilized[:, 1])

        eye_distance = np.linalg.norm(stabilized[1] - stabilized[0])
        face_size = eye_distance * 3.0

        # 3. Validate each landmark distance from center
        for i in range(5):
            dist_from_center = np.linalg.norm(stabilized[i] - [face_center_x, face_center_y])
            if dist_from_center > face_size:
                direction = (stabilized[i] - [face_center_x, face_center_y]) / dist_from_center
                stabilized[i] = [face_center_x, face_center_y] + direction * (face_size * 0.8)

        # 4. Ensure nose is centered
        expected_nose_x = (stabilized[0, 0] + stabilized[1, 0]) / 2
        nose_x_diff = abs(stabilized[2, 0] - expected_nose_x)
        if nose_x_diff > eye_distance * 0.3:
            stabilized[2, 0] = expected_nose_x + np.sign(stabilized[2, 0] - expected_nose_x) * eye_distance * 0.3

        # 5. Ensure mouth landmarks are below nose
        nose_y = stabilized[2, 1]
        for i in [3, 4]:
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

    def align_face(self, image, landmarks):
        """
        Align face using provided landmarks

        Args:
            image: PIL Image
            landmarks: Facial landmarks array [x1,y1,x2,y2,x3,y3,x4,y4,x5,y5]

        Returns:
            Aligned face image or None if alignment fails
        """
        # Convert flat landmarks to (5, 2) array
        if len(landmarks) == 10:
            facial_pts = np.array([[landmarks[i], landmarks[i+1]] for i in range(0, 10, 2)], dtype=np.float32)
        else:
            facial_pts = np.array(landmarks, dtype=np.float32)

        # Apply stabilization
        facial_pts = self._stabilize_landmarks(facial_pts)

        try:
            ref_pts = get_reference_facial_points(
                output_size=self.crop_size,
                inner_padding_factor=0,
                outer_padding=(0, 0),
                default_square=(self.crop_size[0] == self.crop_size[1])
            )

            aligned_face = warp_and_crop_face(
                np.array(image),
                facial_pts,
                reference_pts=ref_pts,
                crop_size=self.crop_size,
                align_type='similarity'
            )

            return Image.fromarray(aligned_face)
        except Exception as e:
            print(f"Face alignment failed: {e}")
            return None

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
            bbox = [x, y, x + w, y + h, face[-1]]
            bboxes.append(bbox)

            # Extract landmarks
            landmarks = []
            for i in range(5):
                x_pt = face[4 + i * 2]
                y_pt = face[4 + i * 2 + 1]
                landmarks.append([x_pt, y_pt])

            # Align face
            facial_pts = np.array(landmarks, dtype=np.float32)
            facial_pts = self._stabilize_landmarks(facial_pts)

            try:
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
                aligned_faces.append(Image.fromarray(aligned_face))
            except Exception as e2:
                # Skip this face if alignment fails
                continue

        return bboxes, aligned_faces
