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

        # Resize to input size (640x640 as per test_npu_inference.py)
        # Updated from 320x320 to match actual model input
        self.input_size = (640, 640)
        resized = cv.resize(cv_image, self.input_size)

        # Convert BGR to RGB (as per test_npu_inference.py)
        rgb_image = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

        # YuNet expects HWC format, not CHW! (as per test_npu_inference.py)
        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        input_tensor = np.expand_dims(rgb_image, axis=0)

        # Calculate scale factors for coordinate conversion
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]

        return input_tensor.astype(np.uint8), scale_x, scale_y

    def _decode_outputs(self, outputs, scale_x, scale_y):
        """
        Decode YuNet NPU outputs to face detections

        YuNet outputs 13 tensors from 3 scales (FPN):
        - Scale 1 (80x80): outputs 1, 3, 10 (landmarks, cls, loc)
        - Scale 2 (40x40): outputs 6, 8, 12 (landmarks, cls, loc)
        - Scale 3 (20x20): outputs 7, 9, 11 (landmarks, cls, loc)

        Args:
            outputs: List of output tensors from NPU (each may be wrapped in a list)
            scale_x, scale_y: Scale factors for coordinate conversion

        Returns:
            faces: List of detected faces [x, y, w, h, x1, y1, ..., x5, y5, confidence]
        """
        faces = []

        if len(outputs) == 0:
            return faces

        # Unwrap list wrappers from all outputs
        unwrapped_outputs = []
        for out in outputs:
            if isinstance(out, list) and len(out) > 0:
                unwrapped_outputs.append(out[0] if isinstance(out[0], np.ndarray) else out)
            elif isinstance(out, np.ndarray):
                unwrapped_outputs.append(out)
            else:
                unwrapped_outputs.append(out)

        print(f"[DEBUG] YuNet NPU outputs count: {len(unwrapped_outputs)}")

        # Print ALL output shapes for debugging
        for i, out in enumerate(unwrapped_outputs):
            if isinstance(out, np.ndarray):
                print(f"[DEBUG]   Output {i}: shape={out.shape}, dtype={out.dtype}, min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
            else:
                print(f"[DEBUG]   Output {i}: type={type(out)}")

        if len(unwrapped_outputs) != 13:
            print(f"[WARNING] Expected 13 outputs, got {len(unwrapped_outputs)}")
            return faces

        # Extract outputs for each scale
        # Based on the debug output pattern:
        # Output 0: shape=(1, 10, 80, 80) - scale 1 landmarks
        # Output 1: shape=(1, 6400, 10) - scale 1 landmarks (flattened)
        # Output 3: shape=(1, 6400, 1) - scale 1 cls scores
        # Output 10: shape=(1, 6400, 4) - scale 1 bboxes
        #
        # Output 6: shape=(1, 1600, 10) - scale 2 landmarks
        # Output 8: shape=(1, 1600, 1) - scale 2 cls scores
        # Output 12: shape=(1, 1600, 4) - scale 2 bboxes
        #
        # Output 7: shape=(1, 400, 10) - scale 3 landmarks
        # Output 9: shape=(1, 400, 1) - scale 3 cls scores
        # Output 11: shape=(1, 400, 4) - scale 3 bboxes

        try:
            # Process all 3 scales
            all_detections = []

            # Scale 1: 80x80 feature map (6400 anchors)
            cls_1 = unwrapped_outputs[3].squeeze()  # (6400, 1) -> (6400,)
            loc_1 = unwrapped_outputs[10].squeeze()  # (6400, 4)
            obj_1 = unwrapped_outputs[1].squeeze()   # (6400, 10)
            scale_detections = self._process_scale(cls_1, loc_1, obj_1, stride=8, input_size=self.input_size[0])
            all_detections.extend(scale_detections)
            print(f"[DEBUG] Scale 1 (80x80): {len(scale_detections)} raw detections")

            # Scale 2: 40x40 feature map (1600 anchors)
            cls_2 = unwrapped_outputs[8].squeeze()   # (1600, 1) -> (1600,)
            loc_2 = unwrapped_outputs[12].squeeze()  # (1600, 4)
            obj_2 = unwrapped_outputs[6].squeeze()   # (1600, 10)
            scale_detections = self._process_scale(cls_2, loc_2, obj_2, stride=16, input_size=self.input_size[0])
            all_detections.extend(scale_detections)
            print(f"[DEBUG] Scale 2 (40x40): {len(scale_detections)} raw detections")

            # Scale 3: 20x20 feature map (400 anchors)
            cls_3 = unwrapped_outputs[9].squeeze()   # (400, 1) -> (400,)
            loc_3 = unwrapped_outputs[11].squeeze()  # (400, 4)
            obj_3 = unwrapped_outputs[7].squeeze()   # (400, 10)
            scale_detections = self._process_scale(cls_3, loc_3, obj_3, stride=32, input_size=self.input_size[0])
            all_detections.extend(scale_detections)
            print(f"[DEBUG] Scale 3 (20x20): {len(scale_detections)} raw detections")

            print(f"[DEBUG] Total detections before NMS: {len(all_detections)}")

            # Apply NMS across all scales
            if len(all_detections) > 0:
                # Scale coordinates back to original image size
                for detection in all_detections:
                    detection[0] *= scale_x  # x
                    detection[1] *= scale_y  # y
                    detection[2] *= scale_x  # w
                    detection[3] *= scale_y  # h
                    for i in range(4, 14, 2):
                        detection[i] *= scale_x      # landmark x
                        detection[i+1] *= scale_y    # landmark y

                faces = self._apply_nms(all_detections, self.nms_threshold)
                print(f"[DEBUG] Detections after NMS: {len(faces)}")

        except Exception as e:
            print(f"[ERROR] Failed to decode YuNet outputs: {e}")
            import traceback
            traceback.print_exc()
            return faces

        return faces

    def _process_scale(self, cls_scores, bboxes, landmarks, stride, input_size):
        """
        Process detections from a single FPN scale

        Args:
            cls_scores: Classification scores (N,) or (N, 1)
            bboxes: Bounding boxes (N, 4) - format depends on YuNet encoding
            landmarks: Facial landmarks (N, 10)
            stride: Feature map stride (8, 16, or 32)
            input_size: Model input size (e.g., 640)

        Returns:
            List of detections [x, y, w, h, x1, y1, ..., x5, y5, confidence]
        """
        detections = []

        # Ensure correct shapes
        if cls_scores.ndim > 1:
            cls_scores = cls_scores.flatten()
        if bboxes.ndim == 3:
            bboxes = bboxes.squeeze(0)
        if landmarks.ndim == 3:
            landmarks = landmarks.squeeze(0)

        # Calculate feature map size
        feat_size = input_size // stride

        # Filter by confidence threshold
        valid_mask = cls_scores >= self.score_threshold
        valid_indices = np.where(valid_mask)[0]

        print(f"[DEBUG]   Stride {stride}: {len(valid_indices)}/{len(cls_scores)} above threshold {self.score_threshold}")
        print(f"[DEBUG]     cls_scores - min: {cls_scores.min():.4f}, max: {cls_scores.max():.4f}, mean: {cls_scores.mean():.4f}")
        if len(valid_indices) > 0:
            print(f"[DEBUG]     First 5 valid scores: {cls_scores[valid_indices[:5]]}")
            print(f"[DEBUG]     First 5 valid bboxes: {bboxes[valid_indices[0]]}")
            print(f"[DEBUG]     First 5 valid landmarks: {landmarks[valid_indices[0]]}")

        for idx in valid_indices:
            confidence = float(cls_scores[idx])

            # Decode bounding box from anchor
            # YuNet uses center-based encoding
            bbox = bboxes[idx]

            # Calculate anchor position
            anchor_y = (idx // feat_size) + 0.5
            anchor_x = (idx % feat_size) + 0.5

            # Decode bbox (assuming YuNet's encoding scheme)
            # Format: [cx_offset, cy_offset, w, h] or [x, y, w, h]
            cx = (anchor_x + bbox[0]) * stride
            cy = (anchor_y + bbox[1]) * stride
            w = bbox[2] * stride
            h = bbox[3] * stride

            # Convert to top-left corner format
            x = cx - w / 2
            y = cy - h / 2

            # Decode landmarks (10 values: 5 points * 2 coordinates)
            lms = landmarks[idx]
            decoded_lms = []
            for i in range(5):
                lm_x = (anchor_x + lms[i*2]) * stride
                lm_y = (anchor_y + lms[i*2 + 1]) * stride
                decoded_lms.extend([lm_x, lm_y])

            # Build detection: [x, y, w, h, x1, y1, ..., x5, y5, confidence]
            detection = [x, y, w, h] + decoded_lms + [confidence]
            detections.append(detection)

        return detections

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

        # Make input contiguous to avoid NPU warning
        input_tensor = np.ascontiguousarray(input_tensor)

        # Run inference on NPU using run() + get_all_task_outputs() pattern
        try:
            self.inference_engine.run(input_tensor)
            outputs = self.inference_engine.get_all_task_outputs()
        except Exception as e:
            print(f"YuNet NPU inference error: {e}")
            import traceback
            traceback.print_exc()
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
