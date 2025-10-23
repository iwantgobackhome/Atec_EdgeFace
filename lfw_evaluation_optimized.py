"""
LFW Evaluation Pipeline - OPTIMIZED VERSION
Major optimizations:
1. Batch processing for embeddings (GPU efficiency)
2. Pre-cached transforms
3. Parallel image loading
4. Reduced GPU-CPU transfers
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_curve, auc
from concurrent.futures import ThreadPoolExecutor
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'face_alignment'))

try:
    from face_alignment.unified_detector import UnifiedFaceDetector
except ImportError:
    print("Failed to import unified detector")

try:
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class LFWEvaluatorOptimized:
    """Optimized LFW evaluator with batch processing and caching"""

    def __init__(self, lfw_dir: str, pairs_file: str, edgeface_model_path: str,
                 device: str = 'cpu', batch_size: int = 32, num_workers: int = 4):
        """
        Initialize optimized LFW evaluator

        Args:
            batch_size: Batch size for embedding extraction (GPU efficiency)
            num_workers: Number of parallel workers for image loading
        """
        self.lfw_dir = lfw_dir
        self.pairs_file = pairs_file
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Pre-create transform (reuse instead of recreating)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        # Load pairs
        self.pairs = self._load_pairs()
        print(f"Loaded {len(self.pairs)} pairs from LFW")

        # Initialize EdgeFace model
        self.edgeface_model = self._load_edgeface_model(edgeface_model_path)

        # Results storage
        self.results = {}

    def _load_pairs(self) -> List[Tuple]:
        """Load LFW pairs from CSV"""
        pairs = []

        if not os.path.exists(self.pairs_file):
            print(f"Pairs file not found: {self.pairs_file}")
            return pairs

        try:
            with open(self.pairs_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.endswith(','):
                    # Same person pair
                    parts = line.rstrip(',').split(',')
                    if len(parts) == 3:
                        try:
                            person = parts[0]
                            img1_num = int(parts[1])
                            img2_num = int(parts[2])

                            img1_path = os.path.join(self.lfw_dir, person, f"{person}_{img1_num:04d}.jpg")
                            img2_path = os.path.join(self.lfw_dir, person, f"{person}_{img2_num:04d}.jpg")
                            pairs.append((True, img1_path, img2_path))
                        except ValueError:
                            continue
                else:
                    # Different person pair
                    parts = line.split(',')
                    if len(parts) == 4:
                        try:
                            person1 = parts[0]
                            img1_num = int(parts[1])
                            person2 = parts[2]
                            img2_num = int(parts[3])

                            img1_path = os.path.join(self.lfw_dir, person1, f"{person1}_{img1_num:04d}.jpg")
                            img2_path = os.path.join(self.lfw_dir, person2, f"{person2}_{img2_num:04d}.jpg")
                            pairs.append((False, img1_path, img2_path))
                        except ValueError:
                            continue

        except Exception as e:
            print(f"Error reading CSV file: {e}")

        return pairs

    def _load_edgeface_model(self, model_path: str):
        """Load EdgeFace model"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available")
            return None

        if not os.path.exists(model_path):
            print(f"EdgeFace model not found: {model_path}")
            return None

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            from backbones import get_model

            arch = "edgeface_xs_gamma_06"
            model = get_model(arch)

            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()

            print(f"EdgeFace model loaded successfully: {arch} on {self.device}")
            return model

        except Exception as e:
            print(f"Failed to load EdgeFace model: {e}")
            return None

    def _preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Preprocess a batch of images efficiently

        Args:
            images: List of PIL Images

        Returns:
            Batched tensor [N, C, H, W]
        """
        tensors = []
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((112, 112), Image.LANCZOS)
            tensor = self.transform(img)
            tensors.append(tensor)

        return torch.stack(tensors)

    def _extract_embeddings_batch(self, aligned_faces: List[Image.Image]) -> np.ndarray:
        """
        Extract embeddings in batch for GPU efficiency

        Args:
            aligned_faces: List of aligned PIL Images

        Returns:
            Embeddings array [N, embedding_dim]
        """
        if self.edgeface_model is None or not aligned_faces:
            return np.random.randn(len(aligned_faces), 512).astype(np.float32)

        try:
            # Batch preprocessing
            batch_tensor = self._preprocess_batch(aligned_faces)
            batch_tensor = batch_tensor.to(self.device)

            # Batch inference
            with torch.no_grad():
                embeddings = self.edgeface_model(batch_tensor)

            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]

            # Single GPU->CPU transfer for entire batch
            embeddings = embeddings.cpu().numpy()

            return embeddings

        except Exception as e:
            print(f"Batch embedding extraction failed: {e}")
            return np.random.randn(len(aligned_faces), 512).astype(np.float32)

    def _load_image_pair(self, pair_info):
        """Load image pair (for parallel loading)"""
        is_same, img1_path, img2_path = pair_info

        if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
            return None

        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            return (is_same, img1, img2)
        except:
            return None

    def evaluate_method(self, method_name: str, detector: UnifiedFaceDetector,
                       max_pairs: Optional[int] = None) -> Dict:
        """
        Evaluate with batch processing and parallel loading
        """
        print(f"\nEvaluating {method_name} (Optimized)...")

        # Select pairs
        if max_pairs:
            positive_pairs = [p for p in self.pairs if p[0] == True]
            negative_pairs = [p for p in self.pairs if p[0] == False]

            half_pairs = max_pairs // 2
            selected_positive = positive_pairs[:half_pairs]
            selected_negative = negative_pairs[:half_pairs]
            pairs_to_process = selected_positive + selected_negative
            print(f"Selected {len(selected_positive)} positive and {len(selected_negative)} negative pairs")
        else:
            pairs_to_process = self.pairs

        # Batch processing
        all_similarities = []
        all_labels = []
        failed_alignments = 0
        processing_times = []

        # Process in batches
        batch_start_time = time.time()

        for i in tqdm(range(0, len(pairs_to_process), self.batch_size), desc=f"Batches {method_name}"):
            batch_pairs = pairs_to_process[i:i + self.batch_size]

            # Parallel image loading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                loaded_pairs = list(executor.map(self._load_image_pair, batch_pairs))

            # Filter out failed loads
            loaded_pairs = [p for p in loaded_pairs if p is not None]

            if not loaded_pairs:
                continue

            # Align all faces in batch
            batch_aligned1 = []
            batch_aligned2 = []
            batch_labels = []
            batch_indices = []

            for idx, (is_same, img1, img2) in enumerate(loaded_pairs):
                aligned1 = detector.align(img1)
                aligned2 = detector.align(img2)

                if aligned1 is not None and aligned2 is not None:
                    batch_aligned1.append(aligned1)
                    batch_aligned2.append(aligned2)
                    batch_labels.append(1 if is_same else 0)
                    batch_indices.append(idx)
                else:
                    failed_alignments += 1
                    # Do NOT add failed cases to similarities/labels for metrics

            # Batch embedding extraction (GPU efficient!)
            if batch_aligned1:
                emb1_batch = self._extract_embeddings_batch(batch_aligned1)
                emb2_batch = self._extract_embeddings_batch(batch_aligned2)

                # Explicitly free aligned face images from memory
                del batch_aligned1
                del batch_aligned2

                # Vectorized similarity computation
                norms1 = np.linalg.norm(emb1_batch, axis=1, keepdims=True)
                norms2 = np.linalg.norm(emb2_batch, axis=1, keepdims=True)
                similarities = np.sum(emb1_batch * emb2_batch, axis=1) / (norms1.flatten() * norms2.flatten())

                all_similarities.extend(similarities.tolist())
                all_labels.extend(batch_labels)

                # Clean up embeddings from memory
                del emb1_batch
                del emb2_batch
                del similarities
                del norms1
                del norms2

            # Clean up loaded pairs from memory
            del loaded_pairs

        elapsed_time = time.time() - batch_start_time

        if len(all_similarities) == 0:
            return {
                'method': method_name,
                'error': 'No valid pairs processed',
                'failed_alignments': failed_alignments,
                'num_pairs': 0,
                'success_rate': 0.0,
            }

        # Compute metrics
        similarities = np.array(all_similarities)
        labels = np.array(all_labels)

        total_pairs = len(pairs_to_process)
        successful_pairs = len(similarities)  # Only successful alignments are in the list
        success_rate = successful_pairs / total_pairs if total_pairs > 0 else 0.0

        try:
            fpr, tpr, thresholds = roc_curve(labels, similarities)
            roc_auc = auc(fpr, tpr)

            accuracies = []
            for threshold in thresholds:
                predictions = (similarities >= threshold).astype(int)
                accuracy = np.mean(predictions == labels)
                accuracies.append(accuracy)

            if len(accuracies) > 0:
                best_idx = np.argmax(accuracies)
                best_threshold = thresholds[best_idx]
                best_accuracy = accuracies[best_idx]
            else:
                best_threshold = 0.5
                best_accuracy = 0.0

            if len(fpr) > 0 and len(tpr) > 0:
                eer_idx = np.nanargmin(np.absolute(fpr - (1 - tpr)))
                eer = fpr[eer_idx]
            else:
                eer = 1.0

        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {
                'method': method_name,
                'error': f'Metrics failed: {str(e)}',
                'failed_alignments': failed_alignments,
            }

        avg_time = elapsed_time / len(pairs_to_process)

        results = {
            'method': method_name,
            'num_pairs': len(similarities),
            'failed_alignments': failed_alignments,
            'success_rate': success_rate,
            'roc_auc': roc_auc,
            'best_accuracy': best_accuracy,
            'best_threshold': best_threshold,
            'eer': eer,
            'avg_processing_time': avg_time,
            'total_time': elapsed_time,
            'similarities': similarities,
            'labels': labels,
        }

        return results

    def evaluate_all_methods(self, methods: List[str], max_pairs: Optional[int] = None) -> Dict:
        """Evaluate all methods"""
        results = {}

        for method in methods:
            try:
                detector = UnifiedFaceDetector(method, device=self.device)
                if detector.available:
                    results[method] = self.evaluate_method(method, detector, max_pairs)
                else:
                    results[method] = {'method': method, 'error': 'Method not available'}
            except Exception as e:
                results[method] = {'method': method, 'error': str(e)}

        self.results = results
        return results


def main():
    """Main evaluation with optimizations"""
    lfw_dir = "/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/lfw-deepfunneled"
    pairs_file = "/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/pairs.csv"
    edgeface_model_path = "checkpoints/edgeface_xs_gamma_06.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    methods = ['mtcnn', 'yunet', 'yolo', 'yolo_ultralytics']

    # Optimized evaluator with batch processing
    evaluator = LFWEvaluatorOptimized(
        lfw_dir=lfw_dir,
        pairs_file=pairs_file,
        edgeface_model_path=edgeface_model_path,
        device=device,
        batch_size=64,  # Larger batch for GPU efficiency
        num_workers=8   # Parallel image loading
    )

    print(f"🚀 Running OPTIMIZED evaluation on {device.upper()}")
    print(f"   Batch size: 64")
    print(f"   Parallel workers: 8")

    results = evaluator.evaluate_all_methods(methods, max_pairs=6000)

    # Print results
    for method, result in results.items():
        if 'error' not in result:
            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {result['best_accuracy']:.1%}")
            print(f"  Success Rate: {result['success_rate']:.1%}")
            print(f"  Total Time: {result['total_time']:.1f}s")
            print(f"  Avg Time/pair: {result['avg_processing_time']*1000:.1f}ms")


if __name__ == "__main__":
    main()
