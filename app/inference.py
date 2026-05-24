import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class Config:
    """Inference configuration."""
    INPUT_SIZE = 300
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    CLASSES = [
        "Actinic keratoses",
        "Basal cell carcinoma",
        "Benign keratosis-like lesions",
        "Dermatofibroma",
        "Melanocytic nevi",
        "Melanoma",
        "Vascular lesions",
        "Squamous cell carcinoma",
        "Unknown/Other"
    ]
    NUM_CLASSES = 9


class SkinInference:
    """ONNX-based skin lesion inference engine."""
    
    def __init__(self, model_path: str, providers: List[str] = None):
        """
        Initialize inference session.
        
        Args:
            model_path: Path to ONNX model file
            providers: List of execution providers (default: CPU)
        
        Raises:
            FileNotFoundError: If model not found
            RuntimeError: If ONNX session creation fails
        """
        if not model_path.endswith('.onnx'):
            raise ValueError(f"Expected .onnx file, got: {model_path}")
        
        try:
            if providers is None:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.classes = Config.CLASSES
            
            logger.info(f"Loaded ONNX model: {model_path}")
            logger.info(f"Input name: {self.input_name}")
            logger.info(f"Output name: {self.output_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model '{model_path}': {e}")
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image bytes to model input.
        
        Args:
            image_bytes: Raw image bytes (JPEG/PNG)
        
        Returns:
            Preprocessed image as (1, 3, 300, 300) float32 array
        
        Raises:
            ValueError: If image cannot be decoded or is invalid
        """
        try:
            if len(image_bytes) == 0:
                raise ValueError("Image is empty")
            
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image. Ensure it's a valid JPEG/PNG.")
            
            if img.size == 0:
                raise ValueError("Image is empty")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, (Config.INPUT_SIZE, Config.INPUT_SIZE))
            
            # Normalize: [0, 255] to [0, 1] --> standardize
            img = img.astype(np.float32) / 255.0
            img = (img - Config.IMAGENET_MEAN) / Config.IMAGENET_STD
            
            # HWC to CHW and add batch dimension
            img = np.transpose(img, (2, 0, 1))  # Explicit for clarity
            img = np.expand_dims(img, axis=0)
            
            return img.astype(np.float32)
        
        except ValueError as e:
            raise ValueError(f"Preprocessing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during preprocessing: {e}")
    
    def predict(self, image_bytes: bytes) -> List[Tuple[str, float]]:
        """
        Run inference on image.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            List of (class_name, confidence) tuples sorted by confidence (descending)
        
        Raises:
            ValueError: If preprocessing fails
            RuntimeError: If inference fails
        """
        try:
            # Preprocess
            input_data = self.preprocess(image_bytes)
            
            # Run inference (returns logits, not probabilities)
            logits = self.session.run(None, {self.input_name: input_data})[0]
            
            # Convert logits to probabilities using softmax (numerical stability)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Get probabilities for batch[0]
            probs = probs[0]
            
            # Return sorted results
            results = [
                (class_name, float(prob))
                for class_name, prob in zip(self.classes, probs)
            ]
            return sorted(results, key=lambda x: x[1], reverse=True)
        
        except ValueError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
    
    def predict_with_metadata(self, image_bytes: bytes) -> dict:
        """
        Run inference and return structured output with metadata.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Dictionary with prediction, confidence, and uncertainty flag
        """
        predictions = self.predict(image_bytes)
        top_pred, top_conf = predictions[0]
        
        # Flag uncertain predictions
        uncertainty_threshold = 0.50
        is_uncertain = top_conf < uncertainty_threshold
        
        return {
            "prediction": top_pred,
            "confidence": top_conf,
            "is_uncertain": is_uncertain,
            "all_predictions": predictions,
            "model_version": "efficientnet_b3_onnx_v1",
        }