"""
configuration module for the skin lesion classifier.

Usage:
    from config import Config, APIConfig, InferenceConfig
    
    # Access any parameter
    print(Config.INPUT_SIZE)  # 300
    print(APIConfig.MAX_FILE_SIZE_MB)  # 10
    print(InferenceConfig.UNCERTAINTY_THRESHOLD)  # 0.50
"""

from dataclasses import dataclass
from typing import List, Tuple
import os
from pathlib import Path


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model architecture and training parameters."""
    
    # Architecture
    MODEL_NAME: str = "EfficientNet-B3"
    NUM_CLASSES: int = 9
    PRETRAINED: bool = True
    
    # Input/Output
    INPUT_SIZE: int = 300
    INPUT_CHANNELS: int = 3
    
    # Normalization (ImageNet stats)
    IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Class names (must match training order)
    CLASSES: List[str] = None
    
    def __post_init__(self):
        if self.CLASSES is None:
            self.CLASSES = [
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


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    """Inference engine parameters."""
    
    # Model path
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "models/skin_classifier_v1.onnx"
    )
    
    # ONNX Runtime settings
    PROVIDERS: List[str] = None
    
    # Thresholds
    UNCERTAINTY_THRESHOLD: float = 0.50  # Flag predictions with lower confidence
    MELANOMA_CLASS: str = "Melanoma"
    
    # Batch processing
    MAX_BATCH_SIZE: int = 50
    
    def __post_init__(self):
        if self.PROVIDERS is None:
            self.PROVIDERS = ['CPUExecutionProvider']


# ============================================================================
# API CONFIGURATION
# ============================================================================

@dataclass
class APIConfig:
    """FastAPI server configuration."""
    
    # Server
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    RELOAD: bool = False
    LOG_LEVEL: str = "info"
    
    # File upload constraints
    MAX_FILE_SIZE_MB: int = 10
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024
    ALLOWED_CONTENT_TYPES: set = None
    
    # Rate limiting
    REQUESTS_PER_MINUTE: int = 60
    
    def __post_init__(self):
        if self.ALLOWED_CONTENT_TYPES is None:
            self.ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Data
    BATCH_SIZE: int = 16
    NUM_WORKERS: int = 2
    
    # Optimization
    LEARNING_RATE: float = 1e-4
    OPTIMIZER: str = "AdamW"
    WEIGHT_DECAY: float = 0.0
    
    # Regularization
    LABEL_SMOOTHING: float = 0.1
    
    # Training schedule
    NUM_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 7
    LR_SCHEDULER_PATIENCE: int = 3
    LR_REDUCTION_FACTOR: float = 0.1
    
    # Reproducibility
    SEED: int = 42


# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

@dataclass
class ExportConfig:
    """ONNX export parameters."""
    
    # Export settings
    OPSET_VERSION: int = 14  # ONNX opset version
    DO_CONSTANT_FOLDING: bool = True
    EXPORT_PARAMS: bool = True
    
    # Output
    OUTPUT_DIR: str = "models"
    MODEL_FILENAME: str = "skin_classifier_v1.onnx"
    
    # Validation
    NUM_VALIDATION_TESTS: int = 5
    NUMERICAL_TOLERANCE: float = 1e-5
    RELATIVE_TOLERANCE: float = 1e-4
    
    # Benchmarking
    NUM_WARMUP_ITERATIONS: int = 5
    NUM_BENCHMARK_ITERATIONS: int = 100


# ============================================================================
# COMPOSITE CONFIG
# ============================================================================

class Config:
    """Unified configuration object."""
    
    # Sub-configs
    model = ModelConfig()
    inference = InferenceConfig()
    api = APIConfig()
    training = TrainingConfig()
    export = ExportConfig()
    
    @classmethod
    def to_dict(cls) -> dict:
        """Export all config as dictionary."""
        return {
            "model": cls.model.__dict__,
            "inference": cls.inference.__dict__,
            "api": cls.api.__dict__,
            "training": cls.training.__dict__,
            "export": cls.export.__dict__,
        }
    
    @classmethod
    def to_json(cls, filepath: str):
        """Save config to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=2, default=str)
        print(f"Config saved to {filepath}")
    
    @classmethod
    def print_config(cls):
        """Pretty-print all configuration."""
        import json
        config_dict = cls.to_dict()
        print(json.dumps(config_dict, indent=2, default=str))


# ============================================================================
# SHORTCUTS
# ============================================================================

ModelConfig = Config.model
InferenceConfig = Config.inference
APIConfig = Config.api
TrainingConfig = Config.training
ExportConfig = Config.export


if __name__ == "__main__":
    Config.print_config()
    
    # You can also export to JSON for reproducibility
    # Config.to_json("config.json")