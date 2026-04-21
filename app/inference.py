import onnxruntime as ort
import numpy as np
import cv2

class SkinInference:
    def __init__(self, model_path):
        # Initialize the ONNX session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.classes = [
            "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions", 
            "Dermatofibroma", "Melanocytic nevi", "Melanoma", 
            "Vascular lesions", "Squamous cell carcinoma", "Unknown/Other"
        ]

    def preprocess(self, image_bytes):
        # Convert bytes to cv2 image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to your Phase 4 resolution (300x300)
        img = cv2.resize(img, (300, 300))
        
        # Normalize (Standard ImageNet values)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) # Force float32
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # Force float32
        img = (img - mean) / std
        
        # HWC to CHW and add batch dimension
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.astype(np.float32)
    

    def predict(self, image_bytes):
        input_data = self.preprocess(image_bytes)
        outputs = self.session.run(None, {self.input_name: input_data})
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0])) # Softmax
        
        # Return sorted results
        results = zip(self.classes, probs[0].tolist())
        return sorted(results, key=lambda x: x[1], reverse=True)