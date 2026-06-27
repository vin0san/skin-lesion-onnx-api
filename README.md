# Skin Lesion Classification — EfficientNet-B3 ONNX API
**[Live Demo](https://vin0san-skin-lesion-classifier.hf.space)** | **[Blog Post](https://mlbuild.hashnode.dev)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/onnx-1.14+-green.svg)](https://onnx.ai/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.104+-black.svg)](https://fastapi.tiangolo.com/)

Production-grade skin lesion classifier trained on **9-class dermoscopic imagery** with ONNX export for edge deployment.

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Macro F1 Score** | 0.6587 |
| **Balanced Accuracy** | 0.6759 |
| **Model Architecture** | EfficientNet-B3 |
| **Model Size (PyTorch)** | 41 MB |
| **Model Size (ONNX)** | 0.9 MB |
| **Compression** | 45× smaller |
| **ONNX Export** | Validated (max diff: 4e-6) |
| **Training Dataset** | ISIC (2,239 train / 118 val) |
| **Classes** | 9 skin lesion types |
| **Early Stopping** | Epoch 9 |

## 📋 Classification Classes

1. Actinic keratoses
2. Basal cell carcinoma
3. Benign keratosis-like lesions
4. Dermatofibroma
5. Melanocytic nevi
6. **Melanoma** (most dangerous)
7. Vascular lesions
8. Squamous cell carcinoma
9. Unknown/Other

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Start API
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/health
```

## 🔬 Single Prediction

```python
from app.inference import SkinInference

predictor = SkinInference("models/skin_classifier_v1.onnx")

with open("lesion.jpg", "rb") as f:
    predictions = predictor.predict(f.read())
    
for class_name, confidence in predictions[:3]:
    print(f"{class_name}: {confidence:.4f}")
```

## 📊 Training Pipeline

```
ISIC Dataset (9-class)
    ↓
ImageFolder + Augmentation (flip, rotate, color jitter)
    ↓
EfficientNet-B3 + Class Weighting
    ↓
AdamW (lr=1e-4) + ReduceLROnPlateau
    ↓
Macro F1 tracking + Early Stopping (patience=7)
    ↓
ONNX Export + Numerical Validation
```

## 🏗️ Architecture

### Model
- **Backbone:** EfficientNet-B3 (ImageNet pretrained)
- **Input:** 300×300 RGB images
- **Output:** 9-class logits
- **Optimization:** ONNX Runtime (no PyTorch dependency)

### Data Processing
- Resize to 300×300
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Train augmentation: horizontal/vertical flip, 90° rotation, color jitter
- Val: resize + normalize only

### Class Imbalance Handling
- Per-class weights: `weight = N / (K × n_k)`
- Label smoothing: 0.1
- Balanced accuracy metric alongside F1

## 📁 Repository Structure

```
skin-lesion-onnx-api/
├── models/
│   ├── best_model.pth                 # PyTorch weights
│   ├── skin_classifier_v1.onnx        # ONNX model
│   ├── metrics.json                   # Training curves
│   └── training_metadata.json         # Benchmark data
├── app/
│   ├── main.py                        # FastAPI server
│   ├── inference.py                   # ONNX inference
│   ├── __init__.py
|   └── ui.py                          # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb              # EDA and training logs
├── results/
│   ├── 01_baseline                    # Baseline results
│   ├── 02_augmentation                # With augmentation
│   ├── 03_class_weighted              # With class weighting
│   ├── 04_scheduler                   # With learning rate scheduler
│   └── 05_final                       # Final model results
├── scripts/
│   ├── run.py                         # Complete Pipeline run script
├── src/
│   ├── data/                          # Data loading and augmentation
│   ├── model/                         # Model architectures
│   ├── train.py                       # Training loop
|   └── export_to_onnx.py              # ONNX export and validation
├── config.py                          # Configuration
├── test_inference.py                  # Tests
├── requirements.txt
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest test_inference.py -v

# Specific test
pytest test_inference.py::TestInference -v
```

**Coverage:**
- ✅ Image preprocessing (JPEG/PNG)
- ✅ Inference correctness
- ✅ ONNX validation
- ✅ API error handling

## 📋 API Reference

### POST `/predict`
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@lesion.jpg"
```

### POST `/predict_batch`
```bash
curl -X POST "http://localhost:8000/predict_batch" \
  -F "files=@img1.jpg" -F "files=@img2.jpg"
```

### GET `/health`
```bash
curl http://localhost:8000/health
```

## ⚙️ Configuration

All parameters in `config.py`. Override via env:

```bash
export API_HOST="0.0.0.0"
export API_PORT="8000"
export MODEL_PATH="models/skin_classifier_v1.onnx"
python -m uvicorn app.main:app
```

## 🔍 Key Implementation Details

**ONNX Export Validation:**
1. ONNX model structure checked
2. Numerical equivalence: PyTorch vs ONNX (max diff < 4e-6)
3. Inference reproducible across runs

**Error Handling:**
- Image validation (JPEG/PNG, max 10 MB)
- Corrupt image detection
- Batch prediction support (max 50 images)
- Uncertainty flagging (confidence < 0.50)

## ⚠️ Limitations

1. **Medical Disclaimer:** Research tool, not clinical device. Consult dermatologist.
2. **Dataset Bias:** ISIC has demographic biases, performance varies by skin tone.
3. **Input:** JPEG/PNG only, max 10 MB, optimized for 300×300.

## 🚀 Production Deployment

### Docker
```bash
docker build -t skin-lesion-api:v1 .
docker run -p 8000:8000 skin-lesion-api:v1
```

### Environment
```bash
export MODEL_PATH="models/skin_classifier_v1.onnx"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🔄 Reproducibility

- **Seed:** 42
- **Hyperparameters:** All in `config.py`
- **ONNX Opset:** 14
- **Validation:** Numerical equivalence tested

## 📦 Dependencies

```
torch>=2.0.0
onnxruntime>=1.16.0
fastapi>=0.104.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

See `requirements.txt`.

## 🎓 Training Ablation

| Experiment | Macro F1 |
|------------|----------|
| Baseline | 0.46 |
| + Augmentation | 0.51 |
| + Class Weights | 0.45 |
| + Both | 0.53 |
| + Scheduler (EfficientNet-B3) | **0.6587** |

**Key insight:** Scheduler + higher resolution + better architecture = significant improvement.

## 📈 Future Work

- [ ] Explainability (saliency maps)
- [ ] Multi-model ensemble
- [ ] GPU optimization (TensorRT)
- [ ] Fine-tuning on additional datasets

## 📝 Citation (In-case)

```bibtex
@software{skin_lesion_2024,
  title={Skin Lesion Classification with EfficientNet-B3 ONNX API},
  author={Vineet},
  year={2024},
  url={https://github.com/vin0san/skin-lesion-onnx-api}
}
```

## 📄 License

[MIT](LICENSE.txt)

## 📧 Contact

Questions? Open a GitHub issue.

---

**Status:** Production-Ready ✅ | Last Updated: May 2026
