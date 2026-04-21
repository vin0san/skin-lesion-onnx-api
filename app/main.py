from fastapi import FastAPI, UploadFile, File
from app.inference import SkinInference

app = FastAPI(title="Skin Lesion Classifier API")
predictor = SkinInference("models/skin_classifier_v1.onnx")

@app.post("/predict")
async def predict_lesion(file: UploadFile = File(...)):
    # 1. Read the uploaded file
    image_bytes = await file.read()
    
    # 2. Run inference
    predictions = predictor.predict(image_bytes)

    top_pred = predictions[0][0]
    top_conf = predictions[0][1]
    
    needs_review = top_conf < 0.50 

    return {
        "filename": file.filename,
        "prediction": top_pred,
        "confidence": round(top_conf, 4),
        "status": "Inconclusive - High Uncertainty" if needs_review else "Categorized",
        "action_item": "Consult a dermatologist" if (top_pred == "Melanoma" or needs_review) else "Monitor for changes",
        "all_probabilities": {name: round(p, 4) for name, p in predictions}
    }