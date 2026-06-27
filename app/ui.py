import streamlit as st
from PIL import Image
import sys
sys.path.insert(0, '.')

from app.inference import SkinInference


st.set_page_config(page_title="Med-Vision AI", page_icon="🩺")


@st.cache_resource
def load_model():
    sess = 'models/skin_classifier_v1.onnx'
    
    return SkinInference(sess)

predictor = load_model()

st.title("🩺 Med-Vision: Skin Lesion Classifier")
st.write("Upload a dermoscopic image to analyze for 9 categories of skin lesions.")

st.warning("⚠️ EDUCATIONAL PURPOSES ONLY. This is not a diagnostic tool.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Lesion'):
        with st.spinner('Running Optimized ONNX Inference...'):
            image_bytes = uploaded_file.getvalue()
            data = predictor.predict_with_metadata(image_bytes)
            st.subheader(f"Prediction: {data['prediction']}")
            st.progress(min(data['confidence'], 1.0))
            st.write(f"Confidence: {data['confidence']*100:.2f}%")
            
            if data.get('is_uncertain'):
                st.warning("⚠️ Model is uncertain about this prediction.")
            
            if data.get('requires_dermatologist_review'):
                st.error("🚨 " + data.get('clinical_note', 'Dermatologist review recommended.'))
            
            # Show all probabilities in a bar chart
            st.subheader("All Class Probabilities")
            st.bar_chart(dict(data['all_predictions']))