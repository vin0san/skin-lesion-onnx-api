import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Med-Vision AI", page_icon="🩺")

st.title("🩺 Med-Vision: Skin Lesion Classifier")
st.write("Upload a dermoscopic image to analyze for 9 categories of skin lesions.")

st.warning("⚠️ EDUCATIONAL PURPOSES ONLY. This is not a diagnostic tool.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Lesion'):
        with st.spinner('Running Optimized ONNX Inference...'):
            # Send to FastAPI backend - CRITICAL: use tuple format (filename, bytes, content_type)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.subheader(f"Prediction: {data['prediction']}")
                st.progress(min(data['confidence'], 1.0))
                st.write(f"Confidence: {data['confidence']*100:.2f}%")
                
                if data.get('is_uncertain'):
                    st.warning("⚠️ Model is uncertain about this prediction.")
                
                if data.get('requires_dermatologist_review'):
                    st.error("🚨 " + data.get('clinical_note', 'Dermatologist review recommended.'))
                
                # Show all probabilities in a bar chart
                st.subheader("All Class Probabilities")
                st.bar_chart(data['all_probabilities'])
            else:
                st.error(f"Error communicating with API. Status: {response.status_code}")
                st.write(response.text)