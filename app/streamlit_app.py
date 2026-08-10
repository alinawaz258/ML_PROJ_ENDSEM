import sys
from pathlib import Path
import os
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import plotly.express as px
from PIL import Image

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common import load_config
from src.deployment.predictor import Predictor

# Streamlit Page Config
st.set_page_config(
    page_title='Tomato Leaf Disease Detector',
    page_icon='🍅',
    layout='wide'
)

# Load config
try:
    cfg = load_config()
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

# Disease descriptions dictionary
DISEASE_DESCRIPTIONS = {
    'Tomato_Bacterial_spot': 'Bacterial spot is caused by Xanthomonas campestris pv. vesicatoria. Symptoms include small, water-soaked spots on leaves that turn brown and necrotic.',
    'Tomato_Early_blight': 'Early blight is caused by the fungus Alternaria solani. It appears as small, brown lesions with concentric rings on older leaves.',
    'Tomato_Late_blight': 'Late blight is caused by the oomycete Phytophthora infestans. It presents as large, irregular, water-soaked lesions on leaves and stems.',
    'Tomato_Leaf_Mold': 'Leaf mold is caused by Passalora fulva. It typically occurs in humid greenhouses, appearing as pale greenish-yellow spots on the upper leaf surface.',
    'Tomato_Septoria_leaf_spot': 'Septoria leaf spot is caused by Septoria lycopersici. It forms numerous small, circular spots with dark borders and gray centers on leaves.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Spider mite damage appears as tiny yellow or white speckles on leaves. Severe infestations lead to webbing and leaf yellowing.',
    'Tomato_Target_Spot': 'Target spot is caused by Corynespora cassiicola. It shows as dark brown lesions with concentric rings, similar to early blight but often smaller.',
    'Tomato_Tomato_YellowLeaf__Curl_Virus': 'TYLCV is transmitted by whiteflies. It causes severe stunting, leaf cupping, and yellowing of leaf margins.',
    'Tomato_Tomato_mosaic_virus': 'Tomato mosaic virus causes mottling, mosaic patterns, and distortion of leaves. It can significantly reduce yield.',
    'Tomato_healthy': 'The plant appears healthy with no visible signs of disease or pest infestation. Continue regular care and monitoring.'
}

@st.cache_resource
def load_predictor(model_type):
    return Predictor(cfg=cfg, model_type=model_type)

def main():
    st.title('🍅 Tomato Leaf Disease Detector')
    st.markdown("Upload a picture of a tomato leaf to detect potential diseases.")
    
    # Sidebar
    st.sidebar.header("Settings")
    model_options = {
        'EfficientNetB0': 'efficientnet',
        'Custom CNN': 'cnn',
        'SVM': 'svm',
        'Random Forest': 'random_forest'
    }
    selected_model_display = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    selected_model_key = model_options[selected_model_display]
    
    st.sidebar.info(
        "**Note:** Ensure that the required models are available in the `artifacts/models/` directory. "
        "Deep learning models typically provide higher accuracy."
    )
    
    # Main area
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Prediction Results")
            with st.spinner(f"Loading {selected_model_display} model and predicting..."):
                try:
                    predictor = load_predictor(selected_model_key)
                    
                    # Save uploaded file to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    results = predictor.predict(tmp_path)
                    
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                        
                    pred_class = results['class_name']
                    confidence = results['confidence']
                    probs = results['all_probabilities']
                    
                    st.markdown(f"### Detected: **{pred_class.replace('_', ' ')}**")
                    
                    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                    
                    st.markdown("#### Description")
                    norm_key = pred_class.replace(' ', '_')
                    desc = DISEASE_DESCRIPTIONS.get(pred_class, DISEASE_DESCRIPTIONS.get(f"Tomato_{norm_key}", DISEASE_DESCRIPTIONS.get(f"Tomato__{norm_key}", "Description not available.")))
                    st.info(desc)
                    
                    # Display probabilities chart
                    prob_df = pd.DataFrame({
                        'Disease': [k.replace('_', ' ') for k in probs.keys()],
                        'Probability': list(probs.values())
                    }).sort_values('Probability', ascending=True)
                    
                    fig = px.bar(
                        prob_df, 
                        x='Probability', 
                        y='Disease', 
                        orientation='h',
                        title='Class Probabilities',
                        color='Probability',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.exception(e)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: gray;'>Tomato Leaf Disease Detection Project | Machine Learning End Semester</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
