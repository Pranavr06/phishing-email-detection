import streamlit as st
import joblib
import pandas as pd
import json
import os
from src.features.build_features import (
    URLFeatureExtractor, HTMLFeatureExtractor, 
    PunctuationFeatureExtractor, UrgencyFeatureExtractor
)

# Page Config
st.set_page_config(page_title="AI Phishing Detection", page_icon="🛡️", layout="centered")

# Custom UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #0F52BA;
        color: white;
        border-radius: 5px;
    }
    .disclaimer {
        font-size: 0.85em;
        color: #888888;
        text-align: center;
        margin-top: 50px;
    }
    .privacy {
        font-size: 0.9em;
        background-color: #FFF3CD;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #FFEBA8;
        margin-bottom: 20px;
    }
    .indicator-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("models/final_pipeline.joblib"):
        return None
    return joblib.load("models/final_pipeline.joblib")

@st.cache_data
def load_metadata():
    if not os.path.exists("models/model_metadata.json"):
        return None
    with open("models/model_metadata.json", "r") as f:
        return json.load(f)

# Header
st.title("🛡️ AI-Driven Phishing Email Detection")
st.markdown("### NLP & Machine Learning Based Email Threat Analysis")

st.markdown("""
<div class="privacy">
<b>Privacy Notice:</b> Avoid submitting confidential, personal, financial, or sensitive email content. This application is intended for educational and demonstration purposes. No data is logged or permanently stored.
</div>
""", unsafe_allow_html=True)

# Main input card
subject = st.text_input("Subject")
body = st.text_area("Email Body", height=250)

model = load_model()
metadata = load_metadata()

if model is None:
    st.error("Model file not found. Please train and serialize the model first.")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    analyze = st.button("Analyze Email")
with col2:
    if st.button("Clear / Reset"):
        st.rerun()

if analyze and (subject or body):
    try:
        df = pd.DataFrame([{"Subject": subject, "Body": body}])
        
        # Predict
        prediction = model.predict(df)[0]
        probabilities = model.predict_proba(df)[0]
        
        phishing_prob = probabilities[1]
        legit_prob = probabilities[0]
        
        is_phishing = prediction == 1
        
        # Result Display
        st.divider()
        if is_phishing:
            st.error(f"## 🚨 CLASSIFICATION: PHISHING")
            st.markdown(f"**Model probability:** {phishing_prob * 100:.2f}%", help="This probability reflects the model's output and is not a guarantee that the email is safe or malicious.")
        else:
            st.success(f"## ✅ CLASSIFICATION: LEGITIMATE")
            st.markdown(f"**Model probability:** {legit_prob * 100:.2f}%", help="This probability reflects the model's output and is not a guarantee that the email is safe or malicious.")
            
        # Indicators
        st.markdown("### Detected Email Indicators")
        
        # Extract features manually to avoid depending on pipeline internals which may break if structure changes
        from src.features.build_features import (
            URLFeatureExtractor, HTMLFeatureExtractor, 
            PunctuationFeatureExtractor, UrgencyFeatureExtractor
        )
        url_feats = URLFeatureExtractor().transform(df)
        html_feats = HTMLFeatureExtractor().transform(df)
        punct_feats = PunctuationFeatureExtractor().transform(df)
        urg_feats = UrgencyFeatureExtractor().transform(df)
        
        indicators_html = f"""
        <div class="indicator-box">
            <ul>
                <li><b>Real URLs detected:</b> {url_feats['num_real_urls'].iloc[0]}</li>
                <li><b>Masked [URL] tokens:</b> {url_feats['num_masked_urls'].iloc[0]}</li>
                <li><b>HTML detected:</b> {'Yes' if html_feats['has_html'].iloc[0] == 1 else 'No'} ({html_feats['num_html_tags'].iloc[0]} tags, {html_feats['num_anchor_tags'].iloc[0]} links)</li>
                <li><b>Exclamation marks:</b> {punct_feats['num_exclamation_marks'].iloc[0]}</li>
                <li><b>Uppercase ratio:</b> {punct_feats['uppercase_ratio'].iloc[0]:.2f}</li>
                <li><b>Urgency phrases detected:</b> {urg_feats['urgency_keyword_count'].iloc[0]}</li>
            </ul>
        </div>
        """
        st.markdown(indicators_html, unsafe_allow_html=True)
        
        with st.expander("Technical Analysis / Model Metadata"):
            if metadata:
                st.code(json.dumps(metadata, indent=4), language="json")
            else:
                st.write("Metadata not available.")
                
    except Exception as e:
        st.error("An error occurred during analysis. Please check your input format.")

st.markdown("""
<div class="disclaimer">
<b>Disclaimer:</b> The evaluated models achieved perfect classification on the provided dataset. Dataset auditing revealed strong synthetic and structural artifacts, so these results should not be interpreted as evidence of equivalent real-world performance. ML-based phishing detection can produce false positives and false negatives. Do not use this as a sole security mechanism.
</div>
""", unsafe_allow_html=True)
