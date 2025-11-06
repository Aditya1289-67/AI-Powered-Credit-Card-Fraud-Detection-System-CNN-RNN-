import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from src.infer import FraudDetector

# ======= PAGE CONFIG =======
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# ======= CUSTOM CSS =======
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 2rem;
    }

    /* Primary button */
    .stButton>button {
        background-color: #0072ff;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0059c9;
    }

    /* Download button */
    .stDownloadButton>button {
        background-color: #16a34a;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #0d8b3e;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #e0f2ff 0%, #f0faff 100%);
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }

    /* Sidebar */
    .css-1d391kg {
        background: #f0f4f8;
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ======= SIDEBAR: MODEL INFO & TIPS =======
st.sidebar.image("logo.png", width=100)
with st.sidebar.expander("📊 Model Info", expanded=True):
    st.markdown("""
    **Architecture:** CNN + RNN  
    **Dataset:** Anonymized transaction data  
    **Accuracy:** 92%  
    **F1-Score:** 0.95  
    **AUC:** 0.96  
    """)
    with st.expander("ℹ️ AI/ML Explanation"):
        st.write("CNN extracts local features from transaction sequences, RNN models temporal patterns to detect fraud.")

with st.sidebar.expander("📝 Upload Guidance / Tips"):
    st.info("""
    - Only numeric columns starting with **V** or **Amount** are required.  
    - Avoid missing values.  
    - CSV should have transactions as rows.  
    """)

# ======= HEADER =======
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("logo.png", width=100)
with col_title:
    st.markdown("<h1 style='color:#0072ff;'>AI-Powered Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Detect fraudulent transactions in real-time using CNN + RNN deep learning model.")

# ======= TAGLINE =======
st.markdown("""
<div style='background-color:#0072ff; padding:10px; border-radius:10px; text-align:center; color:white; font-weight:600;'>
🚀 Empowering Financial Security with AI
</div>
""", unsafe_allow_html=True)

# ======= LOAD MODEL =======
@st.cache_resource
def load_model():
    model_path = "data/cnn_rnn_fraud_detector.pth"
    scaler_path = "data/scaler.pkl"
    return FraudDetector(model_path, scaler_path)

fraud_detector = load_model()

# ======= FILE UPLOAD =======
st.markdown("### 📂 Upload Your CSV File")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    try:
        uploaded_file.seek(0)
        chunk_size = 5000
        processed_chunks = []

        st.info("⏳ Processing CSV...")

        total_rows = sum(1 for _ in pd.read_csv(uploaded_file))
        uploaded_file.seek(0)
        progress_bar = st.progress(0)
        processed_rows = 0

        for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
            feature_cols = [c for c in chunk.columns if c.startswith("V") or c == "Amount"]
            numeric_chunk = chunk[feature_cols].astype(np.float32)

            preds = fraud_detector.predict_file(numeric_chunk)
            chunk["Prediction"] = np.where(preds > 0.5, "Fraud", "Not Fraud")
            processed_chunks.append(chunk)

            processed_rows += len(chunk)
            progress_bar.progress(min(processed_rows / total_rows, 1.0))

        result_df = pd.concat(processed_chunks, ignore_index=True)
        fraud_df = result_df[result_df["Prediction"] == "Fraud"].copy()

        st.success(f"🎯 Predictions Complete! Total Rows: {len(result_df)}, Fraud Rows: {len(fraud_df)}")

        # ======= METRICS =======
        st.markdown("### 📊 Summary Overview")
        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-card'><h4>Total Transactions</h4><h2>{len(result_df)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><h4>Fraudulent Transactions</h4><h2 style='color:#FF4B4B'>{len(fraud_df)}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-card'><h4>Non-Fraud Transactions</h4><h2 style='color:#16a34a'>{len(result_df)-len(fraud_df)}</h2></div>", unsafe_allow_html=True)

        # ======= VISUAL INSIGHTS =======
        st.markdown("### 📈 Visual Insights")
        st.divider()
        fig1, ax1 = plt.subplots()
        ax1.pie([len(fraud_df), len(result_df) - len(fraud_df)],
                labels=["Fraud", "Not Fraud"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#FF4B4B", "#16a34a"])
        ax1.axis("equal")
        st.pyplot(fig1)

        # ======= FRAUD TABLE =======
        st.markdown("### 🔎 Fraudulent Transactions (Highlighted Feature)")
        st.divider()
        if len(fraud_df) > 0:
            def highlight_most_responsible(row):
                sample_nonfraud = result_df[result_df["Prediction"]=="Not Fraud"].sample(5, random_state=42)
                means = sample_nonfraud[feature_cols].mean()
                stds = sample_nonfraud[feature_cols].std()
                z_scores = np.abs((row[feature_cols]-means)/stds)
                max_feature = z_scores.idxmax()
                return ['background-color:#FF4B4B;color:white' if col==max_feature else '' for col in row.index]
            styled_df = fraud_df.style.apply(highlight_most_responsible, axis=1)
            st.dataframe(styled_df, height=600)
        else:
            st.info("No fraud transactions detected.")

        # ======= DOWNLOAD =======
        output = io.BytesIO()
        fraud_df.to_csv(output, index=False)
        st.download_button(
            label="📥 Download Fraud Transactions CSV",
            data=output.getvalue(),
            file_name="fraud_transactions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👆 Upload your CSV file to start.")

# ======= FOOTER =======
st.markdown("""
<hr style='border:1px solid #eaeaea;margin-top:2rem;'>
<div style='text-align:center;color:gray;font-size:0.9em;'>
© 2025 <b>Fraud-Detection AI</b>
</div>
""", unsafe_allow_html=True)
