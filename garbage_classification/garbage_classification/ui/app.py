"""
ui/app.py
---------
Streamlit dashboard for the Garbage Classification service.
Talks to the FastAPI backend (API_URL below) -- it never touches the model
directly, exactly mirroring how a real frontend/backend split would work.

Run locally:
    streamlit run ui/app.py

Set API_URL as an environment variable when the API is deployed elsewhere,
e.g. API_URL=https://your-api.onrender.com
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

API_URL = os.environ.get(
    "API_URL",
    "https://garbageclassification.onrender.com"
)

st.set_page_config(page_title="Garbage Classifier", page_icon="🗑️", layout="wide")
st.title("🗑️ Garbage Classification Dashboard")

tab_predict, tab_insights, tab_retrain, tab_status = st.tabs(
    ["🔍 Predict", "📊 Data Insights", "🔁 Upload & Retrain", "⏱️ Model Status"]
)

st.write("API URL:", API_URL)

# --------------------------------------------------------------------------- #
# TAB 1 -- Predict a single image
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Classify a single image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded image")

        if st.button("Predict", type="primary"):
            with st.spinner("Running inference..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    resp.raise_for_status()
                    result = resp.json()

                    with col2:
                        st.success(f"Prediction: **{result['predicted_class'].upper()}**")
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                        probs_df = pd.DataFrame(
                            result["probabilities"].items(), columns=["class", "probability"]
                        ).sort_values("probability", ascending=False)
                        st.bar_chart(probs_df.set_index("class"))
                except requests.exceptions.RequestException as e:
                    st.error(f"API error: {e}")

# --------------------------------------------------------------------------- #
# TAB 2 -- Data insights / visualizations
# --------------------------------------------------------------------------- #
with tab_insights:
    st.subheader("Dataset insights")
    try:
        resp = requests.get(f"{API_URL}/insights", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        col1, col2, col3 = st.columns(3)

        train_counts = data["train_counts"]
        test_counts = data["test_counts"]
        pending_counts = data["pending_upload_counts"]

        with col1:
            st.markdown("**Training set class balance**")
            fig, ax = plt.subplots()
            ax.bar(train_counts.keys(), train_counts.values(), color="#2E8B57")
            ax.set_ylabel("Image count")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.caption(
                "Story: classes are roughly balanced (~1,700-2,100 images each "
                "after the train/test split), which is why we didn't need class "
                "weighting or heavy oversampling during training."
            )

        with col2:
            st.markdown("**Train vs. test split per class**")
            df = pd.DataFrame({"train": train_counts, "test": test_counts}).fillna(0)
            st.bar_chart(df)
            st.caption(
                "Story: the 85/15 split is preserved per-class (stratified), "
                "so test accuracy is a fair estimate of real-world performance "
                "for every material type, not just the majority classes."
            )

        with col3:
            st.markdown("**Pending images awaiting retraining**")
            if sum(pending_counts.values()) > 0:
                fig2, ax2 = plt.subplots()
                ax2.bar(pending_counts.keys(), pending_counts.values(), color="#CD5C5C")
                plt.xticks(rotation=45)
                st.pyplot(fig2)
                st.caption(
                    "Story: shows which classes users are contributing the most "
                    "new data for -- useful to spot if retraining data is skewed "
                    "toward one material before triggering a retrain."
                )
            else:
                st.info("No pending uploads yet -- upload images in the Retrain tab.")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API for insights: {e}")

# --------------------------------------------------------------------------- #
# TAB 3 -- Bulk upload + trigger retraining
# --------------------------------------------------------------------------- #
with tab_retrain:
    st.subheader("Upload new labeled images for retraining")
    label = st.selectbox("Label for these images",
                          ["cardboard", "glass", "metal", "paper", "plastic", "trash"])
    bulk_files = st.file_uploader(
        "Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if st.button("Upload for retraining"):
        if not bulk_files:
            st.warning("Select at least one file first.")
        else:
            with st.spinner("Uploading..."):
                files_payload = [("files", (f.name, f.getvalue())) for f in bulk_files]
                resp = requests.post(
                    f"{API_URL}/upload", data={"label": label}, files=files_payload, timeout=60
                )
                if resp.ok:
                    st.success(f"Uploaded {resp.json()['saved_count']} images labeled '{label}'.")
                else:
                    st.error(resp.text)

    st.divider()
    st.subheader("Trigger retraining")
    st.caption(
        "Retraining runs as a background job on the API server and fine-tunes "
        "the currently deployed model on the newly uploaded images."
    )

    if st.button("🔁 Retrain model now", type="primary"):
        resp = requests.post(f"{API_URL}/retrain", timeout=10)
        if resp.ok:
            st.success(resp.json()["message"])
        else:
            st.error(resp.text)

    if st.button("Check retraining status"):
        resp = requests.get(f"{API_URL}/retrain/status", timeout=10)
        st.json(resp.json())

# --------------------------------------------------------------------------- #
# TAB 4 -- Model / API status (uptime)
# --------------------------------------------------------------------------- #
with tab_status:
    st.subheader("Service health")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        info = requests.get(f"{API_URL}/model-info", timeout=5)

        col1, col2, col3 = st.columns(3)
        col1.metric("API status", health["status"].upper())
        col1.metric("Uptime (seconds)", f"{health['uptime_seconds']:.0f}")
        col2.metric("Model file present", str(health["model_file_present"]))
        if info.ok:
            info_json = info.json()
            col3.metric("Model size (MB)", info_json["size_mb"])
            st.caption(f"Last modified: {info_json['last_modified']}")
        else:
            st.warning("No model deployed yet -- train and save one to models/garbage_model.h5")
    except requests.exceptions.RequestException as e:
        st.error(f"API unreachable: {e}")

    if st.button("Refresh"):
        st.rerun()
