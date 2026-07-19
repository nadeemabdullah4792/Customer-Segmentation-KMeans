import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Customer Segmentation", page_icon="👥")

model_path = os.path.join(os.path.dirname(__file__), "kmeans_customer_segmentation.pkl")
project = joblib.load(model_path)

model = project["model"]

st.title("👥 Customer Segmentation using K-Means")

st.write("Enter the customer's details below.")

income = st.number_input(
    "Annual Income (k$)",
    min_value=15.0,
    max_value=140.0,
    value=60.0
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1.0,
    max_value=100.0,
    value=50.0
)

if st.button("🔍 Predict Customer Segment"):

    data = np.array([[income, score]])

    cluster = int(model.predict(data)[0])

    st.success(f"Customer belongs to Cluster {cluster}")

    descriptions = {
        0: "Typical customer in Segment 0",
        1: "Typical customer in Segment 1",
        2: "Typical customer in Segment 2",
        3: "Typical customer in Segment 3",
        4: "Typical customer in Segment 4"
    }

    st.info(descriptions.get(cluster, "Customer segment identified successfully."))
