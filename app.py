import streamlit as st
import requests

st.title("Vector Element-wise Multiplication")

st.write("Enter two vectors of the same length (comma-separated):")

TESSERACT_URL = "http://localhost:8000/apply"  # Adjust if your Tesseract service uses a different URL or port

with st.form("vectormult_form"):
    a_str = st.text_input("Vector a", "1,2,3")
    b_str = st.text_input("Vector b", "4,5,6")
    submitted = st.form_submit_button("Multiply")

    if submitted:
        try:
            a = [float(x) for x in a_str.split(",")]
            b = [float(x) for x in b_str.split(",")]
            payload = {"inputs": {"a": a, "b": b}}
            response = requests.post(TESSERACT_URL, json=payload)
            if response.ok:
                result_obj = response.json().get("result", {})
                buffer = result_obj.get("data", {}).get("buffer", [])
                st.write("Result:", buffer)
            else:
                st.error(f"Tesseract error: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")