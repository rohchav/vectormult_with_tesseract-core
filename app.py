import streamlit as st
import numpy as np
from tesseract_api import InputSchema, apply, apply_with_numpy

st.title("Vector Element-wise Multiplication")

st.write("Enter two vectors of the same length (comma-separated):")

with st.form("vectormult_form"):
    a_str = st.text_input("Vector a", "1,2,3")
    b_str = st.text_input("Vector b", "4,5,6")
    submitted = st.form_submit_button("Multiply")

    if submitted:
        try:
            # Convert input strings to numpy arrays
            a = np.array([float(x) for x in a_str.split(",")], dtype=np.float32)
            b = np.array([float(x) for x in b_str.split(",")], dtype=np.float32)
            # Create InputSchema instance
            inputs = InputSchema(a=a, b=b)
            # Get the output using the apply function
            output = apply_with_numpy(inputs)
            st.write("Result:", output.result)
        except Exception as e:
            st.error(f"Error: {e}")