# vectormult_with_tesseract-core

A simple Python project demonstrating element-wise vector multiplication using [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) and a Streamlit web interface.

## Features

- Element-wise multiplication of two user-provided vectors
- Input validation using Pydantic schemas
- Interactive web UI built with Streamlit
- Example API for integration with Tesseract Core

## Requirements

- Python 3.8 or newer
- [tesseract-core](https://github.com/pasteurlabs/tesseract-core) Python package (see below)

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/rohchav/vectormult_with_tesseract-core.git
    cd vectormult_with_tesseract-core
    ```

2. **(Recommended) Create and activate a virtual environment:**
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r tesseract_requirements.txt
    ```

    > **Note:**  
    > If `tesseract-core` is not available on PyPI, follow the [installation instructions here](https://github.com/pasteurlabs/tesseract-core) or add it to `tesseract_requirements.txt` as a Git URL if needed.

## Usage

1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Interact with the web interface:**
    - Open the provided local URL in your browser (usually `http://localhost:8501`).
    - Enter two vectors of the same length in the input fields, using comma-separated values (e.g., `1,2,3` and `4,5,6`).
    - Click the **Multiply** button.

3. **View the results:**
    - The app will display the element-wise product of the two vectors.
    - If the vectors are not the same length or the input is invalid, an error message will be shown.

### Example

- **Input:**
    - Vector a: `1,2,3`
    - Vector b: `4,5,6`
- **Output:**
    - Result: `[4. 10. 18.]`

This tool is useful for:
- Demonstrating API and UI integration with Tesseract Core
- Educational purposes (vector operations, Python data validation, web app prototyping)
- Rapid prototyping of mathematical operations with user input

## Files

- `app.py` – Streamlit web app for user interaction
- `tesseract_api.py` – API and schema definitions for vector multiplication
- `tesseract_requirements.txt` – Python dependencies
- `tesseract_config.yaml` – Tesseract project configuration
