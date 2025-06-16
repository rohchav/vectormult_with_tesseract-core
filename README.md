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

2. **Enter two vectors (comma-separated) and view the result.**

## Files

- `app.py` – Streamlit web app for user interaction
- `tesseract_api.py` – API and schema definitions for vector multiplication
- `tesseract_requirements.txt` – Python dependencies
- `tesseract_config.yaml` – Tesseract project configuration
