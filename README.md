# Setup & Usage Guide

## 1. Install Dependencies

Install all required libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 2. Download the Model (mistral-7b-instruct-v0.2.Q8_0.gguf)

You can download the model in either of the following ways:

### **Option A — Command Line**

```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf
```

### **Option B — Manual Download**

Download from Hugging Face:

**Link:**
[https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/tree/main)

Look for the file:
**`mistral-7b-instruct-v0.2.Q8_0.gguf`**
(Size: ~7.7 GB, free, runs locally)

---

## 3. Extract the Dataset

Unzip `data.zip` into the project directory.

---

## 4. Run the Inference Script

Use `main.py` if you want to run the model on custom input:

```bash
python main.py
```

---

## 5. Run the Evaluation Script

Use `evaluation.py` to evaluate performance on the test dataset:

```bash
python evaluation.py
```

---

# Running Everything in Google Colab / Jupyter Notebook

You can also run the full workflow using **complete_code.ipynb**, which contains the logic of both `main.py` and `evaluation.py`.

### Steps:

1. Upload `complete_code.ipynb` to Google Colab.
2. Open the notebook in a Colab session.
3. Upload `data.zip`.
4. Run all cells in order.
5. All necessary libraries and the model will be downloaded automatically.
6. The notebook includes:
   * Inference code (from `main.py`)
   * Evaluation code (from `evaluation.py`)
7. Execution is typically faster than running locally.
