# ✍️ MNIST CNN Digit Recognizer

An interactive **MNIST handwritten digit recognition web app** built with **TensorFlow/Keras** and **Streamlit**.

The application uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits from **0 to 9**. Users can draw a digit directly on the canvas and get the model's prediction, confidence, and class probabilities.

## 🚀 Live Demo

👉 **[Try the MNIST CNN Digit Recognizer](https://mnist-cnn-project-mitwh7vhfi9tlt2slw3bvn.streamlit.app/)**

## ✨ Features

- 🎨 Draw digits directly on an interactive canvas
- 🤖 CNN-based digit classification
- 🔢 Predicts digits from **0–9**
- 📊 Probability chart for all 10 classes
- 🏆 Top 3 predictions
- 🎯 Prediction confidence
- 🧹 Clear canvas and draw another digit
- 🔍 View the processed **28×28** image
- 🌐 Clean Streamlit interface

## 🧠 Model Architecture

```text
Input (28×28×1)
      ↓
Conv2D (32 filters, 3×3, ReLU)
      ↓
MaxPooling2D (2×2)
      ↓
Flatten
      ↓
Dense (128, ReLU)
      ↓
Dense (10, Softmax)
```

### Training Configuration

- **Dataset:** MNIST
- **Input:** 28×28 grayscale images
- **Classes:** 10 (0–9)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Epochs:** 5
- **Batch Size:** 32

## 🔄 How It Works

1. Draw a digit on the black canvas using the white brush.
2. Convert the drawing to grayscale.
3. Detect and crop the digit.
4. Resize it while preserving its aspect ratio.
5. Center it inside a 28×28 canvas.
6. Normalize pixel values between 0 and 1.
7. Pass the processed image to the CNN.
8. Generate probabilities for all 10 digit classes.
9. Display the digit with the highest probability.

## 📁 Project Structure

```text
MNIST-CNN-Project/
│
├── app.py
├── cnn_model.h5
├── CNN.ipynb
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Matplotlib
- Streamlit
- Streamlit Drawable Canvas
- MNIST Dataset

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd MNIST-CNN-Project
```

### 2. Create an environment

Using Conda:

```bash
conda create -n mnist-cnn python=3.11
conda activate mnist-cnn
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

## 📦 Requirements

```text
streamlit
streamlit-drawable-canvas
tensorflow
numpy
matplotlib
pillow
```

## 🎯 Prediction Workflow

```text
Draw Digit
    ↓
Image Preprocessing
    ↓
28×28 Grayscale Image
    ↓
CNN Model
    ↓
10 Class Probabilities
    ↓
Prediction + Confidence
```

## 📊 Prediction Output

After clicking **Predict Digit**, the app displays:

- Predicted digit
- Confidence percentage
- Top 3 predictions
- Probability chart for digits 0–9
- Processed 28×28 image sent to the CNN

## 📌 Notes

The CNN expects MNIST-style **28×28 grayscale images**. The application preprocesses the hand-drawn digit automatically by cropping, resizing, centering, and normalizing it before prediction.

## 👨‍💻 Author

**Mudassir Khan**

Aspiring Data Scientist | Python | Machine Learning | Data Analytics

---

⭐ If you found this project useful, consider giving the repository a star!
