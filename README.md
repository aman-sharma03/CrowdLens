
# 👤 Gender Detection Using CNN and Webcam

This project detects gender (Male/Woman) from live webcam feed using a Convolutional Neural Network (CNN) trained on facial images.

---

## 📂 Project Structure

- `train.py` – Trains a CNN on facial images for gender classification  
- `detect_gender_webcam.py` – Uses the trained model to detect gender via webcam  
- `gender_detection.h5` – Saved Keras model after training  
- `plot.png` – Training/validation loss and accuracy graph  

---

## 🧠 Model Summary

- CNN built with:
  - Convolutional layers (Conv2D)
  - BatchNormalization
  - MaxPooling
  - Dropout
  - Dense layers
- Trained on 96x96 facial images
- Optimized using Adam and Binary Crossentropy
- Uses `ImageDataGenerator` for image augmentation

---

## 🖼️ Dataset

You’ll need a dataset with two subfolders:
- `man/` – All male face images  
- `woman/` – All female face images  

### 📥 Recommended Dataset

Use the **UTKFace Dataset**  
🔗 https://susanqq.github.io/UTKFace/

After downloading, extract and organize it like this (you may need to manually sort the images by gender):

```

gender\_dataset\_face/
├── man/
├── woman/

````

Place this folder in the root of the project directory.

---

## 🧪 Installation Instructions

### ✅ Python Version

- Python 3.7 or higher recommended

### ✅ Required Packages

Install dependencies using pip:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn
````

---

## 🏋️‍♂️ How to Train the Model

* Ensure your dataset is in the folder `gender_dataset_face/` with subfolders `man/` and `woman/`
* Run the training script:

```bash
python train.py
```

* This will:

  * Load and preprocess the images
  * Train a CNN model for 100 epochs
  * Save the trained model as `gender_detection.h5`
  * Save a training plot as `plot.png`

---

## 🎥 How to Run Gender Detection with Webcam

Once training is complete and `gender_detection.h5` is created:

* Run the webcam-based gender detector:

```bash
python detect_gender_webcam.py
```

* This will:

  * Launch your webcam
  * Detect faces using Haar Cascades
  * Predict gender for each face in real-time
  * Display gender labels and count of men/women on the screen

* Press `Q` to quit the window

---

## 🧰 Technologies Used

* TensorFlow / Keras – for building and training the CNN model
* OpenCV – for image processing and webcam access
* NumPy – for numerical operations
* Matplotlib – for training visualization
* scikit-learn – for dataset splitting

---

## 📈 Output

The training script produces a plot like this (`plot.png`):

* 📉 Training loss
* 📈 Validation loss
* ✅ Training accuracy
* 🟢 Validation accuracy


