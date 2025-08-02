# 🖐️ Real-Time Hand Gesture Drawing and Recognition

This project is a Python application that allows users to draw on a live webcam feed using hand gestures and then uses a trained deep learning model to recognize the drawing. The application is built with OpenCV for video processing, MediaPipe for hand tracking, and PyTorch for the image recognition model.

![Demo Screenshot](https://i.imgur.com/your-demo-image.gif) ---

## ✨ Features

* **Gesture-Based Drawing**: Draw on the screen using your index and middle fingers.
* **Dynamic Controls**:
    * **Color Selection**: Point with your index finger to select colors from an on-screen palette.
    * **Eraser**: Use a full open palm to erase parts of the drawing.
    * **Dynamic Brush Size**: Adjust brush thickness by changing the distance between your thumb and index finger on your left hand.
    * **Clear Canvas**: Show an open palm on your left hand to clear the screen instantly.
* **AI-Powered Recognition**: Press a key to have a trained ResNet/Vision Transformer model predict what you've drawn.
* **Modular Training Script**: A separate script (`train.py`) is provided to train the recognition model on your own dataset.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
* Python 3.12
* A webcam

---

## 🚀 Setup and Installation

Follow these steps to set up the project environment.

### 1. Clone the Repository

```bash
git clone https://github.com/ChinmayBansal010/Hand-Gesture-Canva.git
cd your-repo-name
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Libraries

Install them using pip:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

The model is trained on a dataset of sketch images. Your data should be organized into subdirectories, where each subdirectory name corresponds to a class label.

```
data/
├── airplane/
│   ├── image1.png
│   └── image2.png
├── apple/
│   ├── image3.png
│   └── image4.png
└── ...
```

After organizing your data, you need a `filelist.txt` that contains the relative paths to all your images. For example:

**`data/filelist.txt`**:
```
airplane/image1.png
airplane/image2.png
apple/image3.png
apple/image4.png
```

---

## ⚙️ Usage

The project is split into three main parts: data preparation, model training, and running the application.

### 1. Generate Class Names

Before running the main application, you need to generate a `class_names.txt` file from your `filelist.txt`. This file maps the model's output index to a human-readable label.

Run the `filelist.py` script:

```bash
python filelist.py
```

This will create `data/class_names.txt`. You need to **move this file** to the root directory of the project where `main.py` is located.

### 2. Train the Model

To train the drawing recognition model, run the `train.py` script.

```bash
python train.py
```

* You can configure training parameters like `EPOCHS`, `BATCH_SIZE`, and model architecture (`USE_VIT`) directly inside `train.py`.
* The script will save the best-performing model as `sketch_model_best.pt` and the final model as `sketch_model_final.pt`.

### 3. Run the Drawing Application

Once the model is trained and `class_names.txt` is in place, you can start the main application.

```bash
python main.py
```

#### Controls:

* **Right Hand Gestures**:
    * **Draw**: Index + Middle finger up.
    * **Select Color**: Index finger up to point at the on-screen palette.
    * **Erase**: Full open palm.
* **Left Hand Gestures**:
    * **Adjust Brush Size**: Vary distance between thumb and index finger.
    * **Clear Canvas**: Full open palm.
* **Keyboard Hotkeys**:
    * **`P`**: Predict the current drawing on the canvas.
    * **`C`**: Clear the canvas.
    * **`S`**: Save the current drawing as `hand_drawing_output.png`.
    * **`Q`**: Quit the application.

---

## 🙏 Acknowledgements

* **MediaPipe** by Google for the robust hand tracking solution.
* **PyTorch** team for the deep learning framework.
* **OpenCV** for real-time computer vision capabilities.
