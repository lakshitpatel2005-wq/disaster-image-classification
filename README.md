# Disaster Image Classification using Deep Learning

This project uses deep learning to classify images into different disaster categories such as **flood, fire, earthquake, cyclone, and normal scenes**.
It leverages **transfer learning with MobileNetV2** to achieve efficient and accurate predictions.

---

## Features

* Classifies images into multiple disaster categories
* Uses transfer learning for better performance
* Supports custom image input for prediction
* Lightweight and beginner-friendly implementation

---

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib

---

## Project Structure

```
disaster-image-classification
│
├── dataset/
│   ├── flood/
│   ├── fire/
│   ├── earthquake/
│   ├── cyclone/
│   └── normal/
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses a disaster image dataset containing different categories of natural disasters.

Download a dataset from Kaggle and place it inside the `dataset/` folder with the following structure:
https://www.kaggle.com/datasets/alex1994/natural-disaster-image-dataset

```
dataset/
├── flood/
├── fire/
├── earthquake/
├── cyclone/
└── normal/
```

Each folder should contain images corresponding to its class.

---

## Installation

1. Clone the repository

```
git clone https://github.com/yourusername/disaster-image-classification.git
```

2. Navigate to the project directory

```
cd disaster-image-classification
```

3. Install dependencies

```
pip install -r requirements.txt
```

---

## Training the Model

Run the following command to train the model:

```
python train.py
```

This will train the model and save it as:

```
disaster_model.h5
```

---

## Making Predictions

To test the model on an image:

```
python predict.py
```

Make sure to replace the test image path inside the script.

---

## How It Works

1. Images are loaded and preprocessed (resized and normalized).
2. A pretrained MobileNetV2 model is used as the base.
3. Custom layers are added for classification.
4. The model is trained on disaster image categories.
5. The trained model predicts the class of new images.

---

## Applications

* Disaster monitoring systems
* Emergency response support
* Social media disaster detection
* AI for social good

---

## Future Improvements

* Increase dataset size for better accuracy
* Add real-time video classification
* Deploy as a web application
* Improve model performance using advanced architectures

---

## Author

Lakshit Patel
