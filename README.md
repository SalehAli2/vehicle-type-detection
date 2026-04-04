# 🚗 Vehicle Type Detection

A deep learning web app that classifies vehicle types from images using **MobileNetV2** and **PyTorch**, deployed with **Streamlit**.

## 🔗 Live Demo
👉 [Click here to try the app](https://vehicle-type-detection-fwh6qjajw6fh8axxdjscre.streamlit.app/)

## 📌 Supported Classes

This model can detect the following **17 vehicle types**:

- 🚑 Ambulance
- 🚤 Barge
- 🚲 Bicycle
- ⛴️ Boat
- 🚌 Bus
- 🚗 Car
- 🛒 Cart
- 🚜 Caterpillar
- 🚁 Helicopter
- 🛎️ Limousine
- 🏍️ Motorcycle
- 🛹 Segway
- ❄️ Snowmobile
- 🛡️ Tank
- 🚕 Taxi
- 🚛 Truck
- 🚐 Van

## 🛠️ Tech Stack
- Python
- PyTorch
- Torchvision (MobileNetV2)
- Streamlit
- Roboflow (dataset)

## 📁 Project Structure
```
vehicle-type-detection/
├── data_loader.py       # Data loading and preprocessing
├── model.py             # MobileNetV2 model definition
├── train.py             # Training script
├── evaluate.py          # Evaluation and confusion matrix
├── app.py               # Streamlit web app
├── download_data.py     # Download dataset from Roboflow
├── requirements.txt     # Dependencies
└── models/
    └── best_model.pth   # Trained model
```


## 📊 Results
| Metric | Value |
|--------|-------|
| Train Accuracy | 86% |
| Validation Accuracy | 85% |
| Test Accuracy | 84% |

## 📸 Dataset
Dataset sourced from [Roboflow](https://roboflow.com) with 29000 training images across 17 vehicle classes.


