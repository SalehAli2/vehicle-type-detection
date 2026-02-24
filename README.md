# 🚗 Vehicle Type Detection

A deep learning web app that classifies vehicle types from images using **MobileNetV2** and **PyTorch**, deployed with **Streamlit**.

## 🔗 Live Demo
👉 [Click here to try the app](https://vehicle-type-detection-fwh6qjajw6fh8axxdjscre.streamlit.app/)

## 📌 Classes
The model can detect the following vehicle types:
- 🚌 Bus
- 🚗 Car
- 🏍️ Motorcycle
- 🚛 Truck

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

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/SalehAli2/vehicle-type-detection
cd vehicle-type-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

## 📊 Results
| Metric | Value |
|--------|-------|
| Train Accuracy | 97%+ |
| Validation Accuracy | 87.5% |
| Test Accuracy | 79% |

## 📸 Dataset
Dataset sourced from [Roboflow](https://roboflow.com) with 420 training images across 4 vehicle classes.


