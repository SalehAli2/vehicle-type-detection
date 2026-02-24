import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import build_model

# ---- Config ----
CLASS_LABELS = ["bus", "car", "motorcycle", "truck"]
MODEL_PATH = "models/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Model ----
@st.cache_resource
def load_model():
    model = build_model(num_classes=len(CLASS_LABELS), device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

# ---- Preprocess Image ----
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# ---- UI ----
st.title("🚗 Vehicle Type Detection")
st.write("Upload an image and the model will classify the vehicle type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    model = load_model()
    input_tensor = preprocess(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        predicted_class = CLASS_LABELS[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100

    st.markdown(f"### Prediction: **{predicted_class}**")
    st.markdown(f"### Confidence: **{confidence:.2f}%**")

    st.write("#### All Probabilities:")
    for label, prob in zip(CLASS_LABELS, probabilities):
        st.progress(float(prob), text=f"{label}: {prob*100:.2f}%")