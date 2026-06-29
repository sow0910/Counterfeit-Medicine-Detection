import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# 🔹 Model definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(512, 128)

    def forward_once(self, x):
        return self.backbone(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# 🔹 Load model
model = SiameseNetwork()
model.load_state_dict(torch.load("siamese_model.pth", map_location="cpu"))
model.eval()

# 🔹 Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# 🔹 UI Title
st.title("💊 Counterfeit Medicine Detection")

# 🔹 Description
st.write("This system uses a Siamese Neural Network to compare medicine packaging and detect counterfeit products.")

# 🔹 Upload Inputs
img1 = st.file_uploader("Upload Test Image", type=["jpg","png","jpeg"])
img2 = st.file_uploader("Upload Genuine Reference", type=["jpg","png","jpeg"])

# 🔹 If both images uploaded
if img1 and img2:

    img1 = Image.open(img1).convert("RGB")
    img2 = Image.open(img2).convert("RGB")

    # 🔹 Show images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(img1, caption="Test Image")

    with col2:
        st.image(img2, caption="Reference Image")

    # 🔹 Transform images
    img1_t = transform(img1).unsqueeze(0)
    img2_t = transform(img2).unsqueeze(0)

    # 🔹 Model prediction
    with torch.no_grad():
        out1, out2 = model(img1_t, img2_t)
        distance = torch.sqrt(torch.sum((out1 - out2)**2, dim=1)).item()

    # 🔹 Results
    st.subheader("🔍 Result")

    st.write(f"Similarity Score: {distance:.2f}")

    confidence = max(0, 1 - distance/2)
    st.write(f"Confidence: {confidence*100:.2f}%")

    if distance < 0.7:
        st.success("✅ GENUINE MEDICINE")
    else:
        st.error("❌ COUNTERFEIT MEDICINE")