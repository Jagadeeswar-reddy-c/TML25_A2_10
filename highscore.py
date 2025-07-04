import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import io, base64, json, pickle, os, sys, time
import numpy as np
from PIL import Image

# === SETUP ===
TOKEN = "49390889"
response = requests.get("http://34.122.51.94:9090/stealing_launch", headers={"token": TOKEN})
import time
time.sleep(10)  # wait for 10 seconds before calling the API

answer = response.json()
print(answer)
if 'detail' in answer:
    sys.exit(1)

SEED = str(answer['seed'])
PORT = str(answer['port'])

# data = {'seed': 6239932, 'port': '9578'}
# SEED = str(data['seed'])
# PORT = str(data['port'])

# === WAIT FOR API TO START ===
print(f"Using port {PORT} (seed {SEED}) — waiting for server to boot...")
time.sleep(30)

# === TRANSFORMS WITH AUGMENTATION ===
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.ToTensor(),
])

# === CUSTOM DATASET CLASS ===
class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img.convert("RGB"))
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

# === LOAD DATASET ===
dataset = torch.load("ModelStealingPub.pt", weights_only=False)

# === API QUERY FUNCTION ===
def model_stealing(images, port):
    url = f"http://34.122.51.94:{port}/query"
    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)

    payload = json.dumps(image_data)
    response = requests.get(url, files={"file": payload}, headers={"token": TOKEN})
    if response.status_code == 200:
        return response.json()["representations"]
    else:
        raise Exception(f"Query failed: {response.status_code}, content: {response.json()}")

# === MULTI-BATCH QUERYING ===
batch_size = 1000
num_batches = 10
all_imgs = dataset.imgs
all_images = []
all_reps = []

for i in range(num_batches):
    batch_imgs = all_imgs[i*batch_size : (i+1)*batch_size]
    
    try:
        reps = model_stealing(batch_imgs, PORT)
        all_images.extend(batch_imgs)
        all_reps.extend(reps)
        print(f"✅ Queried batch {i+1}/{num_batches}")
    except Exception as e:
        print(f"⚠️  Error on batch {i+1}: {e}")
        break

    if i != num_batches - 1:
        print("⏳ Waiting 60 seconds before next query...")
        time.sleep(60)

if len(all_images) == 0:
    print("❌ No images successfully queried. Exiting.")
    sys.exit(1)

# Save queried results
with open('full_out.pickle', 'wb') as f:
    pickle.dump((all_images, all_reps), f)

# === MODEL DEFINITION (ResNet18 variant) ===
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1024)

    def forward(self, x):
        return self.backbone(x)

model = ResNetEncoder()

# === TRAINING SETUP ===
def l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.SmoothL1Loss()

images_tensor = torch.stack([transform(img.convert("RGB")) for img in all_images])
targets_tensor = torch.tensor(all_reps, dtype=torch.float32)

for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    outputs = l2_normalize(model(images_tensor))
    targets = l2_normalize(targets_tensor)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"📉 Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === EXPORT TO ONNX ===
path = "stolen_model.onnx"
torch.onnx.export(
    model, torch.randn(1, 3, 32, 32), path,
    input_names=["x"], output_names=["output"],
    export_params=True
)
print("✅ ONNX model exported to 'stolen_model.onnx'")

# === VALIDATE AND SUBMIT ===
import onnxruntime as ort

# Run validation checks
with open(path, "rb") as f:
    model_bytes = f.read()
    try:
        stolen_model = ort.InferenceSession(model_bytes)
    except Exception as e:
        raise Exception(f"Invalid model, {e=}")
    try:
        out = stolen_model.run(None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)})[0][0]
    except Exception as e:
        raise Exception(f"Some issue with the input, {e=}")
    assert out.shape == (1024,), f"Invalid output shape: {out.shape}"

# Submit model to evaluation server
response = requests.post(
    "http://34.122.51.94:9090/stealing",
    files={"file": open(path, "rb")},
    headers={"token": TOKEN, "seed": SEED}
)
print("📬 Server response:", response.json())