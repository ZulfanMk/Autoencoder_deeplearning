import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# ===== Config =====
GRAY_DIR = 'D:/tugas_autoencoder/bola'
COLOR_DIR = 'D:/tugas_autoencoder/ball'
IMG_SIZE = (128, 128)
MAX_IMAGES = 100
BATCH_SIZE = 8
EPOCHS = 640
LR = 1e-3

# ===== Dataset Class =====
class ColorizationDataset(Dataset):
    def __init__(self, gray_dir, color_dir, img_size=(128, 128), scale='tanh', augment=False):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.img_size = img_size
        self.scale = scale
        self.augment = augment
        self.files = sorted([f for f in os.listdir(gray_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return min(len(self.files), MAX_IMAGES)

    def preprocess(self, img, mode):
        img = img.convert(mode).resize(self.img_size)
        img = np.array(img).astype('float32')
        if self.scale == 'tanh':
            img = (img / 127.5) - 1.0
        else:
            img /= 255.0
        return img

    def __getitem__(self, idx):
        gray_path = os.path.join(self.gray_dir, self.files[idx])
        color_path = os.path.join(self.color_dir, self.files[idx])
        
        gray_img = Image.open(gray_path)
        color_img = Image.open(color_path)

        # Augmentasi sederhana
        if self.augment and random.random() > 0.5:
            gray_img = gray_img.transpose(Image.FLIP_LEFT_RIGHT)
            color_img = color_img.transpose(Image.FLIP_LEFT_RIGHT)

        gray_arr = self.preprocess(gray_img, 'L')  # (H, W)
        color_arr = self.preprocess(color_img, 'RGB')  # (H, W, 3)

        gray_arr = np.repeat(gray_arr[:, :, np.newaxis], 3, axis=-1)  # (H, W, 3)

        gray_tensor = torch.tensor(gray_arr).permute(2, 0, 1).float()
        color_tensor = torch.tensor(color_arr).permute(2, 0, 1).float()

        return gray_tensor, color_tensor

# ===== DataLoader =====
dataset = ColorizationDataset(GRAY_DIR, COLOR_DIR, IMG_SIZE, scale='tanh', augment=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== Model =====
class DeeperAutoencoder(nn.Module):
    def __init__(self):
        super(DeeperAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ===== Training Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeeperAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===== Training Loop =====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 20 == 0:
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# ===== Visualisasi =====
def visualize(model, dataset, n=5):
    model.eval()
    fig, axs = plt.subplots(n, 3, figsize=(12, 4 * n))
    with torch.no_grad():
        for i in range(n):
            input_img, target_img = dataset[i]
            input_batch = input_img.unsqueeze(0).to(device)
            output = model(input_batch).squeeze().cpu()

            inp = ((input_img.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
            out = ((output.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
            tgt = ((target_img.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

            axs[i, 0].imshow(inp)
            axs[i, 0].set_title('Input (Sketch)')
            axs[i, 1].imshow(out)
            axs[i, 1].set_title('Output (Generated)')
            axs[i, 2].imshow(tgt)
            axs[i, 2].set_title('Target (Color)')
            for j in range(3):
                axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

visualize(model, dataset, n=10)
