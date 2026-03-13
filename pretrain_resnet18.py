import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# ---- Custom Dataset ----
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
        return img1, img2  # two views for contrastive learning

# ---- SimCLR Transformations ----
def get_simclr_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ---- Projection Head ----
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---- SimCLR Model ----
class SimCLR(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        self.encoder.fc = nn.Identity()  # remove classification head
        self.projector = ProjectionHead(in_dim=512)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)

# ---- NT-Xent Loss ----
def nt_xent_loss(z_i, z_j, temperature=0.5):
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    sim /= temperature

    labels = torch.cat([torch.arange(N) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(z.device)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim = sim[~mask].view(sim.shape[0], -1)

    positives = sim[labels.bool()].view(labels.shape[0], -1)
    negatives = sim[~labels.bool()].view(sim.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# ---- Training ----
def train_simclr(data_dir, save_path, epochs=10, batch_size=128, lr=3e-4):
    transform = get_simclr_transform()
    dataset = FlatImageDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLR(models.resnet18(pretrained=False)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for x_i, x_j in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_i, x_j = x_i.to(device), x_j.to(device)

            z_i = model(x_i)
            z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Pretrained ResNet18 saved to: {save_path}")

# ---- Main ----
if __name__ == "__main__":
    data_dir = "/home/hlcv_team038/detr_dataset/clear/images/train"  # flat folder of KITTI images
    save_path = "/home/hlcv_team038/pretrained_models/resnet18_simclr_kitti_clean.pth"
    train_simclr(data_dir, save_path, epochs=100, batch_size=128)
