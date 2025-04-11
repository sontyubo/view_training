import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 20)
print(f"Using device: {device}")
print("=" * 20)


# MNISTは1チャンネル画像 → 3チャンネルに変換
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # timmモデルに合わせる
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 学習＆評価関数
def train_and_evaluate(model, model_name):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):  # 小規模テスト用
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} - {model_name}"
        ):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(
            f"[{model_name}] Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}"
        )

    # 評価
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"[{model_name}] Accuracy: {100 * correct / total:.2f}%")
