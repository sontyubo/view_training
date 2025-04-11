import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- 準備 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
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


# --- 補助関数 ---
def plot_and_save_loss(model_name, loss_list, filename):
    os.makedirs("logs/figures", exist_ok=True)
    path = f"logs/figures/{filename}"
    plt.figure()
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {model_name}")
    plt.legend()
    plt.savefig(path)
    plt.close()
    return path


def export_tensorboard_plot(model_name, log_dir="logs/tensorboard", tag="Loss/train"):
    path = os.path.join(log_dir, model_name)
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    scalars = event_acc.Scalars(tag)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]

    plt.figure()
    plt.plot(steps, values, label=tag)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"{tag} - {model_name}")
    plt.legend()
    os.makedirs("logs/figures", exist_ok=True)
    out_path = f"logs/figures/{model_name}_tensorboard_{tag.replace('/', '_')}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


# --- メイン関数 ---
def train_and_evaluate(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ログ設定
    writer = SummaryWriter(log_dir=f"logs/tensorboard/{model_name}")
    mlflow.set_tracking_uri("file://" + os.path.abspath("logs/mlruns"))
    mlflow.set_experiment("MNIST with TIMM")
    mlflow.start_run(run_name=model_name)
    mlflow.log_param("model", model_name)

    loss_history = []

    # 学習ループ
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(
            train_loader, desc=f"{model_name} - Epoch {epoch + 1}"
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # 評価
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    writer.add_scalar("Accuracy/test", acc, 0)
    mlflow.log_metric("test_accuracy", acc)

    # 画像保存（matplotlib）
    loss_plot_path = plot_and_save_loss(
        model_name, loss_history, f"{model_name}_loss_matplotlib.png"
    )
    mlflow.log_artifact(loss_plot_path)

    # 画像保存（TensorBoardから抽出）
    writer.close()  # 必ず閉じてから読み込み
    tb_plot_path = export_tensorboard_plot(model_name)
    mlflow.log_artifact(tb_plot_path)

    mlflow.end_run()

    print(f"✅ 結果保存完了: {loss_plot_path}, {tb_plot_path}")
