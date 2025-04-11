import matplotlib.pyplot as plt
import os


def plot_loss_curve(model_name, loss_list):
    plt.figure()
    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve - {model_name}")
    plt.legend()
    os.makedirs("logs/figures", exist_ok=True)
    path = f"logs/figures/{model_name}_loss.png"
    plt.savefig(path)
    plt.close()
