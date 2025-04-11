from utils.training import train_and_evaluate

# モデル名
# MODEL_NAMES = ["resnet18", "efficientnet_b0", "mobilenetv3_small_050"]
MODEL_NAMES = ["resnet18"]


if __name__ == "__main__":
    # 各モデルでループ
    for model_name in MODEL_NAMES:
        train_and_evaluate(model_name)
