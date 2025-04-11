import timm

from utils.training import train_and_evaluate

# モデル名
MODEL_NAMES = ["resnet18", "efficientnet_b0", "mobilenetv3_small_050"]


if __name__ == "__main__":
    # 各モデルでループ
    for model_name in MODEL_NAMES:
        model = timm.create_model(model_name, pretrained=True, num_classes=10)
        train_and_evaluate(model, model_name)
