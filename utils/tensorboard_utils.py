from torch.utils.tensorboard import SummaryWriter
import os


def init_tensorboard(model_name):
    log_dir = os.path.join("logs/tensorboard", model_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    return writer
