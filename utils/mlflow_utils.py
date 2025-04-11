import mlflow
import os


def init_mlflow(model_name):
    mlflow.set_tracking_uri("file://" + os.path.abspath("logs/mlruns"))
    mlflow.set_experiment("MNIST with TIMM")
    mlflow.start_run(run_name=model_name)
    mlflow.log_param("model", model_name)


def log_metrics_mlflow(acc, loss_list):
    mlflow.log_metric("final_loss", loss_list[-1])
    mlflow.log_metric("final_accuracy", acc)
