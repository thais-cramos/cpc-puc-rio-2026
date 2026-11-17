"""Configuração comum do MLflow da fase 3: mesmo banco (mlflow_cpc.db) e mesma pasta de artefatos (mlartifacts) do notebook 09.
Para ver: mlflow ui --backend-store-uri sqlite:///mlflow_cpc.db  (na pasta do projeto)."""
import os
from pathlib import Path
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow

EXPERIMENTO_FASE3 = "CPC2023_fase3_kNN_RandomForest"


def iniciar(experimento=EXPERIMENTO_FASE3):
    mlflow.set_tracking_uri("sqlite:///" + Path("mlflow_cpc.db").resolve().as_posix())
    if mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(experimento, artifact_location=Path("mlartifacts").resolve().as_uri())
    mlflow.set_experiment(experimento)
    return mlflow
