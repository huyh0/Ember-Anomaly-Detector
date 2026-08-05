import argparse
import torch
import numpy as np
import lightning as L
import pandas as pd
from torch import nn
from app.model.train_model import MLP
from joblib import load
from sklearn.preprocessing import StandardScaler
from app.database.postgre_connection import create_table, add_item

def load_scaler(scaler_path):
    scaler = load(scaler_path)
    return scaler

def load_model(checkpoint_path, data):
    input_dim = data.shape[1] - 1

    model = MLP.load_from_checkpoint(
        checkpoint_path,
        input_dim=input_dim,
        hidden_dim=512,
        map_location="cpu",
    )

    model.eval()

    return model


def inference(idx, T, model, scaler, data, engine):
    
    sample = data.iloc[idx, :-1].values.reshape(1, -1)
    scaled_sample = scaler.transform(sample)
    tensor_features = torch.tensor(scaled_sample, dtype=torch.float32)

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

    prediction_list = np.array([])

    for i in range(T):
        with torch.no_grad():
            prediction_list = np.append(prediction_list, model(tensor_features).item())

    if prediction_list.mean() > 0.5:
        predicted_label = "Malicious"
    else:
        predicted_label = "Benign"

    if data.iloc[idx, -1] == 0:
        true_label = "Benign"
    else:
        true_label = "Malicious"


    create_table(name="Predictions", engine=engine)
    add_item(name="Predictions",
            engine=engine, 
            ID = idx, 
            RP = f"{prediction_list.mean():.3f}",  
            U = f"{prediction_list.std():.3f}", 
            PL = predicted_label, 
            TL = true_label,
            T = T
            )

    Results_dict = {"Table Name":"Predictions", 
            "ID": idx, 
            "Raw Prediction" : f"{prediction_list.mean():.3f}",  
            "Uncertainty" : f"{prediction_list.std():.3f}", 
            "Predicted Label" : predicted_label, 
            "True Label" : true_label,
            "T": T
    }

    return Results_dict