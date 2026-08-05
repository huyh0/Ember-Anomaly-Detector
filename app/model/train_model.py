import torch
import numpy as np
import pandas as pd
import lightning as L
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy, BinaryAUROC
from sklearn.preprocessing import StandardScaler
from joblib import dump
from app.env_settings import Settings


settings = Settings()

data = pd.read_parquet(settings.data_path)

train_val_split = int(0.8*len(data))

X_train = data.iloc[:train_val_split, :-1].values
y_train = data.iloc[:train_val_split, -1].values

X_val = data.iloc[train_val_split:, :-1].values
y_val = data.iloc[train_val_split:, -1].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
dump(scaler, 'app/model/scaler.bin', compress=True)
X_val_scaled = scaler.transform(X_val)


def dataloader(X, y):
    tensor_features = torch.tensor(X, dtype=torch.float32)
    tensor_labels = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(tensor_features, tensor_labels)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )

    return loader

class MLP(L.LightningModule):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 2),
        )

        self.train_accuracy = MulticlassAccuracy(num_classes=2)
        self.train_auroc = BinaryAUROC()

        self.val_accuracy = MulticlassAccuracy(num_classes=2)
        self.val_auroc = BinaryAUROC()

        self.save_hyperparameters()

    def training_step(self, batch):
        x, y = batch
        logits = self.network(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        predictions = torch.argmax(logits, dim=1)
        malware_probability = torch.softmax(logits, dim=1)[:, 1]

        self.train_accuracy.update(predictions, y)
        self.train_auroc.update(malware_probability, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.network(x)
        loss = nn.functional.cross_entropy(logits, y)
        
        predictions = torch.argmax(logits, dim=1)
        malware_probability = torch.softmax(logits, dim=1)[:, 1]

        self.val_accuracy.update(predictions, y)
        self.val_auroc.update(malware_probability, y)


        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def forward(self, x):
        logits = self.network(x)
        prediction = torch.argmax(logits, dim=1)
        malware_probability = torch.softmax(logits, dim=1)[:, 1]

        return malware_probability

if __name__ == "__main__":
    
    train_dataloader = dataloader(X_train_scaled, y_train)
    val_dataloader = dataloader(X_val_scaled, y_val)

    train_features, _ = next(iter(train_dataloader))

    model = MLP(input_dim = train_features.shape[1], hidden_dim = 512)

    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


