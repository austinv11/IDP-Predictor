import math

import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam
from torchmetrics.functional import accuracy

from idp_dataset import get_sequence_loader, DatasetMode


class MLPNetwork(pl.LightningModule):

    def __init__(self, lr=0.001, embed_dim=1024, weight_decay=0.001, dropout=0.1, layers=4, optimizer="adamw"):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.layers = layers
        self.dropout = dropout
        self.optimizer = optimizer

        layer_defs = [nn.Linear(embed_dim, embed_dim*2), nn.SELU(), nn.Dropout(dropout)]

        for i in range(1, layers):
            layer_defs.append(nn.Linear(embed_dim*2 // i, embed_dim*2 // (i+1)))
            layer_defs.append(nn.SELU())
            layer_defs.append(nn.Dropout(dropout))

        layer_defs.append(nn.Linear(embed_dim*2 // layers, 3))
        layer_defs.append(nn.Sigmoid())

        self.net = nn.Sequential(*layer_defs)
        self.loss = CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x):
        mapping = None
        for i in range(x.shape[1]):
            res = self.net(x[:, i, :].squeeze(1)).unsqueeze(1)
            if mapping is None:
                mapping = res
            else:
                mapping = torch.cat((mapping, res), dim=1)
        return mapping

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.define_metric('train_accuracy', summary='max')

        preds, loss, acc = self._get_preds_loss_and_accuracy(batch)

        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
            wandb.define_metric('val_accuracy', summary='max')

        preds, loss, acc = self._get_preds_loss_and_accuracy(batch)

        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        return preds

    def test_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_and_accuracy(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return loss

    def _get_preds_loss_and_accuracy(self, batch):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1)
        y_hat = y_hat.view(-1, 3)
        preds = torch.argmax(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        acc = accuracy(preds, y)
        return preds, loss, acc

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd_momentum":
            return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer")


"""
Install tpu support 
!pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
!pip install pytorch-lightning

import wandb
wandb.login()

After training:
import torch_xla.debug.metrics as met

print(met.metrics_report())
"""

train_loader = get_sequence_loader(DatasetMode.TRAIN)
val_loader = get_sequence_loader(DatasetMode.VALIDATION)


def run_model(lr, weight_decay, layers, dropout, optimizer, accelerator="gpu"):
    wandb_logger = WandbLogger(project="IDP-Predictor", name="mlp_network", log_model="all")
    # MLPNetwork.load_from_checkpoint('checkpoints/mlp_network.ckpt')
    # limit_train_batches=100, max_epochs=1,
    trainer = Trainer(logger=wandb_logger, accelerator=accelerator,
                      max_epochs=5,
                      default_root_dir='checkpoints/mlp_network', gpus=1,
                      callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=5, min_delta=0.001)],
                      auto_lr_find=False)

    # Found 0.001 as the best learning rate
    # print("Tuning...")
    # lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # model.hparams.lr = lr_finder.suggestion()
    # print(f'Auto-found model LR: {model.hparams.lr}')

    model = MLPNetwork(
        lr=lr,
        weight_decay=weight_decay,
        layers=layers,
        dropout=dropout,
        optimizer=optimizer
    )
    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model, dataloaders=get_sequence_loader(DatasetMode.TEST))


def sweep_iteration():
    wandb.init()
    run_model(
        lr=wandb.config.lr,
        weight_decay=wandb.config.weight_decay,
        layers=wandb.config.layers,
        dropout=wandb.config.dropout,
        optimizer=wandb.config.optimizer
    )


def main():
    # Sweep
    sweep_config = {
        "method": "random",  # "bayes",
        "metric": {  # We want to minimize val_loss
            "name": "val_loss",
            "goal": "minimize"
        },
        "parameters": {
            "layers": {
                # Choose from pre-defined values
                "values": [2, 4, 6, 8]
            },
            "dropout": {
                # Choose from pre-defined values
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "lr": {
                "distribution": "log_uniform",
                "min": math.log(1e-8),
                "max": math.log(1e-1)
            },
            "weight_decay": {
                "distribution": "log_uniform",
                "min": math.log(1e-8),
                "max": math.log(1e-1)
            },
            "optimizer": {
                # Choose from pre-defined values
                "values": ["adam", "adamw", "sgd", "sgd_momentum"]
            }
        }
    }
    SWEEP = False
    if SWEEP:
        # Run once
        sweep_id = wandb.sweep(sweep_config, project="IDP-Predictor")
        print("Sweep ID:", sweep_id)
        wandb.agent(sweep_id, function=sweep_iteration, count=25)
    else:
        run_model(lr=0.001, weight_decay=0.001, layers=2, dropout=0.0, optimizer="adam", accelerator="cpu")


if __name__ == "__main__":
    main()
