import math

import torch
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer

import wandb
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW, Adam
from torchmetrics.functional import accuracy

from idp_dataset import get_sequence_loader, DatasetMode


class IDPHopfieldNetwork(pl.LightningModule):

    def __init__(self,
                 lr=0.001,
                 weight_decay=0.001,
                 dropout=0.1,
                 activation="selu",
                 optimizer="adamw",
                 window_size=257,
                 linear_first=True,
                 hopfield_layers=1,
                 linear_layers=1,
                 embed_dim=1024):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.optimizer = optimizer
        self.window_size = window_size

        layers = []

        if linear_first:
            layers.append(nn.Flatten())
            layers.append(nn.Linear(self.window_size*embed_dim, self.window_size*embed_dim))
            layers.append(nn.SELU())
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Unflatten(1, (self.window_size, embed_dim)))

        for i in range(hopfield_layers):
            hopfield = Hopfield(
                input_size=embed_dim,
                association_activation=activation,
                dropout=dropout
            )
            layers.append(hopfield)

            for i in range(linear_layers):
                layers.append(nn.Linear(self.window_size*hopfield.output_size, self.window_size*hopfield.output_size))
                layers.append(nn.SELU())
                layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Unflatten(1, (self.window_size, hopfield.output_size)))

        # Projection
        layers.append(nn.Flatten())
        layers.append(nn.Linear(hopfield.output_size * self.window_size, self.window_size))
        layers.append(nn.Sigmoid())

        self.hopfield_net = nn.Sequential(*layers)

        self.loss = MSELoss()

        self.save_hyperparameters()

    def forward(self, x):
        mapping = self.hopfield_net(x)
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
        return loss

    def test_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_and_accuracy(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return loss

    def _get_preds_loss_and_accuracy(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = accuracy(y_hat.round(), y.to(torch.int32))
        return y_hat, loss, acc

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


def run_model(lr,
              weight_decay,
              dropout,
              optimizer,
              activation,
              window_size,
              masking_prob,
              random_offset_size,
              linear_first,
              hopfield_layers,
              linear_layers,
              accelerator="gpu",
              wandb_enabled=True):
    kwargs = dict()
    if not wandb_enabled:
        kwargs["mode"] = "disabled"
    train_loader = get_sequence_loader(DatasetMode.TRAIN, window_size=window_size,
                                       masking_prob=masking_prob, random_offset_size=random_offset_size)
    val_loader = get_sequence_loader(DatasetMode.VALIDATION, window_size=window_size)
    wandb_logger = WandbLogger(project="IDP-Predictor", name="hopfield_network", log_model="all", **kwargs)
    # MLPNetwork.load_from_checkpoint('checkpoints/mlp_network.ckpt')
    # limit_train_batches=100, max_epochs=1,
    trainer = Trainer(logger=wandb_logger, accelerator=accelerator,
                      max_epochs=5,
                      default_root_dir='checkpoints/hopfield_network', gpus=0 if accelerator == "cpu" else 1,
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001),
                                 StochasticWeightAveraging()],
                      auto_lr_find=False, gradient_clip_val=1.0)

    # Found 0.001 as the best learning rate
    # print("Tuning...")
    # lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # model.hparams.lr = lr_finder.suggestion()
    # print(f'Auto-found model LR: {model.hparams.lr}')

    model = IDPHopfieldNetwork(
        lr=lr,
        weight_decay=weight_decay,
        dropout=dropout,
        activation=activation,
        optimizer=optimizer,
        window_size=window_size,
        linear_first=linear_first,
        hopfield_layers=hopfield_layers,
        linear_layers=linear_layers
    )
    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model, dataloaders=get_sequence_loader(DatasetMode.TEST))


def sweep_iteration():
    wandb.init()
    run_model(
        lr=wandb.config.lr,
        weight_decay=wandb.config.weight_decay,
        window_size=wandb.config.window_size,
        dropout=wandb.config.dropout,
        activation=wandb.config.activation,
        optimizer=wandb.config.optimizer,
        masking_prob=wandb.config.masking_prob,
        random_offset_size=wandb.config.random_offset_size,
        linear_first=wandb.config.linear_first,
        hopfield_layers=wandb.config.hopfield_layers,
        linear_layers=wandb.config.linear_layers
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
            "window_size": {
                # Choose from pre-defined values
                "values": [16, 32, 64, 128, 256, 512]
            },
            "dropout": {
                # Choose from pre-defined values
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "masking_prob": {
                # Choose from pre-defined values
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "random_offset_size": {
                # Choose from pre-defined values
                "values": [0.0, 0.1, 0.25, 0.5]
            },
            "activation": {
                # Choose from pre-defined values
                "values": ["", "relu", "tanh", "selu"]
            },
            "linear_first": {
                # Choose from pre-defined values
                "values": [True, False]
            },
            "hopfield_layers": {
                # Choose from pre-defined values
                "values": [1, 2, 3, 4]
            },
            "linear_layers": {
                # Choose from pre-defined values
                "values": [0, 1, 2, 3, 4]
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
        wandb.agent(sweep_id, function=sweep_iteration, count=25, project="IDP-Predictor")
    else:
        run_model(lr=0.001, weight_decay=0.001, dropout=0.0, activation="relu", optimizer="adamw", window_size=256,
                  random_offset_size=0.0, masking_prob=0.0, linear_first=False, hopfield_layers=1, linear_layers=1,
                  accelerator="cpu", wandb_enabled=False)


if __name__ == "__main__":
    main()
