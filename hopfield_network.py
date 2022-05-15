import math

import torch
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
from hflayers.transformer import HopfieldEncoderLayer, HopfieldDecoderLayer
from positional_encodings import PositionalEncoding1D, Summer, PositionalEncoding2D
import wandb
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Adam
from torchmetrics.functional import accuracy

from idp_dataset import get_sequence_loader, DatasetMode


class IDPHopfieldNetwork(pl.LightningModule):

    def __init__(self,
                 lr=0.001,
                 weight_decay=0.001,
                 dropout=0.1,
                 activation="relu",
                 optimizer="adamw",
                 window_size=64,
                 hopfield_layers=1,
                 hopfield_type="association",
                 linear_layers=1,
                 dimension_reduction_factor=1.0,
                 embed_dim=1024,
                 n_heads=1):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.optimizer = optimizer
        self.n_heads = n_heads
        self.window_size = window_size+1  # +1 for start token
        self.embed_dim = int(embed_dim * dimension_reduction_factor)+1  # +1 for start token
        self.embed_dim = self.embed_dim - (self.embed_dim % self.n_heads)

        activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()

        layers = list()
        # Initial convolution before hopfield network
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.window_size*(embed_dim+1), self.window_size*self.embed_dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Unflatten(1, (self.window_size, self.embed_dim)))
        # Add Positional Encoding
        layers.append(Summer(PositionalEncoding1D(self.embed_dim)))

        for i in range(hopfield_layers):
            if hopfield_type == "association":
                hopcls = Hopfield
            elif hopfield_type == "layer":
                hopcls = HopfieldLayer
            else:
                hopcls = None

            if hopcls is not None:
                hopfield = hopcls(
                    input_size=self.embed_dim,
                    association_activation=activation,
                    dropout=dropout,
                    num_heads=self.n_heads
                )
            else:
                hopfield = HopfieldEncoderLayer(
                    hopfield_association=Hopfield(
                        input_size=self.embed_dim,
                        association_activation=activation,
                        dropout=dropout,
                        num_heads=self.n_heads
                    ),
                    dim_feedforward=self.embed_dim*2,
                    dropout=dropout,
                    activation=activation
                )
            layers.append(hopfield)

            for j in range(linear_layers):
                layers.append(nn.Flatten())
                layers.append(nn.Linear(self.window_size*hopfield.output_size, self.window_size*2))
                layers.append(activation_fn)
                layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Linear(self.window_size*2, self.window_size*hopfield.output_size))
                layers.append(activation_fn)
                layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Unflatten(1, (self.window_size, hopfield.output_size)))

        # Projection
        projection = list()
        projection.append(nn.Flatten())
        projection.append(nn.Linear(hopfield.output_size * self.window_size, self.window_size))
        projection.append(nn.Sigmoid())

        self.standard_layers = nn.Sequential(*layers)
        self.projection_layers = nn.Sequential(*projection)

        self.save_hyperparameters()

    def _inject_start_token(self, x):
        # Increase the dimension to indicate start of sequence
        # Add a start token to the beginning of the sequence
        if len(x.shape) == 3:
            start_token = torch.zeros(x.shape[0], 1, x.shape[2] + 1, requires_grad=x.requires_grad, device=x.device)
            start_token[:, 0, 0] = 1
            resized = torch.cat((torch.zeros(x.shape[0], x.shape[1], 1, requires_grad=x.requires_grad, device=x.device), x), dim=2)
            return torch.cat((start_token, resized), dim=1)
        elif len(x.shape) == 2:
            return torch.cat(torch.zeros(x.shape[0], 1, requires_grad=x.requires_grad, device=x.device) + 2, dim=1)

    def _remove_start_token(self, x):
        # Reduce dim and remove start token from the output
        if len(x.shape) == 3:
            return x[:, 1:, 1:]
        elif len(x.shape) == 2:
            return x[:, 1:]

    def forward(self, x):
        mapping = self.standard_layers(self._inject_start_token(x))
        return self._remove_start_token(self.projection_layers(mapping))

    def training_step(self, batch, batch_idx):
        #if self.trainer.global_step == 0:
        #    wandb.define_metric('train_accuracy', summary='max')

        preds, loss, acc = self._get_preds_loss_and_accuracy(batch)

        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        #if self.trainer.global_step == 0:
        #    wandb.define_metric('val_accuracy', summary='max')

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
        loss = F.binary_cross_entropy(y_hat, y)
        acc = accuracy(y_hat.round(), y.to(torch.int32))
        return y_hat, loss, acc

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "adamax":
            return torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
              hopfield_layers,
              linear_layers,
              n_heads,
              hopfield_type="association",
              dimension_reduction_factor=1.0,
              gradient_clipping=1.0,
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
                      auto_lr_find=False, gradient_clip_val=gradient_clipping)

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
        hopfield_layers=hopfield_layers,
        hopfield_type=hopfield_type,
        linear_layers=linear_layers,
        dimension_reduction_factor=dimension_reduction_factor,
        n_heads=n_heads
    )
    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #trainer.test(model, dataloaders=get_sequence_loader(DatasetMode.TEST))


def sweep_iteration():
    #wandb.init()
    run_model(
        lr=wandb.config.lr,
        weight_decay=wandb.config.weight_decay,
        window_size=wandb.config.window_size,
        dropout=wandb.config.dropout,
        activation=wandb.config.activation,
        optimizer=wandb.config.optimizer,
        masking_prob=wandb.config.masking_prob,
        random_offset_size=wandb.config.random_offset_size,
        hopfield_layers=wandb.config.hopfield_layers,
        hopfield_type=wandb.config.hopfield_type,
        linear_layers=wandb.config.linear_layers,
        gradient_clipping=wandb.config.gradient_clipping,
        dimension_reduction_factor=wandb.config.dimension_reduction_factor,
        n_heads=wandb.config.n_heads
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
                "values": [8, 16, 32, 64]
            },
            "n_heads": {
                # Choose from pre-defined values
                "values": [1, 2, 4]
            },
            "dropout": {
                # Choose from pre-defined values
                "values": [0.25, 0.5]
            },
            "masking_prob": {
                # Choose from pre-defined values
                "values": [0.05, 0.15]
            },
            "random_offset_size": {
                # Choose from pre-defined values
                "values": [0.1, 0.25]
            },
            "activation": {
                # Choose from pre-defined values
                "values": ["relu", "tanh"]
            },
            "hopfield_layers": {
                # Choose from pre-defined values
                "values": [1, 2, 3]
            },
            "hopfield_type": {
                # Choose from pre-defined values
                "values": ["association", "layer", "encoder"]#, "decoder" "transformer"]
            },
            "linear_layers": {
                # Choose from pre-defined values
                "values": [0, 1, 2]
            },
            "gradient_clipping": {
                # Choose from pre-defined values
                "values": [0.5, 1.0, 2]
            },
            "dimension_reduction_factor": {
                # Choose from pre-defined values
                "values": [1, 0.75, 0.5, 0.25]
            },
            "lr": {
                "distribution": "log_uniform",
                "min": math.log(5e-5),
                "max": math.log(1e-1)
            },
            "weight_decay": {
                "distribution": "log_uniform",
                "min": math.log(1e-10),
                "max": math.log(1e-4)
            },
            "optimizer": {
                # Choose from pre-defined values
                "values": ["adamw", "sgd", "adamax"]
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
        run_model(lr=0.0001, weight_decay=0.00001, dropout=0.0, activation="relu", optimizer="adamw", window_size=16,
                  random_offset_size=0.0, masking_prob=0.0, hopfield_layers=1, linear_layers=1, hopfield_type="association",
                  accelerator="cpu", gradient_clipping=1.0, dimension_reduction_factor=0.25, n_heads=1, wandb_enabled=False)


if __name__ == "__main__":
    main()
