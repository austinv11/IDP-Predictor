import math

from torch.optim.swa_utils import SWALR
import torch
try:
    torch.gelu = torch.nn.functional.gelu
except AttributeError:
    pass
from hflayers import Hopfield, HopfieldLayer
from hflayers.transformer import HopfieldEncoderLayer
from positional_encodings import PositionalEncoding1D, Summer
import wandb
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torchmetrics.functional import accuracy, auroc

from idp_dataset import get_sequence_loader, DatasetMode


class IDPHopfieldNetwork(pl.LightningModule):

    def __init__(self,
                 lr=0.001,
                 weight_decay=0.001,
                 dropout=0.35,
                 optimizer="sgd",
                 window_size=16,
                 hopfield_layers=4,
                 hopfield_type="encoder",
                 linear_layers=0,
                 dimension_reduction_factor=1.0,
                 embed_dim=1024,
                 n_heads=4,
                 connect_pattern_projection=False,
                 norm_hopfield_space=False,
                 swalr_lr=0.01,
                 swalr_anneal='cos',
                 swalr_epochs=3):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.optimizer = optimizer
        self.n_heads = n_heads
        self.swalr_lr = swalr_lr
        self.swalr_anneal = swalr_anneal
        self.swalr_epochs = swalr_epochs
        self.window_size = window_size+1  # +1 for start token
        self.embed_dim = int(embed_dim * dimension_reduction_factor)+1  # +1 for start token
        self.embed_dim = self.embed_dim - (self.embed_dim % self.n_heads)

        activation_fn = nn.GELU()

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
                    association_activation="gelu",
                    dropout=dropout,
                    num_heads=self.n_heads,
                    normalize_hopfield_space=norm_hopfield_space,
                    pattern_projection_as_connected=connect_pattern_projection
                )
            else:
                hopfield = HopfieldEncoderLayer(
                    hopfield_association=Hopfield(
                        input_size=self.embed_dim,
                        association_activation="gelu",
                        dropout=dropout,
                        num_heads=self.n_heads,
                        normalize_hopfield_space=norm_hopfield_space,
                        pattern_projection_as_connected=connect_pattern_projection
                    ),
                    dim_feedforward=self.embed_dim*2,
                    dropout=dropout,
                    activation="gelu"
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
        if self.trainer.global_step == 0:
           wandb.define_metric('train_aucroc', summary='mean', goal="maximize")

        preds, loss, acc, auc = self._get_preds_loss_and_accuracy(batch)

        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        self.log('train_aucroc', auc)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
           wandb.define_metric('val_aucroc', summary='mean', goal="maximize")

        preds, loss, acc, auc = self._get_preds_loss_and_accuracy(batch)

        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        self.log('val_aucroc', auc)
        return loss

    def test_step(self, batch, batch_idx):
        if self.trainer.global_step == 0:
           wandb.define_metric('test_aucroc', summary='mean', goal="maximize")

        preds, loss, acc, auc = self._get_preds_loss_and_accuracy(batch)

        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        self.log('val_aucroc', auc)
        return loss

    def _get_preds_loss_and_accuracy(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        acc = accuracy(y_hat.round(), y.to(torch.int32))
        auc = auroc(y_hat.round(), y.to(torch.int32))
        return y_hat, loss, acc, auc

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "adamax":
            optimizer = torch.optim.Adamax(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError("Unknown optimizer")
        scheduler = SWALR(optimizer, swa_lr=self.swalr_lr, anneal_strategy=self.swalr_anneal, anneal_epochs=self.swalr_epochs)
        return [optimizer], [scheduler]


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
              random_offset_size=0.25,
              linear_layers=0,
              optimizer="sgd",
              window_size=16,
              dropout=0.35,
              hopfield_layers=4,
              n_heads=4,
              hopfield_type="encoder",
              dimension_reduction_factor=1.0,
              connect_pattern_projection=False,
              norm_hopfield_space=False,
              accelerator="gpu",
              wandb_enabled=True):
    kwargs = dict()
    if not wandb_enabled:
        kwargs["mode"] = "disabled"
    train_loader = get_sequence_loader(DatasetMode.TRAIN, window_size=window_size,
                                       masking_prob=0.2, random_offset_size=random_offset_size)
    val_loader = get_sequence_loader(DatasetMode.VALIDATION, window_size=window_size)
    wandb_logger = WandbLogger(project="IDP-Predictor", name="hopfield_network", log_model="all", **kwargs)
    # MLPNetwork.load_from_checkpoint('checkpoints/mlp_network.ckpt')
    # limit_train_batches=100, max_epochs=1,
    checkpoint_callback = ModelCheckpoint(
                                     dirpath='checkpoints/hopfield_network',
                                     monitor='val_loss',
                                     verbose=True,
                                     save_last=True,
                                     save_top_k=3,
                                     mode='min',
                                     auto_insert_metric_name=True)
    trainer = Trainer(logger=wandb_logger, accelerator=accelerator,
                      max_epochs=3, enable_checkpointing=True,
                      default_root_dir='checkpoints/hopfield_network', devices=None if accelerator == "cpu" else 1,
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=0.001),
                                 StochasticWeightAveraging(),
                                 LearningRateMonitor(logging_interval='step'),
                                 checkpoint_callback],
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
        optimizer=optimizer,
        window_size=window_size,
        hopfield_layers=hopfield_layers,
        hopfield_type=hopfield_type,
        linear_layers=linear_layers,
        dimension_reduction_factor=dimension_reduction_factor,
        n_heads=n_heads,
        connect_pattern_projection=connect_pattern_projection,
        norm_hopfield_space=norm_hopfield_space
    )
    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("=====TRAINING COMPLETED=====")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best model val_loss: {checkpoint_callback.best_model_score}")

    print("=====TESTING=====")
    trainer.test(ckpt_path="best", dataloaders=get_sequence_loader(DatasetMode.TEST))


def sweep_iteration():
    wandb.init()
    run_model(
        lr=wandb.config.lr,
        weight_decay=wandb.config.weight_decay,
        norm_hopfield_space=(wandb.config.norm_hopfield_space == 1),
        connect_pattern_projection=(wandb.config.connect_pattern_projection == 1)
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
            "lr": {
                "distribution": "log_uniform",
                "min": math.log(5e-5),
                "max": math.log(1e-1)
            },
            "weight_decay": {
                "distribution": "log_uniform",
                "min": math.log(1e-8),
                "max": math.log(1e-5)
            },
            "norm_hopfield_space": {
                # Choose from pre-defined values
                "values": [1, 0]
            },
            "connect_pattern_projection": {
                # Choose from pre-defined values
                "values": [1, 0]
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
        run_model(lr=0.0001, weight_decay=0.00001, dropout=0.0, optimizer="sgd", window_size=16,
                  random_offset_size=0.0, hopfield_layers=1, linear_layers=0, hopfield_type="encoder",
                  accelerator="cpu", dimension_reduction_factor=0.25, n_heads=1, wandb_enabled=False)


if __name__ == "__main__":
    main()
