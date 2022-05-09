from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch import nn
from torch.optim import AdamW

from idp_dataset import get_sequence_loader, DatasetMode


class HopfieldNetwork(pl.LightningModule):

    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        ...
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("validation_loss", test_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        ...
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)


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
def main():
    wandb_logger = WandbLogger(project="IDP-Predictor", name="hopfield_network", log_model="all")
    #HopfieldNetwork.load_from_checkpoint('checkpoints/hopfield_network.ckpt')
    trainer = Trainer(limit_train_batches=100, max_epochs=1, logger=wandb_logger, accelerator="auto",
                      default_root_dir='checkpoints/hopfield_network', enable_checkpointing=True,
                      callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=5, min_delta=0.001)],
                      auto_lr_find=True)
    model = HopfieldNetwork()
    trainer.tune(model)
    trainer.fit(model, train_dataloaders=get_sequence_loader(DatasetMode.TRAIN),
                val_dataloaders=get_sequence_loader(DatasetMode.VALIDATION))
    trainer.test(model, dataloaders=get_sequence_loader(DatasetMode.TEST))


if __name__ == "__main__":
    main()
