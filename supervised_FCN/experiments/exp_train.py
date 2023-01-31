import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score

from supervised_FCN.models.fcn import FCNBaseline
from supervised_FCN.experiments.exp_base import *
from supervised_FCN.utils import *


class ExpFCN(ExpBase):
    def __init__(self,
                 config: dict,
                 n_train_samples: int,
                 n_classes: int,
                 ):
        super().__init__()
        self.config = config
        self.T_max = config['trainer_params']['max_epochs'] * (np.ceil(n_train_samples / config['dataset']['batch_size']) + 1)
        in_channels = config['dataset']['in_channels']

        self.fcn = FCNBaseline(in_channels, n_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.fcn.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ], weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.squeeze()

        yhat = self.fcn(x)  # (b n_classes)
        loss = self.criterion(yhat, y)

        # log
        acc = accuracy_score(y.flatten().detach().cpu().numpy(),
                             yhat.argmax(dim=-1).flatten().cpu().detach().numpy())
        loss_hist = {'loss': loss, 'acc': acc}

        detach_the_unnecessary(loss_hist)
        return loss_hist
