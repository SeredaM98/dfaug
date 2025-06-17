import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy
import torch
from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI
from plotly.subplots import make_subplots
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import eval.eval as ecb
from dataloader import loader as data_base
from diff_model.diffusion import GaussianDiffusion, make_beta_schedule
from diff_model.model import AE, UNet
from util import Layout, LineBackground, OuterRing, ScatterVis, pad_table_data
from ResNet import ResNetMnist
import wandb
import torchvision.transforms as transforms
torch.set_num_threads(1)

def gpu2np(a):
    return a.cpu().detach().numpy()


class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        m_l.append(nn.BatchNorm1d(out_dim))

        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)


class DMTEVT_Encoder(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Encoder, self).__init__()
        self.data_name = data_name
        self.l_token = l_token
        self.l_token2 = l_token2
        (self.model_token,) = self.InitNetworkMLP(
            l_token_2=l_token2,
            num_input_dim=num_input_dim,
        )

    def forward(self, x):
        # def forward_fea(self, x, batch_idx):
        lat_high_dim = self.model_token(x)

        return lat_high_dim  # , lat_vis

    def InitNetworkMLP(
        self,
        l_token_2=100,
        num_input_dim=64,
    ):
        # self.hparams.l_token_2 = 20
        m_p = []
        m_p.append(NN_FCBNRL_MM(num_input_dim, 500))
        m_p.append(NN_FCBNRL_MM(500, 500))
        m_p.append(NN_FCBNRL_MM(500, 500))
        m_p.append(NN_FCBNRL_MM(500, l_token_2))
        model_token = nn.Sequential(*m_p)
        return (
            model_token,
        )


class DMTEVT_Vis(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Vis, self).__init__()
        self.model_down = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, lat_high_dim):
        lat_vis = self.model_down(lat_high_dim)
        return lat_vis

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):

        m_b = []
        m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
        m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
        model_down = nn.Sequential(*m_b)

        return model_down


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def progressive_samples_fn_simple(
    model, diffusion, shape, device, cond, include_x0_pred_freq=50
):
    samples, history = diffusion.p_sample_loop_progressive_simple(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        cond=cond,
    )
    return {"samples": samples}


class ContrastiveDiff(LightningModule):
    def __init__(
        self,
        lr=0.001,
        nu=0.01,
        data_name="mnist",
        max_epochs=1000,
        class_num=30,
        steps=20000,
        num_pat=8,
        num_fea_aim=500,
        n_timestep=100,
        l_token=50,
        l_token_2=50,
        num_layers_Transformer=1,
        num_latent_dim=2,
        num_input_dim=64,
        laten_down=500,
        weight_decay=0.0001,
        kmean_scale=0.01,
        loss_rec_weight=0.01,
        preprocess_epoch=100,
        transformer2_indim=3750,
        marker_size=2,
        joint_epoch=1400,
        rand_rate=0.9,
        E_epochs = 25,
        D_epochs = 50,
        **kwargs,
    ):
        super().__init__()
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.dictinputdict = {}
        self.t = 0.1
        self.alpha = 0.1
        self.stop = False
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.wandb_logs = {}
        self.mse = torch.nn.MSELoss()  # torch.nn.CrossEntropyLoss()
        self.center_change = False
        self.train_state = "train"
        self.base_acc = 0
        self.cnt = 0
        self.update_aug_bank = False
        self.cnt_for_switch = 0

        self.validation_step_outputs_data = []
        self.validation_step_outputs_lat = []
        self.validation_step_outputs_lat_high = []
        self.validation_step_outputs_label = []

        self.enc = ResNetMnist(self.hparams.l_token_2) #DMTEVT_Encoder(
            #l_token=self.hparams.l_token,
            #l_token2=self.hparams.l_token_2,
            #data_name=self.hparams.data_name,
            #transformer2_indim=self.hparams.transformer2_indim,
            #laten_down=self.hparams.laten_down,
            #num_layers_Transformer=self.hparams.num_layers_Transformer,
            #num_input_dim=self.hparams.num_input_dim,
        #)

        self.vis = DMTEVT_Vis(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )
        self.data_aug = torch.zeros(48000, self.hparams.num_input_dim)

        self.UNet_model = AE(in_dim=self.hparams.num_input_dim, mid_dim=2000)
        self.UNet_ema = AE(in_dim=self.hparams.num_input_dim, mid_dim=2000)

        n_timestep = self.hparams.n_timestep
        self.betas = make_beta_schedule(
            schedule="linear", start=1e-4, end=2e-2, n_timestep=n_timestep
        )
        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type="eps",
            model_var_type="fixedlarge",
            loss_type="mse",
        )
        self.start_from_ckpt = True

    def forward(self, x, batch_idx):
        lat_high_dim = self.enc(x)
        lat_vis = self.vis(lat_high_dim)
        return lat_high_dim, lat_vis

    def _DistanceSquared(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)

        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        # losssum = P_ * torch.log(P_) - P_ * torch(Q_) #KL divergence?
        return losssum.mean()

    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (                                  # Откуда тут первые два множителя?
        gamma
        * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1)/2)
        )

        return Pij

    def LossManifold(
        self,
        v_input,
        input_data,
        latent_data,
        v_latent,

    ):

        batch_size = input_data.shape[0] // 2
        data_1 = input_data[:batch_size] # фичи оригиналов
        dis_P_2 = self._DistanceSquared(data_1, data_1) # попарные расстояния между фичами оригиналов
        latent_data_1 = latent_data[:batch_size] # двумерные фичи оригиналов
        gamma = self._CalGamma(v_input)
        P_2 = self._Similarity(dist=dis_P_2, gamma=gamma, v=v_input)
        latent_data_2 = latent_data[batch_size:] # двумерные представления фич аугментаций
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        loss_ce_2 = self._TwowaydivergenceLoss(P_=P_2, Q_=Q_2)

        return loss_ce_2

    def augment_data_simple(self, cond_input_val):
        shape = (cond_input_val.shape[0], 1, self.hparams.num_input_dim)
        self.UNet_ema.eval()
        sample = progressive_samples_fn_simple(
            self.UNet_ema,
            self.diffusion,
            shape,
            device="cuda",
            cond=cond_input_val,
            include_x0_pred_freq=50,
        )
        return sample["samples"]



    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        self.n_token_feature = checkpoint["n_token_feature"]
        self.enc.n_token_feature = checkpoint["n_token_feature"]


    def diffusion_loss(self, data_after_tokened, lat_high_dim):
        data_diff = data_after_tokened
        views = data_diff.reshape(data_diff.shape[0], -1)
        time = (
            (torch.rand(data_diff.shape[0]) * self.hparams.n_timestep)
            .type(torch.int64)
            .to(data_diff.device)
        )
        loss_diff = self.diffusion.training_losses(
            model=self.UNet_model,
            x_0=views,
            t=time,
            lab=lat_high_dim.detach(),
        ).mean()
        return loss_diff * 0.01

    def EM_switch(self):

        if self.current_epoch < self.hparams.preprocess_epoch:
            return "s1"
        if self.current_epoch < self.hparams.joint_epoch:
            if self.current_epoch == self.hparams.joint_epoch - 1:
                self.update_aug_bank = True
                self.cnt_for_switch = self.current_epoch
            return "s2"
        if self.current_epoch - self.cnt_for_switch <= self.hparams.E_epochs:
            return "e"
        else:
            if self.current_epoch - self.cnt_for_switch == self.hparams.E_epochs + self.hparams.D_epochs:
                self.update_aug_bank = True
                self.cnt_for_switch = self.current_epoch
            return "m"

    def augment_mnist_batch(self, batch_images):
        """
        Аугментация батча изображений MNIST.

        Параметры:
        - batch_images: тензор батча изображений MNIST (размерность: [batch_size, 1, 28, 28])

        Возвращает:
        - augmented_images: аугментированный тензор батча изображений
        """
        batch_images = batch_images.reshape(-1, 1, 28, 28)
        # Список преобразований для аугментации
        augmentations = transforms.Compose([
            transforms.RandomRotation(10),  # Случайный поворот на угол до 10 градусов
            transforms.RandomAffine(0, translate=(0.1, 0.1),fill=0),  # Случайный сдвиг изображения
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Случайное изменение яркости и контраста
        ])

        # Применение аугментаций к каждому изображению в батче
        augmented_images = torch.stack([augmentations(image) for image in batch_images])

        return augmented_images.reshape(-1, 28 * 28)

    def training_step(self, batch, batch_idx):
        (
            (data_at, data_aat, data_rec),
            label,
            index,
            n_token_feature,
        ) = batch
        batch_size = index.shape[0]
        self.em_state = self.EM_switch()
        self.n_token_feature = n_token_feature
        self.enc.n_token_feature = n_token_feature
        #if self.current_epoch >= self.hparams.preprocess_epoch:
            #cond, _ = self(data_at, batch_idx)
            #augs = self.augment_data_simple(cond).reshape(batch_size, -1)
            #rand_bool = torch.randn(augs.shape[0]).to(self.device) > self.hparams.rand_rate
            #data_aat[rand_bool] = augs[rand_bool]
        if self.current_epoch > self.hparams.joint_epoch+1:
            new_aug = self.data_aug[index.cpu()].to(self.device)
            rand_bool = torch.randn(new_aug.shape[0]).to(self.device) > self.hparams.rand_rate
            data_aat[rand_bool] = new_aug[rand_bool]

        data = torch.cat([data_at, data_aat])
        # if self.current_epoch > self.hparams.joint_epoch + 1:
        if self.current_epoch == self.cnt:
            self.cnt = self.cnt + 1
            print(self.em_state)
            plt.clf()
            for i in range(8):
                plt.subplot(2,4,i+1)
                plt.axis('off')
                if i < 4:
                    plt.imshow(data_at[i].cpu().detach().numpy().reshape(28, 28))
                else:
                    plt.imshow(data_aat[i-4].cpu().detach().numpy().reshape(28, 28))
            plt.savefig(f'data_trainexpl_epoch_{self.current_epoch}.png')
            plt.clf()

        if self.em_state == "s1" or self.em_state == "e":
            
            self.enc.train()
            self.vis.train()
            lat_high_dim, lat_vis = self(data, batch_idx)
            loss_topo = self.LossManifold(
                v_input=100,
                input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
                latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
                v_latent=self.hparams.nu,
            )
            loss_diff = torch.tensor(0)
        elif self.em_state == "s2" or self.em_state == "m":
            self.enc.eval()
            self.vis.eval()
            cond, _ = self(data, batch_idx)
            loss_topo = torch.tensor(0)
            loss_diff = self.diffusion_loss(data_at, cond[:batch_size])

        if self.em_state == "s1" or self.em_state == "e":
            self.wandb_logs = {
                "loss_topo": loss_topo.item(),
                "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
                "epoch": float(self.current_epoch),
                # ""
            }
        else:
            self.wandb_logs = {
                "loss_diff": loss_diff.item(),
                "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
                "epoch": float(self.current_epoch),
                # ""
            }
        self.log_dict(self.wandb_logs, sync_dist=True)

        accumulate(
            self.UNet_ema,
            self.UNet_model.module
            if isinstance(self.UNet_model, nn.DataParallel)
            else self.UNet_model,
            0.9999,
        )
        return loss_topo + loss_diff

    def validation_step(self, batch, batch_idx, test=False):
        if test:
            data_at, label, index, n_token_feature = batch
        else:
            (
                (data_at, data_aat, dat_rec),
                label,
                index,
                n_token_feature,
            ) = batch
        batch_size = index.shape[0]
        self.em_state = self.EM_switch()
        self.n_token_feature = n_token_feature
        # if self.em_state == "e" or self.em_state == "m":
        if self.update_aug_bank:
            print("\naugmented")
            self.vis.eval()
            self.enc.eval()
            #cond, _ = self(data_at, batch_idx)
            with torch.no_grad():
                #samples = self.augment_data_simple(cond[:batch_size])
                samples = self.augment_mnist_batch(data_at[:batch_size])
                self.data_aug[index] = samples.reshape(batch_size, -1).cpu()

        data = data_at
        lat_high_dim, lat_vis = self(data, batch_idx)
        if batch_idx == 0:
            self.lat_high_dim = lat_high_dim
            self.data = data_at

        self.validation_step_outputs_data.append(gpu2np(data)[:batch_size])
        self.validation_step_outputs_lat.append(gpu2np(lat_vis)[:batch_size])
        self.validation_step_outputs_lat_high.append(gpu2np(lat_high_dim)[:batch_size])
        self.validation_step_outputs_label.append(gpu2np(label))
        return data.mean()

    def on_validation_epoch_end(self):
        data = np.concatenate(self.validation_step_outputs_data)
        emb_vis = np.concatenate(self.validation_step_outputs_lat)
        emb_high = np.concatenate(self.validation_step_outputs_lat_high)
        label = np.concatenate(self.validation_step_outputs_label)
        self.update_aug_bank = False

        ecb_e_train = ecb.Eval(input=data, latent=emb_vis, label=label, k=10)
        ecb_lat_train = ecb.Eval(input=data, latent=emb_high, label=label, k=10)
        SVC_acc = ecb_e_train.E_Classifacation_SVC()
        KNN_acc = ecb_e_train.E_Clasting_Kmeans()
        SVC_high_acc = ecb_lat_train.E_Classifacation_SVC()
        KNN_high_acc = ecb_lat_train.E_Clasting_Kmeans()
        
        if self.current_epoch == self.hparams.joint_epoch-1:
            self.base_acc = SVC_high_acc
        
        if data.shape[0] <= 1000 and self.train_state == "train":
            pass
        else:
            self.wandb_logs.update(
                {
                    f"data_number_{self.train_state}": data.shape[0],
                    f"SVC_{self.train_state}": SVC_acc,
                    f"SVC_high_{self.train_state}": SVC_high_acc,
                    f"KNN_{self.train_state}": KNN_acc,
                    f"KNN_high_{self.train_state}": KNN_high_acc,
                    f"acc_delta_{self.train_state}": SVC_high_acc - self.base_acc,
                }
            )
        formatted_logs = {key: (f"{value:.4f}" if isinstance(value, float) else value) for key, value in self.wandb_logs.items()}
        sorted_keys = sorted(formatted_logs)
        for key in sorted_keys:
            print(f"{key:<15} : {formatted_logs[key]}")
        print('-------------------')
        
        self.log_dict(self.wandb_logs, sync_dist=True)
        self.wandb_logs.clear()
        
        if self.start_from_ckpt:
            self.start_from_ckpt = False
            self.log_dict({'SVC_train': 0})
        
        self.samples = self.augment_data_simple(self.lat_high_dim[:50])

        if "Mnist" in self.hparams.data_name:
            plt.clf()
            for i in range(10):
                plt.subplot(2, 5 ,i+1)
                aug_img = self.samples[i].cpu().detach().numpy().reshape(28, 28)
                plt.imshow(aug_img)
                plt.axis('off')
            plt.suptitle('Примеры аугментаций')
            plt.savefig(f'da_aug{self.current_epoch}.png')
            plt.clf()
        self.plot_scatter(
            emb_vis, label
            ).write_image("fig1.png", scale=3)
        self.logger.log_image(f"fig_{self.train_state}", ["fig1.png"])
        
        self.validation_step_outputs_lat_high.clear()
        self.validation_step_outputs_data.clear()
        self.validation_step_outputs_lat.clear()
        self.validation_step_outputs_label.clear()

    def test_step(self, batch, batch_idx):

        self.train_state = "test"
        self.validation_step(batch, batch_idx, test=True)

    def on_test_epoch_end(self):
        # Here we just reuse the validation_epoch_end for testing
        print("------------")
        print("----test----")
        print("------------")
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        parameters = [
            {"params": self.enc.parameters(), "lr": self.hparams.lr},
            {"params": self.vis.parameters(), "lr": self.hparams.lr},
            {"params": self.UNet_model.parameters(), "lr": self.hparams.lr},
        ]

        optimizer = torch.optim.AdamW(
            parameters, weight_decay=self.hparams.weight_decay
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.hparams.max_epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def plot_scatter(self, emb_vis, labels):

        emb_vis = emb_vis[:, :2]

        fig = go.Figure()
        fig = ScatterVis(fig, emb_vis, labels, size=self.hparams.marker_size)
        fig = Layout(fig)

        return fig

class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_name: str = "Digits",
        data_path: str = "/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 0,
        rrc_rate: float = 0.8,
        trans_range: int = 2,
        preprocess_bool: bool = True,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.preprocess_bool = preprocess_bool

    def setup(self, stage: str):
        dataset_f = getattr(data_base, self.data_name + "Dataset")
        self.data_train = dataset_f(
            data_name=self.data_name,
            train=True,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            n_cluster=self.n_cluster,
            n_f_per_cluster=self.n_f_per_cluster,
            l_token=self.l_token,
            seed=self.seed,
            rrc_rate=self.rrc_rate,
            trans_range=self.trans_range,
            preprocess_bool=self.preprocess_bool,
        )
        self.data_val = self.data_train
        self.data_test = dataset_f(
            data_name=self.data_name,
            train=False,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            n_cluster=self.n_cluster,
            n_f_per_cluster=self.n_f_per_cluster,
            l_token=self.l_token,
            seed=self.seed,
            rrc_rate=self.rrc_rate,
            trans_range=0,
            preprocess_bool=self.preprocess_bool,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=3,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            drop_last=False,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            drop_last=False,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.max_epochs")
        parser.link_arguments(
            "model.l_token",
            "data.l_token",
        )
        parser.link_arguments(
            "data.data_name",
            "model.data_name",
        )


def main():
    cli = MyLightningCLI(
        ContrastiveDiff,
        #CIFAR10DataModule,
        MyDataModule,
        save_config_callback=None,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.model.eval()
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
