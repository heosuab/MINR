import os
import logging

import torch

from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)
SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


class TrainerTemplate:
    def __init__(
        self,
        model,
        args,
        model_ema,
        dataset_trn,
        dataset_val_01,
        dataset_val_02,
        dataset_val_03,
        exp_name,
        val_lists,
        config,
        writer,
        device,
        distenv,
        model_aux=None,
        **kwargs,
    ):
        super().__init__()

        num_workers = 16

        if SMOKE_TEST:
            if not torch.distributed.is_initialized():
                num_workers = 0
            config.experiment.test_freq = 1
            config.experiment.save_ckpt_freq = 1

        self.model = model
        self.model_ema = model_ema
        self.model_aux = model_aux

        self.config = config
        self.writer = writer
        self.device = device
        self.distenv = distenv
        self.args = args

        self.exp_name = exp_name
        self.val_lists = val_lists

        self.dataset_trn = dataset_trn
        self.dataset_val_01 = dataset_val_01
        self.dataset_val_02 = dataset_val_02
        self.dataset_val_03 = dataset_val_03
        self.dataset_vals = [self.dataset_val_01, self.dataset_val_02, self.dataset_val_03]

        self.sampler_trn = torch.utils.data.distributed.DistributedSampler(
            self.dataset_trn,
            num_replicas=self.distenv.world_size,
            rank=self.distenv.world_rank,
            shuffle=True,
            seed=self.config.seed,
        )
        self.loader_trn = DataLoader(
            self.dataset_trn,
            sampler=self.sampler_trn,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
        )

        self.sampler_val_01 = torch.utils.data.distributed.DistributedSampler(
            self.dataset_val_01, num_replicas=self.distenv.world_size, rank=self.distenv.world_rank, shuffle=False
        )
        self.loader_val_01 = DataLoader(
            self.dataset_val_01,
            sampler=self.sampler_val_01,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
        )

        self.sampler_val_02 = torch.utils.data.distributed.DistributedSampler(
            self.dataset_val_02, num_replicas=self.distenv.world_size, rank=self.distenv.world_rank, shuffle=False
        )
        self.loader_val_02 = DataLoader(
            self.dataset_val_02,
            sampler=self.sampler_val_02,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
        )

        self.sampler_val_03 = torch.utils.data.distributed.DistributedSampler(
            self.dataset_val_03, num_replicas=self.distenv.world_size, rank=self.distenv.world_rank, shuffle=False
        )
        self.loader_val_03 = DataLoader(
            self.dataset_val_03,
            sampler=self.sampler_val_03,
            shuffle=False,
            pin_memory=True,
            batch_size=config.experiment.batch_size,
            num_workers=num_workers,
        )
        self.sampler_vals = [self.sampler_val_01, self.sampler_val_02, self.sampler_val_03]
        self.loader_vals = [self.loader_val_01, self.loader_val_02, self.loader_val_03]

        self._scaler = None

    def train(self, optimizer=None, scheduler=None, scaler=None, epoch=0):
        raise NotImplementedError

    def eval(self, valid=True, ema=False, verbose=False, epoch=0):
        raise NotImplementedError

    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = GradScaler(enabled=self.config.experiment.amp)
        return self._scaler

    def run_epoch(self, optimizer=None, scheduler=None, epoch_st=0, device='cuda'):
        scaler = self.scaler

        for i in range(epoch_st, self.config.experiment.epochs):
            self.sampler_trn.set_epoch(i)
            torch.cuda.empty_cache()
            summary_trn = self.train(optimizer, scheduler, scaler, epoch=i, device=device)
            for val_num in range(0, 3):
                if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                    torch.cuda.empty_cache()
                    summary_val = self.eval(val_num=val_num, 
                                            val_name=self.val_lists[val_num], 
                                            epoch=i,
                                            device=device,
                                            args=self.args)
                    if self.model_ema is not None:
                        summary_val_ema = self.eval(val_num=val_num, 
                                                    val_name=self.val_lists[val_num], 
                                                    ema=True, 
                                                    epoch=i,
                                                    device=device,
                                                    args=self.args)

            if self.distenv.master:
                self.logging(summary_trn, scheduler=scheduler, epoch=i + 1, mode="train")

                if i == 0 or (i + 1) % self.config.experiment.test_freq == 0:
                    self.logging(summary_val, scheduler=scheduler, epoch=i + 1, mode="valid")
                    if self.model_ema is not None:
                        self.logging(summary_val_ema, scheduler=scheduler, epoch=i + 1, mode="valid_ema")

                if (i + 1) % self.config.experiment.save_ckpt_freq == 0:
                    self.save_ckpt(optimizer, scheduler, i + 1)

    def save_ckpt(self, optimizer, scheduler, epoch):
        ckpt_path = os.path.join(self.config.result_path, "epoch%d_model.pt" % epoch)
        num_ckpt = len([name for name in os.listdir(self.config.result_path) if name.endswith(".pt")])
        if num_ckpt >= 3:
            ckpt_list = sorted([name for name in os.listdir(self.config.result_path) if name.endswith(".pt")])
            os.remove(os.path.join(self.config.result_path, ckpt_list[0]))
        logger.info("epoch: %d, saving %s", epoch, ckpt_path)
        ckpt = {
            "epoch": epoch,
            "state_dict": self.model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if self.model_ema is not None:
            ckpt.update(state_dict_ema=self.model_ema.module.module.state_dict())
        torch.save(ckpt, ckpt_path)
