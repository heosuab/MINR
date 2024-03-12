import os

import torch

from .mydatasets import ImageNette, FFHQ, ImageOnlyDataset, LearnitShapenet, LibriSpeech, Celeba, Indoor
from .transforms import create_transforms

SMOKE_TEST = bool(os.environ.get("SMOKE_TEST", 0))


def create_dataset(config, args, is_eval=False, logger=None):
    transforms_trn = create_transforms(config.train_dataset, split="train", is_eval=is_eval)
    transforms_val_01 = create_transforms(config.train_dataset, split="val", is_eval=is_eval)
    transforms_val_02 = create_transforms(config.val_dataset_01, split="val", is_eval=is_eval)
    transforms_val_03 = create_transforms(config.val_dataset_02, split="val", is_eval=is_eval)

    if config.train_dataset.type == "imagenette":
        dataset_trn = ImageNette(split="train", transform=transforms_trn, masking=args.train_masking)
        dataset_val_01 = ImageNette(split="val", transform=transforms_val_01, masking=args.eval_masking)
        dataset_val_02 = Celeba(split="val", transform=transforms_val_02, masking=args.eval_masking)
        dataset_val_03 = Indoor(split="val", transform=transforms_val_03, masking=args.eval_masking)
    elif config.train_dataset.type == 'indoor':
        dataset_trn = Indoor(split="train", transform=transforms_trn, masking=args.train_masking)
        dataset_val_01 = Indoor(split="val", transform=transforms_val_01, masking=args.eval_masking)
        dataset_val_02 = Celeba(split="val", transform=transforms_val_02, masking=args.eval_masking)
        dataset_val_03 = ImageNette(split="val", transform=transforms_val_03, masking=args.eval_masking)
    elif config.train_dataset.type == "ffhq":
        dataset_trn = FFHQ(split="train", transform=transforms_trn)
        # dataset_val = FFHQ(split="val", transform=transforms_val)
    elif config.train_dataset.type == "celeba":
        dataset_trn = Celeba(split="train", transform=transforms_trn, masking=args.train_masking)
        dataset_val_01 = Celeba(split="val", transform=transforms_val_01, masking=args.eval_masking)
        dataset_val_02 = ImageNette(split="val", transform=transforms_val_02, masking=args.eval_masking)
        dataset_val_03 = Indoor(split="val", transform=transforms_val_03, masking=args.eval_masking)
    elif config.train_dataset.type == "librispeech":
        dataset_trn = LibriSpeech("train-clean-100", transform=transforms_trn)
        # dataset_val = LibriSpeech("test-clean", transform=transforms_val)
    elif config.train_dataset.type in ["LearnitShapenet-cars", "LearnitShapenet-chairs", "LearnitShapenet-lamps"]:
        category_name = config.dataset.type.split("-")[-1]
        dataset_trn = LearnitShapenet(category_name, config=config.dataset.train_config)
        dataset_val = LearnitShapenet(category_name, config=config.dataset.val_config)
    else:
        raise ValueError("%s not supported..." % config.dataset.type)

    if config.get("trainer", "") == "stage_inr":
        dataset_trn = ImageOnlyDataset(dataset_trn)
        dataset_val_01 = ImageOnlyDataset(dataset_val_01)
        dataset_val_02 = ImageOnlyDataset(dataset_val_02)
        dataset_val_03 = ImageOnlyDataset(dataset_val_03)

    if SMOKE_TEST:
        dataset_len = config.experiment.total_batch_size * 2
        dataset_trn = torch.utils.data.Subset(dataset_trn, torch.randperm(len(dataset_trn))[:dataset_len])
        dataset_val_01 = torch.utils.data.Subset(dataset_val_01, torch.randperm(len(dataset_val_01))[:dataset_len])
        dataset_val_02 = torch.utils.data.Subset(dataset_val_02, torch.randperm(len(dataset_val_02))[:dataset_len])
        dataset_val_03 = torch.utils.data.Subset(dataset_val_03, torch.randperm(len(dataset_val_03))[:dataset_len])

    if logger is not None:
        logger.info(f"\n #train samples: {len(dataset_trn)}, #valid01 samples: {len(dataset_val_01)}, #valid02 samples: {len(dataset_val_02)}, #valid03 samples: {len(dataset_val_03)}")

    return dataset_trn, dataset_val_01, dataset_val_02, dataset_val_03
