import datasets
import torch
from datasets import register
from torch.utils.data import Dataset
from torchvision import transforms


@register("imgrec_dataset")
class ImgrecDataset(Dataset):
    def __init__(self, imageset, resize):
        self.imageset = datasets.make(imageset)
        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, idx):
        x = self.transform(self.imageset[idx])
        input = self.random_masking(x)
        return {"inp": input, "gt": x}

    def random_masking(self, image, ratio=0.75, patch_num=14):
        """
        image: (3, H, W)
        mask with ones in randomly selected patches
        """
        C, H, W = image.shape
        patch_size = H // patch_num
        mask = torch.ones((H, W), dtype=torch.float32, device=image.device)
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                if torch.rand(1) < ratio:
                    mask[i : i + patch_size, j : j + patch_size] = 0
        mask = mask.view(1, H, W).repeat(C, 1, 1)
        return image * mask
