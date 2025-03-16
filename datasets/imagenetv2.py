import os
from torch.utils.data import Dataset

from typing import Any, Tuple
from PIL import Image
from .imagenet import imagenet_classes, imagenet_templates


class ImageNetV2(Dataset):
    """ImageNetV2.

    This dataset is used for testing only.
    """
    def __init__(self, root, transform) -> None:
        super().__init__()
        self.template = imagenet_templates
        self.classnames = imagenet_classes
        self.transform = transform
        self.samples = []
        print(f"{root=}")
        for cur_root, dirs, fnames in os.walk(root):
            if len(fnames) > 0:
                cur_class_id = int(cur_root.split('/')[-1])
                for fname in fnames:
                    img_path = os.path.join(cur_root, fname)
                    self.samples.append((img_path, cur_class_id))
        
    def loader(self, path )-> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
            
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        new_sample = self.transform(sample)
        return new_sample, target
    
    def __len__(self) -> int:
        return len(self.samples)