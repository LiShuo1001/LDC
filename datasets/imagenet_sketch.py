import os
from torch.utils.data import Dataset

from typing import Any, Tuple
from PIL import Image
from .imagenet import imagenet_classes, imagenet_templates


class ImageNetSketch(Dataset):
    """ImageNetSketch.

    This dataset is used for testing only.
    """
    def __init__(self, root, transform) -> None:
        super().__init__()
        self.template = imagenet_templates
        self.classnames = imagenet_classes
        self.transform = transform
        self.samples = []
        print(f"{root=}")
        folders = sorted(f.name for f in os.scandir(root) if f.is_dir())
        for class_id, folder in enumerate(folders):
            for cur_root, dirs, fnames in os.walk(os.path.join(root, folder)):
                if len(fnames) > 0:
                    for fname in fnames:
                        img_path = os.path.join(cur_root, fname)
                        self.samples.append((img_path, class_id))
        
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