
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

SEED = 6304
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
ALL_DOMAINS = SOURCE_DOMAINS + [TARGET_DOMAIN]

# Adjust these two if your specific PACS download differs.
ROOT_SUBDIR = "kfold"  # set to "" if your download has no kfold/ level
DOMAIN_DIR_MAP = {d: d for d in ALL_DOMAINS}  # e.g. {"art_painting": "art painting"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def domain_root(pacs_root: str, domain: str) -> str:
    d = DOMAIN_DIR_MAP.get(domain, domain)
    return os.path.join(pacs_root, ROOT_SUBDIR, d) if ROOT_SUBDIR else os.path.join(pacs_root, d)


def list_domain_files(pacs_root: str, domain: str):
    files, labels = [], []
    root = domain_root(pacs_root, domain)
    for cls_idx, cls_name in enumerate(PACS_CLASSES):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {cls_dir}\n"
                f"Check ROOT_SUBDIR / DOMAIN_DIR_MAP in shared/pacs.py against "
                f"your actual PACS download's folder layout.")
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(cls_dir, fname))
                labels.append(cls_idx)
    return files, labels


class PACSDomainDataset(Dataset):

    def __init__(self, files, labels, indices, transform):
        self.files = [files[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]
