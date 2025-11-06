import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageFolderDataset(Dataset):
    def __init__(self, root, size=256):
        self.root = root
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(exts)]
        self.tf = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img)
