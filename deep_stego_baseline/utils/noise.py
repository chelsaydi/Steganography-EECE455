import io, random
from PIL import Image
import torch
import torchvision.transforms.functional as TF

class NoiseLayer(torch.nn.Module):
    def __init__(self, jpeg_min_q=50, jpeg_max_q=95, p_jpeg=0.5, p_resize=0.5, p_gauss=0.5):
        super().__init__()
        self.jpeg_min_q = jpeg_min_q
        self.jpeg_max_q = jpeg_max_q
        self.p_jpeg = p_jpeg
        self.p_resize = p_resize
        self.p_gauss = p_gauss

    def forward(self, x):
        B = x.size(0)
        outs = []
        for i in range(B):
            img = TF.to_pil_image(x[i].cpu())
            if random.random() < self.p_jpeg:
                q = random.randint(self.jpeg_min_q, self.jpeg_max_q)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=q, optimize=True)
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            if random.random() < self.p_resize:
                w, h = img.size
                scale = random.uniform(0.7, 1.0)
                img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC).resize((w, h), Image.BICUBIC)
            tens = TF.to_tensor(img)
            if random.random() < self.p_gauss:
                noise = torch.randn_like(tens) * 0.01
                tens = torch.clamp(tens + noise, 0.0, 1.0)
            outs.append(tens)
        return torch.stack(outs, dim=0).to(x.device)
