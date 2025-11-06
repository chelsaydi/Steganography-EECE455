import os, glob
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T

from models.cnn_hidden import Encoder, Decoder
from payload import payload_to_map

# -------- Config --------
device = "cuda" if torch.cuda.is_available() else "cpu"
size = 256
ckpt_path = "runs/last.pt"
val_dir = "data/val"
secret_msg = "HI"
stego_out = "stego_demo.png"
panel_out = "demo_panel.png"
# ------------------------

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location=device)
num_bits = int(ckpt["num_bits"])
enc = Encoder(payload_channels=1).to(device)
dec = Decoder(num_bits=num_bits).to(device)
enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"])
enc.eval(); dec.eval()
print(f"[info] num_bits={num_bits}")

# Pick a cover
cands = [p for p in glob.glob(os.path.join(val_dir, "*"))
         if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
if not cands:
    raise RuntimeError(f"No images in {val_dir}")
cover_path = cands[0]
print(f"[info] cover={cover_path}")

# Load cover
to_tensor = T.Compose([T.Resize((size,size)), T.ToTensor()])
cover_img = Image.open(cover_path).convert("RGB")
cover = to_tensor(cover_img).unsqueeze(0).to(device)
B,_,H,W = cover.shape

# Build payload (message at start of num_bits)
msg_bits = ''.join(format(ord(c), '08b') for c in secret_msg)
msg_len = len(msg_bits)
if msg_len > num_bits:
    raise ValueError(f"Message too long ({msg_len} bits) > num_bits {num_bits}.")
bits_vec = np.zeros(num_bits, dtype=np.float32)
bits_vec[:msg_len] = [int(b) for b in msg_bits]
bits = torch.tensor(bits_vec, device=device).unsqueeze(0)     # [1, num_bits]
pmap = payload_to_map(bits, H, W)                              # [1,1,H,W]

# Encode & decode
with torch.no_grad():
    stego = enc(cover, pmap).clamp(0,1)
    logits = dec(stego)
    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

# Recover text (first msg_len bits)
rec = (probs[:msg_len] > 0.5).astype(np.uint8)
chars = []
for i in range(0, msg_len, 8):
    byte = rec[i:i+8]
    if len(byte) < 8: break
    chars.append(chr(int(''.join(map(str, byte.tolist())), 2)))
rec_msg = ''.join(chars)

# Save stego separately
stego_np = (stego[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
Image.fromarray(stego_np).save(stego_out)
print(f"[done] stego -> {stego_out}")
print(f"[result] recovered -> {rec_msg}")

# Build side-by-side panel
def make_panel(cover_img, stego_img, secret_text, recovered_text):
    cover_img = cover_img.resize((size,size))
    stego_img = stego_img.resize((size,size))
    caption_h = 80
    panel = Image.new("RGB", (2*size, size+caption_h), (255,255,255))
    panel.paste(cover_img, (0,0))
    panel.paste(stego_img, (size,0))

    draw = ImageDraw.Draw(panel)
    draw.text((size//3, 5), "COVER", fill=(0,0,0))
    draw.text((size + size//3, 5), "STEGO", fill=(0,0,0))
    draw.text((10, size+10), f"Secret: {secret_text}", fill=(0,0,0))
    draw.text((10, size+35), f"Recovered: {recovered_text}", fill=(0,0,0))
    draw.text((10, size+60), "Left: Cover | Right: Stego", fill=(0,0,0))
    return panel

panel = make_panel(cover_img, Image.fromarray(stego_np), secret_msg, rec_msg)
panel.save(panel_out)
print(f"[done] panel -> {panel_out}")
