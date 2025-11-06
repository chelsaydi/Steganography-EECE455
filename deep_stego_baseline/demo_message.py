import os, glob, math
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from models.cnn_hidden import Encoder, Decoder
from payload import payload_to_map

# ------- Config -------
device = "cuda" if torch.cuda.is_available() else "cpu"
size = 256                              # must match training
ckpt_path = "runs/last.pt"              # trained weights
val_dir = "data/val"                    # pick a cover from here
msg = "HELLO CLASS"                     # your secret message
out_path = "stego_demo.png"
# ----------------------

# 1) Load checkpoint and restore model shapes (especially num_bits)
ckpt = torch.load(ckpt_path, map_location=device)
num_bits = int(ckpt.get("num_bits"))    # payload length used in training (e.g., ~6553 for bpp=0.10)
print(f"[info] num_bits from checkpoint: {num_bits}")

enc = Encoder(payload_channels=1).to(device)
dec = Decoder(num_bits=num_bits).to(device)
enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"])
enc.eval(); dec.eval()

# 2) Pick a real cover image from val_dir
candidates = [p for p in glob.glob(os.path.join(val_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
if not candidates:
    raise RuntimeError(f"No JPG/PNG/BMP images found in {val_dir}")
cover_path = candidates[0]
print(f"[info] Using cover: {cover_path}")

# 3) Prepare the cover tensor
to_tensor = T.Compose([T.Resize((size, size)), T.ToTensor()])
cover_img = Image.open(cover_path).convert("RGB")
cover = to_tensor(cover_img).unsqueeze(0).to(device)   # [1,3,H,W]
B, _, H, W = cover.shape

# 4) Build the payload bits: place message bits at the start of a length = num_bits vector
msg_bits = ''.join(format(ord(c), '08b') for c in msg)
msg_len = len(msg_bits)
if msg_len > num_bits:
    raise ValueError(f"Message too long: {msg_len} bits > num_bits {num_bits}. Use a shorter message or train with higher bpp.")
bits_vec = np.zeros(num_bits, dtype=np.float32)
bits_vec[:msg_len] = [int(b) for b in msg_bits]
bits = torch.tensor(bits_vec, device=device).unsqueeze(0)  # [1, num_bits]

# 5) Convert to spatial payload map for the encoder
pmap = payload_to_map(bits, H, W)                          # [1,1,H,W]

# 6) Encode -> get stego; Decode -> recover logits
with torch.no_grad():
    stego = enc(cover, pmap).clamp(0, 1)
    logits = dec(stego)                                    # [1, num_bits]
    probs  = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()

# 7) Threshold to bits and reconstruct message (read only the first msg_len bits)
recovered = (probs[:msg_len] > 0.5).astype(np.uint8)
chars = []
for i in range(0, msg_len, 8):
    byte = recovered[i:i+8]
    if len(byte) < 8: break
    chars.append(chr(int(''.join(map(str, byte.tolist())), 2)))
rec_msg = ''.join(chars)

# 8) Save the stego image
stego_np = (stego[0].detach().cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
Image.fromarray(stego_np).save(out_path)

print(f"[done] Saved stego to: {out_path}")
print(f"[result] Recovered message: {rec_msg}")
