# Deep Image Steganography — Baseline (HiDDeN-style CNN)
Minimal PyTorch baseline to hide an already-encrypted message (ciphertext) inside images.

## Quickstart
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train.py --data_dir data/train --val_dir data/val --epochs 5 --bpp 0.1
python evaluate.py --ckpt runs/last.pt --data_dir data/val --bpp 0.1
