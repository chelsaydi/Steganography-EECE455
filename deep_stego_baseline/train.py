import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageFolderDataset
from payload import bits_per_image, make_payload, payload_to_map
from models.cnn_hidden import Encoder, Decoder
# from utils.noise import NoiseLayer   # <-- disabled for overfit test
from utils.metrics import mse, psnr, ber


def parse_args():
    ap = argparse.ArgumentParser(description="Train CNN (HiDDeN-style) steganography baseline")
    ap.add_argument("--data_dir", type=str, required=True, help="Path to training images folder")
    ap.add_argument("--val_dir", type=str, default=None, help="Optional path to validation images")
    ap.add_argument("--size", type=int, default=256, help="Resize images to size x size")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size (set 1 for tiny sets)")
    ap.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--bpp", type=float, default=0.10, help="Target bits-per-pixel capacity")
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for cover MSE loss")
    ap.add_argument("--beta", type=float, default=10.0, help="Weight for payload BCE loss")
    ap.add_argument("--out", type=str, default="runs", help="Output dir for checkpoints")
    return ap.parse_args()


def make_loader(root, size, batch_size, shuffle):
    ds = ImageFolderDataset(root, size=size)
    if len(ds) == 0:
        raise RuntimeError(f"No images found in '{root}'. Supported extensions: .jpg .jpeg .png .bmp")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,      # Windows-friendly
        drop_last=False     # keep even tiny batches
    )
    return ds, dl


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    # --- Data ---
    train_ds, train_dl = make_loader(args.data_dir, args.size, args.batch_size, shuffle=True)
    if args.val_dir:
        val_ds, val_dl = make_loader(args.val_dir, args.size, args.batch_size, shuffle=False)
    else:
        val_dl = None

    # --- Payload length (per image) ---
    num_bits = bits_per_image(args.size, args.size, args.bpp)
    if num_bits <= 0:
        raise ValueError(f"Computed num_bits <= 0 for size={args.size}, bpp={args.bpp}. Increase bpp.")

    print(f"[Info] Image size: {args.size}x{args.size}  |  bpp: {args.bpp}  |  bits/image: {num_bits}")

    # --- Models / Opt / Losses ---
    enc = Encoder(payload_channels=1).to(device)
    dec = Decoder(num_bits=num_bits).to(device)
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    # ---- Disable noise for overfit sanity check ----
    class Identity(nn.Module):
        def forward(self, x): 
            return x
    noise = Identity().to(device)
    # ------------------------------------------------

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", ncols=80)

        for cover in pbar:
            cover = cover.to(device)  # [B,3,H,W]
            B, _, H, W = cover.shape

            # Random payload (replace with real encrypted bits if you have them)
            bits = make_payload(B, num_bits, device=device)  # [B, L]
            payload_map = payload_to_map(bits, H, W).to(device)  # [B,1,H,W]

            # Forward
            stego = enc(cover, payload_map)            # [B,3,H,W]
            noisy = noise(stego)                       # identity (no noise)
            logits = dec(noisy)                        # [B, L]

            # Loss
            loss_cover = mse(stego, cover)
            loss_bits = bce(logits, bits)
            loss = args.alpha * loss_cover + args.beta * loss_bits

            # Step
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                psnr_val = psnr(loss_cover).item()
                ber_val = ber(bits, torch.sigmoid(logits)).item()

            pbar.set_postfix(loss=float(loss.item()), psnr=f"{psnr_val:.2f}", ber=f"{ber_val:.4f}")
            global_step += 1

        # --- Simple validation ---
        if val_dl is not None:
            enc.eval()
            dec.eval()
            v_mse, v_psnr, v_ber, v_n = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for cover in val_dl:
                    cover = cover.to(device)
                    B, _, H, W = cover.shape
                    bits = make_payload(B, num_bits, device=device)
                    payload_map = payload_to_map(bits, H, W).to(device)
                    stego = enc(cover, payload_map)
                    logits = dec(stego)
                    lc = mse(stego, cover).item()
                    v_mse += lc
                    v_psnr += psnr(torch.tensor(lc)).item()
                    v_ber += ber(bits, torch.sigmoid(logits)).item()
                    v_n += 1
            if v_n > 0:
                print(f"[VAL] MSE: {v_mse / v_n:.6f}  PSNR: {v_psnr / v_n:.2f} dB  BER: {v_ber / v_n:.4f}")
            else:
                print("[VAL] No validation batches (check val folder and batch_size).")

        # --- Save checkpoint ---
        ckpt = {
            "enc": enc.state_dict(),
            "dec": dec.state_dict(),
            "num_bits": num_bits,
            "size": args.size,
            "bpp": args.bpp,
        }
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        print(f"[Info] Saved: {os.path.join(args.out, 'last.pt')}")

    print("[Done] Training completed.")


if __name__ == "__main__":
    main()
