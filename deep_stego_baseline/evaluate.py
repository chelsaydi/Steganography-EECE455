import argparse, torch
from torch.utils.data import DataLoader
from dataset import ImageFolderDataset
from payload import bits_per_image, make_payload, payload_to_map
from models.cnn_hidden import Encoder, Decoder
from utils.metrics import mse, psnr, ber

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--bpp", type=float, default=0.1)
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ImageFolderDataset(args.data_dir, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    ckpt = torch.load(args.ckpt, map_location=device)
    num_bits = ckpt.get("num_bits", bits_per_image(args.size, args.size, args.bpp))

    enc = Encoder(payload_channels=1).to(device)
    dec = Decoder(num_bits=num_bits).to(device)
    enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"])
    enc.eval(); dec.eval()

    mse_vals, psnr_vals, ber_vals = [], [], []
    with torch.no_grad():
        for cover in dl:
            cover = cover.to(device)
            B, _, H, W = cover.shape
            bits = make_payload(B, num_bits, device=device)
            payload_map = payload_to_map(bits, H, W).to(device)
            stego = enc(cover, payload_map)
            logits = dec(stego)
            m = mse(stego, cover).item()
            p = psnr(torch.tensor(m)).item()
            b = ber(bits, torch.sigmoid(logits)).item()
            mse_vals.append(m); psnr_vals.append(p); ber_vals.append(b)

    print(f"MSE: {sum(mse_vals)/len(mse_vals):.6f}")
    print(f"PSNR: {sum(psnr_vals)/len(psnr_vals):.2f} dB")
    print(f"BER: {sum(ber_vals)/len(ber_vals):.4f}")

if __name__ == "__main__":
    main()
