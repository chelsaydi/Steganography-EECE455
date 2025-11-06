import torch
import math

def bits_per_image(h: int, w: int, bpp: float) -> int:
    """
    Number of payload bits to embed for an image of size HxW,
    given target capacity in bits-per-pixel (bpp).
    """
    return int(h * w * bpp)

def make_payload(batch_size: int, num_bits: int, device: str = "cpu") -> torch.Tensor:
    """
    Create a random bitstream payload: shape [B, L] with values in {0,1}.
    In your real pipeline you can replace this with real encrypted bits.
    """
    return torch.randint(0, 2, (batch_size, num_bits), device=device).float()

def payload_to_map(bits: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert payload bits [B, L] to a spatial tensor [B, 1, H, W].
    We repeat the bit vector enough times to cover H*W exactly, then trim.
    This avoids any shape mismatch for arbitrary L, H, W.
    """
    B, L = bits.shape
    total = H * W
    if L <= 0:
        raise ValueError("Payload length L must be > 0.")
    reps = math.ceil(total / L)        # how many repeats to reach >= total
    flat = bits.repeat(1, reps)[:, :total]   # [B, total]
    return flat.view(B, 1, H, W)
