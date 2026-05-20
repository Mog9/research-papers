import torch
import triton
from main import precompute_rope, launch_rope


def apply_rope_torch(x, rope_emb):
    seq_len = x.shape[0]
    rope_emb_sliced = rope_emb[:seq_len, :]
    cos = rope_emb_sliced.cos()
    sin = rope_emb_sliced.sin()
    x_reshaped = x.float().reshape(seq_len, -1, 2)
    x_partner = torch.stack([-x_reshaped[..., 1], x_reshaped[...,0]], dim=-1)
    x_partner = x_partner.reshape(seq_len, -1)
    return (x * cos + x_partner * sin).type_as(x)


def benchmark_rope():
    seq_len, head_dim = 512, 2048
    x = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
    emb = precompute_rope(head_dim, seq_len).to('cuda')
    cos, sin = emb.cos(), emb.sin()
    
    launch_rope(x, cos, sin)
    
    ms_triton = triton.testing.do_bench(lambda: launch_rope(x, cos, sin))
    ms_torch = triton.testing.do_bench(lambda: apply_rope_torch(x, emb))
    
    print(f"Triton: {ms_triton:.4f} ms")
    print(f"Torch:  {ms_torch:.4f} ms")
    print(f"Speedup: {ms_torch/ms_triton:.2f}x")


if __name__ == "__main__":
    benchmark_rope()
