import torch
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


def test_rope():
    seq_len, head_dim = 512, 2048
    x = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
    
    emb = precompute_rope(head_dim, seq_len).to('cuda')
    cos, sin = emb.cos(), emb.sin()
    
    out_torch = apply_rope_torch(x, emb)
    out_triton = launch_rope(x, cos, sin)
    
    print("PyTorch Output:")
    print(out_torch)
    print("\nTriton Output:")
    print(out_triton)
    
    if torch.allclose(out_torch, out_triton, atol=1e-3):
        print("\nCorrectness: PASSED")
    else:
        print("\nCorrectness: FAILED")
        print(f"Max diff: {torch.max(torch.abs(out_torch - out_triton))}")


if __name__ == "__main__":
    test_rope()
