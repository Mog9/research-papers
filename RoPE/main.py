import torch
import triton
import triton.language as tl

def precompute_rope(head_dim: int, max_seq_len: int, base=10000):
    inv_frequency = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_frequency)
    emb = freqs.repeat_interleave(2, dim=-1)
    return emb


@triton.jit
def rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE * 2
    offsets = tl.arange(0, BLOCK_SIZE)
    even_offset = block_start + 2 * offsets
    odd_offset = block_start + 2 * offsets + 1

    mask_even = even_offset < n_elements
    mask_odd = odd_offset < n_elements

    x_even = tl.load(x_ptr + even_offset, mask=mask_even) 
    x_odd = tl.load(x_ptr + odd_offset, mask=mask_odd)
    
    cos = tl.load(cos_ptr + even_offset, mask=mask_even)
    sin = tl.load(sin_ptr + even_offset, mask=mask_even)

    result_even = x_even * cos + (-x_odd) * sin
    result_odd = x_odd * cos + x_even * sin
    
    tl.store(out_ptr + even_offset, result_even, mask=mask_even)
    tl.store(out_ptr + odd_offset, result_odd, mask=mask_odd) 


def launch_rope(x, cos, sin):
    x_flat = x.contiguous().view(-1)
    cos_flat = cos.contiguous().view(-1)
    sin_flat = sin.contiguous().view(-1)
    
    out_flat = torch.empty_like(x_flat)
    
    n_elements = x_flat.numel()
    BLOCK_SIZE = 1024
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE'] * 2),)
    
    rope_kernel[grid](
        x_flat, cos_flat, sin_flat, out_flat, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_flat.view_as(x)
