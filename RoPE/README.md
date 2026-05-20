# RoPE Triton Implementation

## Overview

Rotary Position Embedding injects positional information into transformer models by rotating query and key vectors based on their position in the sequence. This allows the attention mechanism to naturally capture relative distances between tokens without adding learnable position embeddings.

## File Structure

### main.py
Contains the core Triton kernel implementation and the Python wrapper. It handles precomputing the frequency table, launching the GPU kernel, and managing memory allocation for the rotated output.

### test.py
Validates the Triton implementation against a PyTorch reference. It generates random input tensors, applies both implementations, and prints the outputs side by side for visual verification.

### benchmark.py
Measures execution time for both implementations across multiple runs to determine throughput and latency improvements.

## Implementation Logic

The kernel processes tokens in pairs. For each pair at position m, it applies a 2D rotation matrix using precomputed cosine and sine values derived from the frequency formula:

theta_i = 10000^(-2i/d)

The rotation is computed as:
- Even indices: x_even * cos + (-x_odd) * sin
- Odd indices: x_odd * cos + x_even * sin

This ensures efficient vectorized execution on the GPU by separating even and odd offsets and applying the rotation math in a single pass.

## Correctness Verification

The test script compares outputs from both implementations.

PyTorch Output:
```
tensor([[-1.3887,  0.8535, -1.1699,  ...,  1.7471,  0.2566,  0.3557],
        [ 0.7891, -0.7451, -0.8325,  ...,  0.4885,  0.1254,  0.0751],
        ...
        [-0.2081, -1.2217,  0.1074,  ...,  0.0369,  0.0884,  0.6489]],
       device='cuda:0', dtype=torch.float16)
```

Triton Output:
```
tensor([[-1.3887,  0.8535, -1.1699,  ...,  1.7471,  0.2566,  0.3557],
        [ 0.7891, -0.7451, -0.8325,  ...,  0.4885,  0.1254,  0.0751],
        ...
        [-0.2081, -1.2217,  0.1074,  ...,  0.0369,  0.0884,  0.6489]],
       device='cuda:0', dtype=torch.float16)
```

The outputs match numerically within floating point tolerance.

## Benchmarks

| Implementation | Time (ms) |
| :--- | :--- |
| Triton | 0.1056 |
| PyTorch | 0.5622 |

**Speedup: 5.33x**

## Conclusion

The Triton implementation achieves a significant speedup over PyTorch by fusing the rotation operations and leveraging GPU parallelism. This makes it suitable for high-performance inference engines where latency and throughput are critical.
