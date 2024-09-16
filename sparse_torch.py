import torch
import blockwise_sparse  # This is the name specified in setup.py
import stk

def create_sparse_vector_blocks(n, s, r, device='cuda'):
    """
    Create a block-wise sparse vector of size n with block size r and sparsity s on the GPU.

    Parameters:
    n (int): Size of the vector.
    s (float): Sparsity level (between 0 and 1), where 1 means all blocks are sparse.
    r (int): Block size, must divide n exactly.
    device (str): Device where the tensor will be allocated ('cuda' or 'cpu').

    Returns:
    torch.Tensor: Sparse vector of size n with block-wise sparsity on the specified device.
    torch.Tensor: Indices of active (non-sparse) blocks.
    """
    assert n % r == 0, "Block size r must divide n exactly."
    assert 0 <= s <= 1, "Sparsity must be between 0 and 1."

    num_blocks = n // r
    # Generate a mask where 1 indicates the block is dense, 0 is sparse
    block_mask = (torch.rand(num_blocks, device=device) > s).int()

    # Extract indices of active blocks
    active_blocks = torch.nonzero(block_mask, as_tuple=False).squeeze()

    # Handle case when no blocks are active
    if active_blocks.dim() == 0:
        active_blocks = active_blocks.unsqueeze(0) if block_mask[active_blocks].item() else torch.tensor([], device=device)

    # Create the full mask
    full_mask = block_mask.repeat_interleave(r).float()

    # Create a dense random vector
    dense_vector = torch.randn(n, device=device)

    # Apply the block-wise mask to create a sparse vector
    sparse_vector = dense_vector * full_mask

    return sparse_vector, active_blocks.to(torch.int32)

def create_dense_matrix(m, n, device='cuda'):
    """
    Create a dense random matrix of size m x n on the specified device.

    Parameters:
    m (int): Number of rows.
    n (int): Number of columns.
    device (str): Device where the tensor will be allocated ('cuda' or 'cpu').

    Returns:
    torch.Tensor: Dense matrix of size m x n.
    """
    return torch.randn(m, n, device=device)

"""
Please continue your experiments on bechmarking the speedups with different frameworks (CUDA, Triton):
You may refer to the following matrix multiplication dimension as a starting point: weight matrix of size (2048, 4096), input size (4096, 1), input can be divided into 64 blocks (each with a dimension of 64), around 25% - 50% of the blocks can be all-zero (the sparsity mask is determined at runtime);
Please try the Triton-based implementation first, if it works, it serves as a baseline with validated performance; if it does not work, please post your findings in this channel and we can look into it together
"""

# Example usage
m, n, s, r = 8000, 8000, 0.90, 64
sparse_matrix = create_dense_matrix(m, n, device='cpu')
sparse_vector, active_blocks = create_sparse_vector_blocks(n, s, r, device='cpu')
print(sparse_matrix)
print(sparse_vector)

sparse_matrix=sparse_matrix.to(device='cuda')
sparse_vector=sparse_vector.to(device='cuda')

# Using torch matmul
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
result1 = torch.matmul(sparse_matrix, sparse_vector)
end.record()
torch.cuda.synchronize()
elapsed_time_ms = start.elapsed_time(end)
print(elapsed_time_ms)

# Using custom CUDA kernel
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
result2 = blockwise_sparse.blockwise_multiply(sparse_matrix, sparse_vector,active_blocks,r)
end.record()
torch.cuda.synchronize()
elapsed_time_ms = start.elapsed_time(end)
print(elapsed_time_ms)

"""
# Convert dense matrix to an STK block-sparse matrix
sparse_vector=sparse_vector.reshape((n,1))
print(sparse_vector.dim())
sparse_matrix_stk = stk.ops.to_sparse(sparse_vector)
# Perform block-sparse matrix-vector multiplication using STK
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
result_stk = stk.ops.dds(sparse_matrix, sparse_matrix_stk)
end.record()
torch.cuda.synchronize()
elapsed_time_stk = start.elapsed_time(end)
print(f"STK Block-Sparse MatVec Time: {elapsed_time_stk:.4f} ms")
"""

if torch.allclose(result1, result2, atol=1e-5):
    print("Verification passed: Results match.")
else:
    print("Verification failed: Results do not match.")
    max_diff = (result1 - result2).abs().max()
    print(f"Results differ! Max difference: {max_diff.item()}")
