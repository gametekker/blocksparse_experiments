#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for blockwise sparse matrix-vector multiplication
__global__ void blockwise_multiply_kernel(
    const float* __restrict__ matrix, 
    const float* __restrict__ vector, 
    float* __restrict__ output,
    int rows, int cols, int block_size, 
    const int* __restrict__ active_blocks, int num_active_blocks) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        float sum = 0.0f;
        for (int b = 0; b < num_active_blocks; b++) {
            int block_idx = active_blocks[b];
            int base = block_idx * block_size;

            // Ensure we don't go out of bounds
            if (base + block_size <= cols) {
                for (int j = 0; j < block_size; j++) {
                    sum += matrix[row * cols + base + j] * vector[base + j];
                }
            }
        }
        output[row] = sum;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor blockwise_multiply(
    torch::Tensor matrix, 
    torch::Tensor vector, 
    torch::Tensor active_blocks,
    int block_size) 
{
    const int rows = matrix.size(0);
    const int cols = matrix.size(1);
    const int num_active_blocks = active_blocks.size(0);

    // Initialize output tensor on the same device
    auto output = torch::zeros({rows}, matrix.options());

    // Configure CUDA launch parameters
    const int threads = 256;
    const int blocks = (rows + threads - 1) / threads;

    // Launch the CUDA kernel
    blockwise_multiply_kernel<<<blocks, threads>>>(
        matrix.data_ptr<float>(), 
        vector.data_ptr<float>(), 
        output.data_ptr<float>(), 
        rows, cols, block_size, 
        active_blocks.data_ptr<int>(), 
        num_active_blocks
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    return output;
}

// Bind the C++ function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockwise_multiply", &blockwise_multiply, "Blockwise Sparse Matrix-Vector Multiplication (CUDA)");
}
