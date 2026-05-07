#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

// Colab's custom PyTorch build inlines CUDAStream::query() into the header and
// never emits it as an exported symbol in libc10_cuda.so. Defining it here
// makes csrc.so self-contained: the linker resolves all internal references
// locally instead of depending on whichever libc10_cuda.so is present at runtime.
namespace c10 { namespace cuda {
    bool CUDAStream::query() const {
        return cudaStreamQuery(stream()) == cudaSuccess;
    }
    void CUDAStream::synchronize() const {
        cudaStreamSynchronize(stream());
    }
}}
