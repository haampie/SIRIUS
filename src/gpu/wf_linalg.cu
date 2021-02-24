// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file residuals_aux.cu
 *
 *  \brief CUDA kernel to compute wave-function residuals on GPUs.
 */

#include "gpu/cuda_common.hpp"
#include "gpu/acc_runtime.hpp"

__global__ void wf_dot_kernel
(
    int num_rows_loc__,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t const* wf_y__,
    int reduced__,
    int mpi_rank__,
    acc_complex_double_t* result__
)
{
    int N = num_blocks(num_rows_loc__, blockDim.x);

    ACC_DYNAMIC_SHARED( char, sdata_ptr)
    acc_complex_double_t* sdata = (acc_complex_double_t*)&sdata_ptr[0];

    sdata[threadIdx.x] = make_accDoubleComplex(0, 0);

    for (int n = 0; n < N; n++) {
        int j = n * blockDim.x + threadIdx.x;
        if (j < num_rows_loc__) {
            int k = array2D_offset(j, blockIdx.x, num_rows_loc__);
            sdata[threadIdx.x] = accCadd(sdata[threadIdx.x], accCmul(wf_x__[k], wf_y__[k]));
        }
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (threadIdx.x % (2 * s) == 0) {
            sdata[threadIdx.x] = accCadd(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // if (!reduced__) {
            result__[blockIdx.x] = accCadd(result__[blockIdx.x], sdata[0]);
        // } else {
        //     if (mpi_rank__ == 0) {
        //         double x = wf__[array2D_offset(0, blockIdx.x, num_rows_loc__)].x;
        //         result__[blockIdx.x] += (2 * sdata[0] - x * x);
        //     }
        //     else {
        //         result__[blockIdx.x] += 2 * sdata[0];
        //     }
        // }
    }
}

extern "C" void wf_dot_gpu(acc_complex_double_t const* wf_x__,
                           acc_complex_double_t const* wf_y__,
                           int num_rows_loc__,
                           int nwf__,
                           int reduced__,
                           int mpi_rank__,
                           acc_complex_double_t* result__)
{
    dim3 grid_t(64);
    dim3 grid_b(nwf__);

    accLaunchKernel((wf_dot_kernel), dim3(grid_b), dim3(grid_t), grid_t.x * sizeof(acc_complex_double_t), 0, 
        num_rows_loc__,
        wf_x__,
        wf_y__,
        reduced__,
        mpi_rank__,
        result__
    );
}


__global__ void wf_axpby_kernel
(
    int const num_rows_loc__,
    acc_complex_double_t alpha,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t beta,
    acc_complex_double_t * wf_y__
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        wf_y__[k] = accCadd(accCmul(alpha, wf_x__[k]), accCmul(beta, wf_y__[k]));
    }
}

extern "C" void wf_axpby_gpu(
    acc_complex_double_t alpha,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t beta,
    acc_complex_double_t* wf_y__,
    int num_rows_loc__,
    int num_bands__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((wf_axpby_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_rows_loc__,
        alpha,
        wf_x__,
        beta,
        wf_y__
    );
}

// Y <= X + Y * B where B is a diagonal matrix.
__global__ void wf_xpby_kernel
(
    int const num_rows_loc__,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t const* betas,
    acc_complex_double_t * wf_y__
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        wf_y__[k] = accCadd(wf_x__[k], accCmul(betas[ibnd], wf_y__[k]));
    }
}

extern "C" void wf_xpby_gpu(
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t const* betas,
    acc_complex_double_t* wf_y__,
    int num_rows_loc__,
    int num_bands__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((wf_xpby_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_rows_loc__,
        wf_x__,
        betas,
        wf_y__
    );
}

// Y <= X * A + Y where A is a diagonal matrix.
__global__ void wf_axpy_kernel
(
    int const num_rows_loc__,
    acc_complex_double_t const* alphas,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t * wf_y__
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;

    if (j < num_rows_loc__) {
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        wf_y__[k] = accCadd(accCmul(alphas[ibnd], wf_x__[k]), wf_y__[k]);
    }
}

extern "C" void wf_axpy_gpu(
    acc_complex_double_t const* alphas,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t* wf_y__,
    int num_rows_loc__,
    int num_bands__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((wf_axpy_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_rows_loc__,
        alphas,
        wf_x__,
        wf_y__
    );
}


// Y[:, ids[i]] <= X[:, i] * A[i, i] + Y[:, i]
__global__ void wf_axpy_scatter_kernel
(
    int const num_rows_loc__,
    acc_complex_double_t const* alphas,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t * wf_y__,
    int const * ids
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int ibnd = blockIdx.y;
    int ibnd_scatter = ids[blockIdx.y];

    if (j < num_rows_loc__) {
        int k = array2D_offset(j, ibnd, num_rows_loc__);
        int ids_k = array2D_offset(j, ibnd_scatter, num_rows_loc__);
        wf_y__[ids_k] = accCadd(accCmul(alphas[ibnd], wf_x__[k]), wf_y__[k]);
    }
}

extern "C" void wf_axpy_scatter_gpu(
    acc_complex_double_t const* alphas,
    acc_complex_double_t const* wf_x__,
    acc_complex_double_t* wf_y__,
    int const * ids__,
    int num_rows_loc__,
    int num_bands__)
{
    dim3 grid_t(64);
    dim3 grid_b(num_blocks(num_rows_loc__, grid_t.x), num_bands__);

    accLaunchKernel((wf_axpy_scatter_kernel), dim3(grid_b), dim3(grid_t), 0, 0, 
        num_rows_loc__,
        alphas,
        wf_x__,
        wf_y__,
        ids__
    );
}
