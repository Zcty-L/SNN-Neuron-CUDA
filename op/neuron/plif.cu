#include "neuron.h"


namespace PLIFNode
{
    // grad_v_to_h
    template<typename T>
    struct GradVToHHardReset;

    template<>
    struct GradVToHHardReset<float>
    {
        __device__ __forceinline__ float operator()(float over_th) const
        { return over_th < 0; }
    };

    template<>
    struct GradVToHHardReset<half2>
    {
        __device__ __forceinline__ half2 operator()(half2 over_th) const
        { return __hle2(over_th, __float2half2_rn(0)); }
    };

    template<typename T>
    struct GradVToHSoftReset;

    template<>
    struct GradVToHSoftReset<float>
    {
        __device__ __forceinline__ float operator()(float over_th) const
        { return 1.0f; }
    };

    template<>
    struct GradVToHSoftReset<half2>
    {
        __device__ __forceinline__ half2 operator()(half2 over_th) const
        { return __float2half2_rn(1.0f); }
    };

    template<template<typename> class GradVToHFunc, typename T>
    __device__ T GradVToH(T over_th)
    { return GradVToHFunc<T>()(over_th); }
}


// --- --- --- --- --- --- --- --- LIFNode Backward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool decay_input, bool detach_reset, bool padding
>
__global__ void PLIFNodeBPTTFLOATKernel(
        float *__restrict__ grad_spike_seq,
        float *__restrict__ grad_v_seq,
        float *__restrict__ h_seq,
        float *__restrict__ v_seq,
        float *__restrict__ grad_x_seq,
        float *__restrict__ grad_tau_seq,
        const float v_th, const float v_reset, const float decay,
        const float alpha, const float args, const uint32_t numel,
        const uint32_t time_step)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    __shared__ float smem[8];

    bool isLegalIndex = idx + 3 < numel;
    uint32_t edgeIndex = numel - idx;

    float h[4], load[4];
    float var[4], grad_v_to_h[4], grad_h[4];
    float grad_tau = 0;

    const float grad_h_to_x = decay_input ? decay : 1.0f;
    const float grad_h_next_to_v = 1.0f - decay;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        h[i] = 0;
        var[i] = 0;
        grad_h[i] = 0;
    }

    if (idx < numel)
    {
        uint32_t index;
        for (int t = time_step - 1; t >= 0; t--)
        {
            index = numel * t + idx;

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
            }
            else
            {
                for (int i = 0; i < edgeIndex; i++)
                {
                    h[i] = h_seq[index + i];
                    load[i] = grad_spike_seq[index + i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - v_th;     // var = over_th
                grad_v_to_h[i] = PLIFNode::GradVToH<GradVToHFunc, float>(var[i]);
            }


            if (surrogateFunc == SurrogateFunc::ATan)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    // 2alpha / (4 + (math.pi * alpha * x).pow_(2)) * grad_output
                    //  alpha / (4 +                   pai * x * x) * grad_output
                    var[i] = alpha / (4.0f + args * var[i] * var[i]);  // var = grad_s_to_h
                }
            }
            else // SurrogateFunc::Sigmoid
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = 1.0f / (1.0f + expf(-alpha * var[i])); // 1.0f / (1.0f + expf(-alpha * over_th));
                    var[i] = (1.0f - var[i]) * var[i] * alpha;      // var = grad_s_to_h
                }
            }

            if (!detach_reset)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    grad_v_to_h[i] += (v_reset - h[i]) * var[i]; // var = grad_s_to_h
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
            }

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_v_seq[index]);
            }
            else
            {
                for (int i = 0; i < edgeIndex; i++)
                {
                    load[i] = grad_v_seq[index + i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_h[i] = grad_h[i] * grad_h_next_to_v + load[i];
                grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

                var[i] = grad_h[i] * grad_h_to_x;
            }

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
            }
            else
            {
                for (int i = 0; i < edgeIndex; i++)
                {
                    grad_x_seq[index + i] = var[i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                load[i] = 0;
            }
            if (t > 0)
            {
                if (!padding || isLegalIndex)
                {
                    FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(v_seq[index - numel]);
                }
                else
                {
                    for (int i = 0; i < edgeIndex; i++)
                    {
                        load[i] = v_seq[index - numel + i];
                    }
                }
            }

            if (decay_input)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (h[i] - load[i]) * grad_h[i]; // load = v
                    var[i] = var[i] / decay;
                    grad_tau += var[i];
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (v_reset - load[i]) * grad_h[i]; // load = v
                    grad_tau += var[i];
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        grad_tau += __shfl_xor_sync(0xFFFFFFFF, grad_tau, offset);

    if (lane_id == 0)
    {
        smem[warp_id] = grad_tau;
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        grad_tau = smem[threadIdx.x];
    }
    __syncthreads();

    if (warp_id == 0)
    {
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 4, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 2, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 1, 8);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(grad_tau_seq, grad_tau);
    }
}

// --- --- --- --- --- --- --- --- PLIFNode Backward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool decay_input, bool detach_reset, bool padding
>
__global__ void PLIFNodeBPTTHALFKernel(
        half *__restrict__ grad_spike_seq,
        half *__restrict__ grad_v_seq,
        half *__restrict__ h_seq,
        half *__restrict__ v_seq,
        half *__restrict__ grad_x_seq,
        float *__restrict__ grad_tau_seq,
        const half2 v_th, const half2 v_reset, const half2 decay,
        const half2 alpha, const half2 args, const uint32_t numel,
        const uint32_t time_step)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;

    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    __shared__ float smem[8];

    bool isLegalIndex = idx + 7 < numel;
    uint32_t edgeIndex = numel - idx;

    half2 h[4], load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];
    float grad_tau = 0;

    const half2 grad_h_to_x = decay_input ? decay : __float2half2_rn(1);
    const half2 grad_h_next_to_v = __float2half2_rn(1) - decay;

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        h[i] = __float2half2_rn(0);
        var[i] = __float2half2_rn(0);
        grad_h[i] = __float2half2_rn(0);
    }

    if (idx < numel)
    {
        uint32_t index;
        for (int t = time_step - 1; t >= 0; t--)
        {
            index = numel * t + idx;

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(h[0]) = FETCH_FLOAT4(h_seq[index]);
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
            }
            else
            {
                auto *h_ptr = (half *) &h[0];
                auto *grad_spike_ptr = (half *) &load[0];
                for (int i = 0; i < edgeIndex; i++)
                {
                    h_ptr[i] = h_seq[index + i];
                    grad_spike_ptr[i] = grad_spike_seq[index + i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = h[i] - v_th;      // var = over_th
                grad_v_to_h[i] = PLIFNode::GradVToH<GradVToHFunc, half2>(var[i]);
            }

            if (surrogateFunc == SurrogateFunc::ATan)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    // 2alpha / (4 + (math.pi * alpha * x).pow_(2)) * grad_output
                    // alpha_ / (4 +                   pai * x * x) * grad_output
                    var[i] = alpha / (__float2half2_rn(4.0f) + args * var[i] * var[i]);  // grad_s_to_h
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {    // 1 / (1 + exp(-alpha * over_th));
                    var[i] = __float2half2_rn(1) / (__float2half2_rn(1) + h2exp(-alpha * var[i]));
                    var[i] = (__float2half2_rn(1) - var[i]) * var[i] * alpha;  // grad_s_to_h
                }
            }

            if (!detach_reset)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    grad_v_to_h[i] += (v_reset - h[i]) * var[i]; // var = grad_s_to_h
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                var[i] = var[i] * load[i]; // var = grad_s_to_h(var) * grad_spike(load)
            }

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_v_seq[index]);
            }
            else
            {
                auto *ptr = (half *) &load[0];
                for (int i = 0; i < edgeIndex; i++)
                {
                    ptr[i] = grad_v_seq[index + i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                grad_h[i] = grad_h[i] * grad_h_next_to_v + load[i];
                grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];

                var[i] = grad_h[i] * grad_h_to_x;
            }

            if (!padding || isLegalIndex)
            {
                FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(var[0]);
            }
            else
            {
                auto *ptr = (half *) &var[0];
                for (int i = 0; i < edgeIndex; i++)
                {
                    grad_x_seq[index + i] = ptr[i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                load[i] = __float2half2_rn(0);
            }
            if (t > 0)
            {
                if (!padding || isLegalIndex)
                {
                    FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(v_seq[index - numel]);
                }
                else
                {
                    auto *ptr = (half *) &load[0];
                    for (int i = 0; i < edgeIndex; i++)
                    {
                        ptr[i] = v_seq[index - numel + i];
                    }
                }
            }

            if (decay_input)
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (h[i] - load[i]) * grad_h[i]; // load = v
                    var[i] = var[i] / decay;
                    grad_tau += __half2float(var[i].x), grad_tau += __half2float(var[i].y);
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    var[i] = (v_reset - load[i]) * grad_h[i]; // load = v

                    grad_tau += __half2float(var[i].x), grad_tau += __half2float(var[i].y);
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        grad_tau += __shfl_xor_sync(0xFFFFFFFF, grad_tau, offset);

    if (lane_id == 0)
    {
        smem[warp_id] = grad_tau;
    }
    __syncthreads();

    if (threadIdx.x < 8)
    {
        grad_tau = smem[threadIdx.x];
    }
    __syncthreads();

    if (warp_id == 0)
    {
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 4, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 2, 8);
        grad_tau += __shfl_xor_sync(0xFF, grad_tau, 1, 8);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(grad_tau_seq, grad_tau);
    }
}


// --- --- --- --- --- --- --- --- PLIFNode Backward Launch Sigmoid --- --- --- --- --- --- --- --- ---
template<SurrogateFunc surrogateFunc>
void PLIFNodeBPTTFLOATLaunch(
        float *grad_spike_seq, float *grad_v_seq, float *h_seq, float *v_seq,
        float *grad_x_seq, float *grad_tau,
        const float v_th, const float v_reset, const float decay, float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads)
{
    float args = 0;
    if (surrogateFunc == SurrogateFunc::ATan)
    {
        args = 3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha;
        alpha = 2.0f * alpha;
    }

    bool padding = numel % 4 != 0 ? true : false;
    uint32_t blocks = (numel / 4 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, true, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, true, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, false, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, false, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, true, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, true, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, false, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, false, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, true, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, true, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, false, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, false, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, true, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, true, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, false, true> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
            else
            {
                PLIFNodeBPTTFLOATKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, false, false> <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th, v_reset, decay, alpha, args, numel, T);
            }
        }
    }
}

template<SurrogateFunc surrogateFunc>
void PLIFNodeBPTTHALFLaunch(
        half *grad_spike_seq, half *grad_v_seq, half *h_seq, half *v_seq,
        half *grad_x_seq, float *grad_tau,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads)
{
    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 decay_half2 = __float2half2_rn(decay);
    half2 alpha_half2 = __float2half2_rn(alpha);
    half2 args_half2 = __float2half2_rn(0);

    if (surrogateFunc == SurrogateFunc::ATan)
    {
        args_half2 = __float2half2_rn(3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha);
        alpha_half2 = __float2half2_rn(2.0f * alpha);
    }

    bool padding = numel % 8 != 0 ? true : false;
    uint32_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, true, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, true, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, false, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, true, false, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, true, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, true, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, false, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHHardReset, surrogateFunc, false, false, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
    }
    else
    {
        if (decay_input && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, true, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, true, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if (decay_input && (!detach_reset))
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, false, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, true, false, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else if ((!decay_input) && detach_reset)
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, true, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, true, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
        else
        {
            if (padding)
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, false, true>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
            else
            {
                PLIFNodeBPTTHALFKernel<PLIFNode::GradVToHSoftReset, surrogateFunc, false, false, false>  <<<blocks, threads>>>(
                        grad_spike_seq, grad_v_seq, h_seq, v_seq, grad_x_seq, grad_tau,
                        v_th_half2, v_reset_half2, decay_half2, alpha_half2, args_half2, numel, T);
            }
        }
    }
}


template void PLIFNodeBPTTFLOATLaunch<SurrogateFunc::Sigmoid>(
        float *, float *, float *, float *, float *, float *,
        const float, const float, const float, float,
        ResetType, const bool, const bool,
        const uint32_t, const uint32_t, const uint32_t);

template void PLIFNodeBPTTFLOATLaunch<SurrogateFunc::ATan>(
        float *, float *, float *, float *, float *, float *,
        const float, const float, const float, float,
        ResetType, const bool, const bool,
        const uint32_t, const uint32_t, const uint32_t);

template void PLIFNodeBPTTHALFLaunch<SurrogateFunc::Sigmoid>(
        half *, half *, half *, half *, half *, float *,
        const float, const float, const float, const float,
        ResetType, const bool, const bool,
        const uint32_t, const uint32_t, const uint32_t);

template void PLIFNodeBPTTHALFLaunch<SurrogateFunc::ATan>(
        half *, half *, half *, half *, half *, float *,
        const float, const float, const float, const float,
        ResetType, const bool, const bool,
        const uint32_t, const uint32_t, const uint32_t);

