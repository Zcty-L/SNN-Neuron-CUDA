#include "neuron.h"


namespace IFNode
{
    // reset
    template<typename T>
    struct HardReset
    {
        __device__ __forceinline__ T operator()(const T &v, const T &spike, const T &v_reset, const T &v_th) const
        { return v - spike * v + spike * v_reset; }
    };

    template<typename T>
    struct SoftReset
    {
        __device__ __forceinline__ T operator()(const T &v, const T &spike, const T &v_reset, const T &v_th) const
        { return v - spike * v_th; }
    };

    template<template<typename> class ResetFunc, typename T>
    __inline__ __device__ T NeuronReset(T v, T spike, T v_reset, T v_th)
    {
        v = ResetFunc<T>()(v, spike, v_reset, v_th);
        return v;
    }

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


// --- --- --- --- --- --- --- --- IFNode Forward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class ResetFunc, bool padding
>
__global__ void IFNodeFPTTFLOATKernel(
        float *__restrict__ inputs,
        float *__restrict__ spikes_seq,
        float *__restrict__ h_seq,
        float *__restrict__ v_seq,
        const float v_th, const float v_reset,
        const uint32_t numel, const uint32_t time_step)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    uint32_t edgeIndex = numel - idx;

    float v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = 0;
    }

    uint32_t index;
    for (int t = 0; t < time_step; t++)
    {
        index = idx + numel * t;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                v[i] = inputs[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            v[i] = last_v[i] + v[i];
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = v[i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = v[i] >= v_th;
            last_v[i] = IFNode::NeuronReset<ResetFunc, float>(v[i], spikes[i], v_reset, v_th);
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
            FETCH_FLOAT4(v_seq[index]) = FETCH_FLOAT4(last_v[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spikes[i];
                v_seq[index + i] = last_v[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- IFNode Forward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class ResetFunc, bool padding
>
__global__ void IFNodeFPTTHALFKernel(
        half *__restrict__ inputs,
        half *__restrict__ spikes_seq,
        half *__restrict__ h_seq,
        half *__restrict__ v_seq,
        const half2 v_th, const half2 v_reset,
        const uint32_t numel, const uint32_t time_step)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    uint32_t edgeIndex = numel - idx;

    half2 v[4], spikes[4], last_v[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        last_v[i] = __float2half2_rn(0);
    }

    uint32_t index;
    for (int t = 0; t < time_step; t++)
    {
        index = idx + numel * t;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(v[0]) = FETCH_FLOAT4(inputs[index]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = inputs[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            v[i] = last_v[i] + v[i];
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(h_seq[index]) = FETCH_FLOAT4(v[0]);
        }
        else
        {
            auto *ptr = (half *) &v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                h_seq[index + i] = ptr[i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            spikes[i] = __hge2(v[i], v_th);
            last_v[i] = IFNode::NeuronReset<ResetFunc, half2>(v[i], spikes[i], v_reset, v_th);
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(spikes_seq[index]) = FETCH_FLOAT4(spikes[0]);
            FETCH_FLOAT4(v_seq[index]) = FETCH_FLOAT4(last_v[0]);
        }
        else
        {
            auto *spike_ptr = (half *) &spikes[0];
            auto *v_ptr = (half *) &last_v[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                spikes_seq[index + i] = spike_ptr[i];
                v_seq[index + i] = v_ptr[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- IFNode Backward FLOAT --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool detach_reset, bool padding
>
__global__ void IFNodeBPTTFLOATKernel(
        float *__restrict__ grad_spike_seq,
        float *__restrict__ grad_v_seq,
        float *__restrict__ h_seq,
        float *__restrict__ grad_x_seq,
        const float v_th, const float v_reset, const float alpha,
        const float args, const uint32_t numel, const uint32_t time_step)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 2;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 3 < numel;
    uint32_t edgeIndex = numel - idx;

    float load[4];
    float var[4], grad_v_to_h[4], grad_h[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = 0;
    }

    uint32_t index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = h_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th;     // var = over_th | load = h
            grad_v_to_h[i] = IFNode::GradVToH<GradVToHFunc, float>(var[i]);
        }

        if (surrogateFunc == SurrogateFunc::ATan)
        {
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                // 2alpha / (4 + (math.pi * alpha * x).pow_(2)) * grad_output
                // alpha_ / (4 +                   pai * x * x) * grad_output
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
            for (int i = 0; i < 4; i++) // no detach reset
            {
                grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
            }
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                load[i] = grad_spike_seq[index + i];
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
            grad_h[i] = grad_h[i] + load[i]; // grad_h[i] * grad_h_next_to_v(1.0f) + grad_v[i];
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];
            // grad_x_seq = grad_h[i] * grad_h_to_x(1.0f)
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = grad_h[i];
            }
        }
    }
}

// --- --- --- --- --- --- --- --- IFNode Backward HALF --- --- --- --- --- --- --- --- --- ---
template<
        template<typename> class GradVToHFunc, SurrogateFunc surrogateFunc,
        bool detach_reset, bool padding
>
__global__ void IFNodeBPTTHALFKernel(
        half *__restrict__ grad_spike_seq,
        half *__restrict__ grad_v_seq,
        half *__restrict__ h_seq,
        half *__restrict__ grad_x_seq,
        const half2 v_th, const half2 v_reset, const half2 alpha,
        const half2 args, const uint32_t numel, const uint32_t time_step)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx = idx << 3;
    if (idx >= numel) return;

    bool isLegalIndex = idx + 7 < numel;
    uint32_t edgeIndex = numel - idx;

    half2 load[4];
    half2 var[4], grad_v_to_h[4], grad_h[4];

#pragma unroll
    for (int i = 0; i < 4; i++)
    {
        grad_h[i] = __float2half2_rn(0);
    }

    uint32_t index;
    for (int t = time_step - 1; t >= 0; t--)
    {
        index = numel * t + idx;

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(h_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = h_seq[index + i];
            }
        }

#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            var[i] = load[i] - v_th;      // var = over_th
            grad_v_to_h[i] = IFNode::GradVToH<GradVToHFunc, half2>(var[i]);
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
            for (int i = 0; i < 4; i++) // no detach reset
            {
                grad_v_to_h[i] += (v_reset - load[i]) * var[i]; // var = grad_s_to_h
            }
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(load[0]) = FETCH_FLOAT4(grad_spike_seq[index]);
        }
        else
        {
            auto *ptr = (half *) &load[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                ptr[i] = grad_spike_seq[index + i];
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
            grad_h[i] = grad_h[i] + load[i];
            grad_h[i] = grad_h[i] * grad_v_to_h[i] + var[i];
        }

        if (!padding || isLegalIndex)
        {
            FETCH_FLOAT4(grad_x_seq[index]) = FETCH_FLOAT4(grad_h[0]);
        }
        else
        {
            auto *ptr = (half *) &grad_h[0];
            for (int i = 0; i < edgeIndex; i++)
            {
                grad_x_seq[index + i] = ptr[i];
            }
        }
    }
}


// --- --- --- --- --- --- --- --- IFNode Forward Launch --- --- --- --- --- --- --- --- ---
void IFNodeFPTTFLOATLaunch(
        float *inputs,
        float *spike_seq, float *h_seq, float *v_seq,
        const float v_th, const float v_reset, ResetType resetType,
        const uint32_t T, const uint32_t numel, const uint32_t threads)
{
    bool padding = numel % 4 != 0 ? true : false;
    uint32_t blocks = (numel / 4 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (padding)
        {
            IFNodeFPTTFLOATKernel<IFNode::HardReset, true><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th, v_reset, numel, T);
        }
        else
        {
            IFNodeFPTTFLOATKernel<IFNode::HardReset, false><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th, v_reset, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeFPTTFLOATKernel<IFNode::SoftReset, true><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th, v_reset, numel, T);
        }
        else
        {
            IFNodeFPTTFLOATKernel<IFNode::SoftReset, false><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th, v_reset, numel, T);
        }
    }
}

void IFNodeFPTTHALFLaunch(
        half *inputs,
        half *spike_seq, half *h_seq, half *v_seq,
        const float v_th, const float v_reset, ResetType resetType,
        const uint32_t T, const uint32_t numel, const uint32_t threads)
{
    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);

    bool padding = numel % 8 != 0 ? true : false;
    uint32_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset)
    {
        if (padding)
        {
            IFNodeFPTTHALFKernel<IFNode::HardReset, true><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th_half2, v_reset_half2, numel, T);
        }
        else
        {
            IFNodeFPTTHALFKernel<IFNode::HardReset, false><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th_half2, v_reset_half2, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeFPTTHALFKernel<IFNode::SoftReset, true><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th_half2, v_reset_half2, numel, T);
        }
        else
        {
            IFNodeFPTTHALFKernel<IFNode::SoftReset, false><<<blocks, threads>>>(
                    inputs, spike_seq, h_seq, v_seq, v_th_half2, v_reset_half2, numel, T);
        }
    }

}

// --- --- --- --- --- --- --- --- IFNode Backward Launch Sigmoid --- --- --- --- --- --- --- --- ---
template<SurrogateFunc surrogateFunc>
void IFNodeBPTTFLOATLaunch(
        float *grad_spike_seq, float *grad_v_seq,
        float *h_seq, float *grad_x_seq,
        const float v_th, const float v_reset, float alpha,
        ResetType resetType, const bool detach_reset,
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

    if (resetType == ResetType::HardReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, surrogateFunc, true, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, surrogateFunc, true, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
    }
    else if (resetType == ResetType::HardReset && (!detach_reset))
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, surrogateFunc, false, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHHardReset, surrogateFunc, false, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
    }
    else if (resetType == ResetType::SoftReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, surrogateFunc, true, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, surrogateFunc, true, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, surrogateFunc, false, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
        else
        {
            IFNodeBPTTFLOATKernel<IFNode::GradVToHSoftReset, surrogateFunc, false, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq, v_th, v_reset, alpha, args, numel, T);
        }
    }
}

template<SurrogateFunc surrogateFunc>
void IFNodeBPTTHALFLaunch(
        half *grad_spike_seq, half *grad_v_seq,
        half *h_seq, half *grad_x_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads)
{
    half2 v_th_half2 = __float2half2_rn(v_th);
    half2 v_reset_half2 = __float2half2_rn(v_reset);
    half2 alpha_half2 = __float2half2_rn(alpha);
    half2 args = __float2half2_rn(0);

    if (surrogateFunc == SurrogateFunc::ATan)
    {
        args = __float2half2_rn(3.14159265358979323846f * 3.14159265358979323846f * alpha * alpha);
        alpha_half2 = __float2half2_rn(2.0f * alpha);
    }

    bool padding = numel % 8 != 0 ? true : false;
    uint32_t blocks = (numel / 8 + threads - 1) / threads;

    if (resetType == ResetType::HardReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, surrogateFunc, true, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, surrogateFunc, true, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
    else if (resetType == ResetType::HardReset && (!detach_reset))
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, surrogateFunc, false, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHHardReset, surrogateFunc, false, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
    else if (resetType == ResetType::SoftReset && detach_reset)
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, surrogateFunc, true, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, surrogateFunc, true, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
    else
    {
        if (padding)
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, surrogateFunc, false, true> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
        else
        {
            IFNodeBPTTHALFKernel<IFNode::GradVToHSoftReset, surrogateFunc, false, false> <<<blocks, threads>>>(
                    grad_spike_seq, grad_v_seq, h_seq, grad_x_seq,
                    v_th_half2, v_reset_half2, alpha_half2, args, numel, T);
        }
    }
}


template void IFNodeBPTTFLOATLaunch<SurrogateFunc::Sigmoid>(
        float *, float *, float *, float *,
        const float, const float, float,
        ResetType, const bool,
        const uint32_t, const uint32_t, const uint32_t);

template void IFNodeBPTTFLOATLaunch<SurrogateFunc::ATan>(
        float *, float *, float *, float *,
        const float, const float, float,
        ResetType, const bool,
        const uint32_t, const uint32_t, const uint32_t);

template void IFNodeBPTTHALFLaunch<SurrogateFunc::Sigmoid>(
        half *, half *, half *, half *,
        const float, const float, const float,
        ResetType, const bool,
        const uint32_t, const uint32_t, const uint32_t);

template void IFNodeBPTTHALFLaunch<SurrogateFunc::ATan>(
        half *, half *, half *, half *,
        const float, const float, const float,
        ResetType, const bool,
        const uint32_t, const uint32_t, const uint32_t);



