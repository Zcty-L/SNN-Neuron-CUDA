#pragma once

#include <torch/torch.h>
#include <c10/util/Half.h>

#include "neuron.h"

// --- --- --- --- --- --- --- --- IFNode Forward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> IFNodeFPTTFLOATTorchImpl(
        const torch::Tensor &inputs, const float v_th, const float v_reset,
        ResetType resetType, const int threads)
{
    const uint32_t T = inputs.size(0);         // [T, ...]
    const uint32_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());

    auto *inputs_ptr = inputs.data_ptr<float>();
    auto *spike_seq_ptr = spike_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *v_seq_ptr = v_seq.data_ptr<float>();

    IFNodeFPTTFLOATLaunch(
            inputs_ptr, spike_seq_ptr, h_seq_ptr, v_seq_ptr,
            v_th, v_reset, resetType, T, numel, threads
    );

    return {spike_seq, h_seq, v_seq};
}

std::vector<torch::Tensor> IFNodeFPTTHALFTorchImpl(
        const torch::Tensor &inputs, const float v_th, const float v_reset,
        ResetType resetType, const int threads)
{
    const uint32_t T = inputs.size(0);         // [T, ...]
    const uint32_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());

    auto *inputs_ptr = inputs.data_ptr<at::Half>();
    auto *spike_seq_ptr = spike_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();

    IFNodeFPTTHALFLaunch(
            reinterpret_cast<half *>(inputs_ptr),
            reinterpret_cast<half *>(spike_seq_ptr),
            reinterpret_cast<half *>(h_seq_ptr),
            reinterpret_cast<half *>(v_seq_ptr),
            v_th, v_reset, resetType, T, numel, threads
    );

    return {spike_seq, h_seq, v_seq};
}

// --- --- --- --- --- --- --- --- IFNode Backward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> IFNodeBPTTFLOATTorchImpl(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &grad_v_seq, const torch::Tensor &h_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset,
        SurrogateFunc surrogateFunc, const int threads)
{
    const uint32_t T = grad_spike_seq.size(0);         // [T, ...]
    const uint32_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();

    switch (surrogateFunc)
    {
        case SurrogateFunc::ATan:
        {
            IFNodeBPTTFLOATLaunch<SurrogateFunc::ATan>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr,
                    v_th, v_reset, alpha, resetType, detach_reset,
                    T, numel, threads
            );
            break;
        }
        default:
        {
            IFNodeBPTTFLOATLaunch<SurrogateFunc::Sigmoid>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr,
                    v_th, v_reset, alpha, resetType, detach_reset,
                    T, numel, threads
            );
            break;
        }
    }

    return {grad_x_seq};
}

std::vector<torch::Tensor> IFNodeBPTTHALFTorchImpl(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &grad_v_seq, const torch::Tensor &h_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset,
        SurrogateFunc surrogateFunc, const int threads)
{
    const uint32_t T = grad_spike_seq.size(0);         // [T, ...]
    const uint32_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();

    switch (surrogateFunc)
    {
        case SurrogateFunc::ATan:
        {
            IFNodeBPTTHALFLaunch<SurrogateFunc::ATan>(
                    reinterpret_cast<half *>(grad_spike_ptr),
                    reinterpret_cast<half *>(grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr),
                    reinterpret_cast<half *>(grad_x_ptr),
                    v_th, v_reset, alpha, resetType, detach_reset,
                    T, numel, threads
            );
            break;
        }
        default:
        {
            IFNodeBPTTHALFLaunch<SurrogateFunc::Sigmoid>(
                    reinterpret_cast<half *>(grad_spike_ptr),
                    reinterpret_cast<half *>(grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr),
                    reinterpret_cast<half *>(grad_x_ptr),
                    v_th, v_reset, alpha, resetType, detach_reset,
                    T, numel, threads
            );
            break;
        }
    }

    return {grad_x_seq};
}

// --- --- --- --- --- --- --- --- LIFNode Forward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> LIFNodeFPTTFLOATTorchImpl(
        const torch::Tensor &inputs, const float v_th, const float v_reset, const float decay,
        ResetType resetType, bool decay_input, const int threads)
{
    const uint32_t T = inputs.size(0);         // [T, ...]
    const uint32_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());

    auto *inputs_ptr = inputs.data_ptr<float>();
    auto *spike_seq_ptr = spike_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *v_seq_ptr = v_seq.data_ptr<float>();

    LIFNodeFPTTFLOATLaunch(
            inputs_ptr, spike_seq_ptr, h_seq_ptr, v_seq_ptr,
            v_th, v_reset, decay, resetType, decay_input, T, numel, threads
    );

    return {spike_seq, h_seq, v_seq};
}

std::vector<torch::Tensor> LIFNodeFPTTHALFTorchImpl(
        const torch::Tensor &inputs, const float v_th, const float v_reset, const float decay,
        ResetType resetType, bool decay_input, const int threads)
{
    const uint32_t T = inputs.size(0);         // [T, ...]
    const uint32_t numel = inputs.numel() / T;

    torch::Tensor spike_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor h_seq = torch::empty(inputs.sizes(), inputs.options());
    torch::Tensor v_seq = torch::empty(inputs.sizes(), inputs.options());

    auto *inputs_ptr = inputs.data_ptr<at::Half>();
    auto *spike_seq_ptr = spike_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();

    LIFNodeFPTTHALFLaunch(
            reinterpret_cast<half *>(inputs_ptr),
            reinterpret_cast<half *>(spike_seq_ptr),
            reinterpret_cast<half *>(h_seq_ptr),
            reinterpret_cast<half *>(v_seq_ptr),
            v_th, v_reset, decay, resetType, decay_input, T, numel, threads
    );

    return {spike_seq, h_seq, v_seq};
}


// --- --- --- --- --- --- --- --- LIFNode Backward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> LIFNodeBPTTFLOATTorchImpl(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &grad_v_seq, const torch::Tensor &h_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        SurrogateFunc surrogateFunc, const int threads)
{
    const uint32_t T = grad_spike_seq.size(0);         // [T, ...]
    const uint32_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();

    switch (surrogateFunc)
    {
        case SurrogateFunc::ATan:
        {
            LIFNodeBPTTFLOATLaunch<SurrogateFunc::ATan>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr,
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
        default:
        {
            LIFNodeBPTTFLOATLaunch<SurrogateFunc::Sigmoid>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, grad_x_ptr,
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
    }

    return {grad_x_seq};
}

std::vector<torch::Tensor> LIFNodeBPTTHALFTorchImpl(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &grad_v_seq, const torch::Tensor &h_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        SurrogateFunc surrogateFunc, const int threads)
{
    const uint32_t T = grad_spike_seq.size(0);         // [T, ...]
    const uint32_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();

    switch (surrogateFunc)
    {
        case SurrogateFunc::ATan:
        {
            LIFNodeBPTTHALFLaunch<SurrogateFunc::ATan>(
                    reinterpret_cast<half *>(grad_spike_ptr),
                    reinterpret_cast<half *>(grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr),
                    reinterpret_cast<half *>(grad_x_ptr),
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
        default:
        {
            LIFNodeBPTTHALFLaunch<SurrogateFunc::Sigmoid>(
                    reinterpret_cast<half *>(grad_spike_ptr),
                    reinterpret_cast<half *>(grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr),
                    reinterpret_cast<half *>(grad_x_ptr),
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
    }

    return {grad_x_seq};
}


// --- --- --- --- --- --- --- --- PLIFNode Backward Launch --- --- --- --- --- --- --- --- ---
std::vector<torch::Tensor> PLIFNodeBPTTFLOATTorchImpl(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &h_seq, const torch::Tensor &v_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        SurrogateFunc surrogateFunc, const int threads)
{
    const uint32_t T = grad_spike_seq.size(0);         // [T, ....]
    const uint32_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());
    torch::Tensor grad_tau_seq = torch::zeros({1}, grad_spike_seq.options());

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<float>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<float>();
    auto *h_seq_ptr = h_seq.data_ptr<float>();
    auto *v_seq_ptr = v_seq.data_ptr<float>();

    auto *grad_x_ptr = grad_x_seq.data_ptr<float>();
    auto *grad_tau_ptr = grad_tau_seq.data_ptr<float>();

    switch (surrogateFunc)
    {
        case SurrogateFunc::ATan:
        {
            PLIFNodeBPTTFLOATLaunch<SurrogateFunc::ATan>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
        default:
        {
            PLIFNodeBPTTFLOATLaunch<SurrogateFunc::Sigmoid>(
                    grad_spike_ptr, grad_v_ptr, h_seq_ptr, v_seq_ptr, grad_x_ptr, grad_tau_ptr,
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
    }

    return {grad_x_seq, grad_tau_seq};
}

std::vector<torch::Tensor> PLIFNodeBPTTHALFTorchImpl(
        const torch::Tensor &grad_spike_seq, const torch::Tensor &grad_v_seq,
        const torch::Tensor &h_seq, const torch::Tensor &v_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        SurrogateFunc surrogateFunc, const int threads)
{
    const uint32_t T = grad_spike_seq.size(0);         // [T, ...]
    const uint32_t numel = grad_spike_seq.numel() / T;

    torch::Tensor grad_x_seq = torch::empty(grad_spike_seq.sizes(), grad_spike_seq.options());
    torch::Tensor grad_tau_seq = torch::zeros({1}, grad_spike_seq.options().dtype(torch::kFloat32));

    auto *grad_spike_ptr = grad_spike_seq.data_ptr<at::Half>();
    auto *grad_v_ptr = grad_v_seq.data_ptr<at::Half>();
    auto *h_seq_ptr = h_seq.data_ptr<at::Half>();
    auto *v_seq_ptr = v_seq.data_ptr<at::Half>();

    auto *grad_x_ptr = grad_x_seq.data_ptr<at::Half>();
    auto *grad_tau = grad_tau_seq.data_ptr<float>();

    switch (surrogateFunc)
    {
        case SurrogateFunc::ATan:
        {
            PLIFNodeBPTTHALFLaunch<SurrogateFunc::ATan>(
                    reinterpret_cast<half *>(grad_spike_ptr),
                    reinterpret_cast<half *>(grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr),
                    reinterpret_cast<half *>(grad_x_ptr), grad_tau,
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
        default:
        {
            PLIFNodeBPTTHALFLaunch<SurrogateFunc::Sigmoid>(
                    reinterpret_cast<half *>(grad_spike_ptr),
                    reinterpret_cast<half *>(grad_v_ptr),
                    reinterpret_cast<half *>(h_seq_ptr),
                    reinterpret_cast<half *>(v_seq_ptr),
                    reinterpret_cast<half *>(grad_x_ptr), grad_tau,
                    v_th, v_reset, decay, alpha, resetType, decay_input, detach_reset,
                    T, numel, threads
            );
            break;
        }
    }

    return {grad_x_seq, grad_tau_seq};
}
