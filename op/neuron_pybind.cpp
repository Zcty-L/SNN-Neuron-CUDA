#include <torch/extension.h>
#include "neuron/neuron_torch_api.hpp"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor (C++)")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous (C++)")
#define CHECK_FP16(x) TORCH_CHECK(x.scalar_type() == torch::kHalf, #x "must be CUDA and of type Half (C++)")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> IFNodeForward(
        const torch::Tensor &inputs, const torch::Tensor &args, bool hard_reset, const int threads = 256)
{
    CHECK_INPUT(inputs)

    // v_th v_reset
    auto *data_ptr = args.data_ptr<float>();

    if (inputs.options().dtype() == torch::kFloat16)
    {
        return IFNodeFPTTHALFTorchImpl(inputs, data_ptr[0], data_ptr[1], (ResetType) hard_reset, threads);
    }
    else
    {
        return IFNodeFPTTFLOATTorchImpl(inputs, data_ptr[0], data_ptr[1], (ResetType) hard_reset, threads);
    }
}

std::vector<torch::Tensor> IFNodeBackward(
        const torch::Tensor &grad_spike_seq,
        const torch::Tensor &grad_v_seq,
        const torch::Tensor &h_seq,
        const torch::Tensor &args,
        int func_id, bool hard_reset, bool detach_reset, int threads = 256)
{
    CHECK_INPUT(grad_spike_seq)

    auto surrogateFunc = (SurrogateFunc) func_id;
    auto *data_ptr = args.data_ptr<float>();

    if (grad_spike_seq.options().dtype() == torch::kFloat16)
    {
        return IFNodeBPTTHALFTorchImpl(
                grad_spike_seq, grad_v_seq, h_seq, data_ptr[0], data_ptr[1], data_ptr[2],
                (ResetType) hard_reset, detach_reset, surrogateFunc, threads);
    }
    else
    {
        return IFNodeBPTTFLOATTorchImpl(
                grad_spike_seq, grad_v_seq, h_seq, data_ptr[0], data_ptr[1], data_ptr[2],
                (ResetType) hard_reset, detach_reset, surrogateFunc, threads);
    }
}


std::vector<torch::Tensor> LIFNodeForward(
        const torch::Tensor &inputs, const torch::Tensor &args,
        bool hard_reset, bool decay_input, const int threads = 256)
{
    CHECK_INPUT(inputs)

    // v_th v_reset tau [alpha, ...]
    auto *data_ptr = args.data_ptr<float>();

    if (inputs.options().dtype() == torch::kFloat16)
    {
        return LIFNodeFPTTHALFTorchImpl(
                inputs, data_ptr[0], data_ptr[1], data_ptr[2], (ResetType) hard_reset, decay_input, threads);
    }
    else
    {
        return LIFNodeFPTTFLOATTorchImpl(
                inputs, data_ptr[0], data_ptr[1], data_ptr[2], (ResetType) hard_reset, decay_input, threads);
    }
}

std::vector<torch::Tensor> LIFNodeBackward(
        const torch::Tensor &grad_spike_seq,
        const torch::Tensor &grad_v_seq,
        const torch::Tensor &h_seq,
        const torch::Tensor &args,
        bool hard_reset, bool decay_input, bool detach_reset, int func_id, int threads = 256)
{
    CHECK_INPUT(grad_spike_seq)

    auto surrogateFunc = (SurrogateFunc) func_id;
    auto *data_ptr = args.data_ptr<float>();

    if (grad_spike_seq.options().dtype() == torch::kFloat16)
    {
        return LIFNodeBPTTHALFTorchImpl(
                grad_spike_seq, grad_v_seq, h_seq, data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3],
                (ResetType) hard_reset, decay_input, detach_reset, surrogateFunc, threads);
    }
    else
    {
        return LIFNodeBPTTFLOATTorchImpl(
                grad_spike_seq, grad_v_seq, h_seq, data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3],
                (ResetType) hard_reset, decay_input, detach_reset, surrogateFunc, threads);
    }
}


std::vector<torch::Tensor> PLIFNodeForward(
        const torch::Tensor &inputs, const torch::Tensor &args,
        bool hard_reset, bool decay_input, const int threads = 256)
{
    return LIFNodeForward(inputs, args, hard_reset, decay_input, threads);
}

std::vector<torch::Tensor> PLIFNodeBackward(
        const torch::Tensor &grad_spike_seq,
        const torch::Tensor &grad_v_seq,
        const torch::Tensor &h_seq,
        const torch::Tensor &v_seq,
        const torch::Tensor &args,
        bool hard_reset, bool decay_input, bool detach_reset, int func_id, int threads = 256)
{
    CHECK_INPUT(grad_spike_seq)

    auto surrogateFunc = (SurrogateFunc) func_id;
    auto *data_ptr = args.data_ptr<float>();

    if (grad_spike_seq.options().dtype() == torch::kFloat16)
    {
        return PLIFNodeBPTTHALFTorchImpl(
                grad_spike_seq, grad_v_seq, h_seq, v_seq, data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3],
                (ResetType) hard_reset, decay_input, detach_reset, surrogateFunc, threads);
    }
    else
    {
        return PLIFNodeBPTTFLOATTorchImpl(
                grad_spike_seq, grad_v_seq, h_seq, v_seq, data_ptr[0], data_ptr[1], data_ptr[2], data_ptr[3],
                (ResetType) hard_reset, decay_input, detach_reset, surrogateFunc, threads);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("IFNodeForward", &IFNodeForward);
    m.def("IFNodeBackward", &IFNodeBackward);

    m.def("LIFNodeForward", &LIFNodeForward);
    m.def("LIFNodeBackward", &LIFNodeBackward);

    m.def("PLIFNodeForward", &PLIFNodeForward);
    m.def("PLIFNodeBackward", &PLIFNodeBackward);
}
