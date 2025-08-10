#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <string>

#include <cuda_fp16.h>


#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

enum class SurrogateFunc : int
{
    Sigmoid = 0,
    ATan = 1,
};

enum class NeuronType : int
{
    IF = 0,
    LIF = 1,
    PLIF = 2,
};

enum class ResetType : int
{
    SoftReset = 0,
    HardReset = 1,
};

// --- --- --- --- --- --- --- --- IFNode --- --- --- --- --- --- --- --- --- ---
void IFNodeFPTTFLOATLaunch(
        float *inputs,
        float *spike_seq, float *h_seq, float *v_seq,
        const float v_th, const float v_reset, ResetType resetType,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

void IFNodeFPTTHALFLaunch(
        half *inputs,
        half *spike_seq, half *h_seq, half *v_seq,
        const float v_th, const float v_reset, ResetType resetType,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

template<SurrogateFunc surrogateFunc>
void IFNodeBPTTFLOATLaunch(
        float *grad_spike_seq, float *grad_v_seq,
        float *h_seq, float *grad_x_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

template<SurrogateFunc surrogateFunc>
void IFNodeBPTTHALFLaunch(
        half *grad_spike_seq, half *grad_v_seq,
        half *h_seq, half *grad_x_seq,
        const float v_th, const float v_reset, const float alpha,
        ResetType resetType, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads);


// --- --- --- --- --- --- --- --- LIFNode --- --- --- --- --- --- --- --- --- ---
void LIFNodeFPTTFLOATLaunch(
        float *inputs,
        float *spike_seq, float *h_seq, float *v_seq,
        const float v_th, const float v_reset, const float decay,
        ResetType resetType, bool decay_input,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

void LIFNodeFPTTHALFLaunch(
        half *inputs,
        half *spike_seq, half *h_seq, half *v_seq,
        const float v_th, const float v_reset, const float decay,
        ResetType resetType, bool decay_input,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

template<SurrogateFunc surrogateFunc>
void LIFNodeBPTTFLOATLaunch(
        float *grad_spike_seq, float *grad_v_seq,
        float *h_seq, float *grad_x_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

template<SurrogateFunc surrogateFunc>
void LIFNodeBPTTHALFLaunch(
        half *grad_spike_seq, half *grad_v_seq,
        half *h_seq, half *grad_x_seq,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads);


// --- --- --- --- --- --- --- --- PLIFNode --- --- --- --- --- --- --- --- --- ---
template<SurrogateFunc surrogateFunc>
void PLIFNodeBPTTFLOATLaunch(
        float *grad_spike_seq, float *grad_v_seq,
        float *h_seq, float *v_seq,
        float *grad_x_seq, float *grad_tau,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads);

template<SurrogateFunc surrogateFunc>
void PLIFNodeBPTTHALFLaunch(
        half *grad_spike_seq, half *grad_v_seq,
        half *h_seq, half *v_seq,
        half *grad_x_seq, float *grad_tau,
        const float v_th, const float v_reset, const float decay, const float alpha,
        ResetType resetType, const bool decay_input, const bool detach_reset,
        const uint32_t T, const uint32_t numel, const uint32_t threads);


#endif
