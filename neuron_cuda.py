# -*- coding: utf-8 -*-
# @Time    : 2025/5/12 13:25
# @Author  : if
# @File    : neuron.py
import math
import numpy as np

import torch
import torch.nn as nn

from typing import Callable
from spikingjelly.activation_based import surrogate

import spike_cuda_op

__all__ = ["IFNode", "LIFNode", "ParametricLIFNode"]


class DeviceEnvironment:
    def __init__(self, device: int):
        self.device = device
        self.previous_device = None

    def __enter__(self):
        current_device = torch.cuda.current_device()
        if current_device != self.device:
            torch.cuda.set_device(self.device)
            self.previous_device = current_device

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.previous_device is not None:
            torch.cuda.set_device(self.previous_device)


class IFNodeCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, param_args, hard_reset, detach_reset, surrogate_func_id):
        inputs = inputs.contiguous()

        spike_seq, h_seq, v_seq = spike_cuda_op.IFNodeForward(inputs, param_args, bool(hard_reset), 256)

        ctx.save_for_backward(h_seq)
        ctx.param_args = param_args
        ctx.hard_reset = hard_reset
        ctx.detach_reset = detach_reset
        ctx.surrogate_func_id = surrogate_func_id

        return spike_seq, v_seq

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        h_seq, = ctx.saved_tensors

        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()

        grad_x_seq = spike_cuda_op.IFNodeBackward(
            grad_spike_seq, grad_v_seq, h_seq, ctx.param_args,
            bool(ctx.hard_reset), bool(ctx.detach_reset), int(ctx.surrogate_func_id), 256
        )

        return grad_x_seq[0], None, None, None, None


class IFNode(nn.Module):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, store_v_seq: bool = False):
        super().__init__()
        self.v_th = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.store_v_seq = store_v_seq

        # v_th v_reset [alpha, ...]
        self.args = torch.tensor([v_threshold, 0, 0], dtype=torch.float32, requires_grad=False, device="cpu")

        self.hard_reset = True
        if v_reset is None:
            self.hard_reset = False
            self.args[1] = 0.
        else:
            self.args[1] = v_reset

        self.alpha = 0
        self.surrogate_func_id = -1
        if isinstance(surrogate_function, surrogate.Sigmoid):
            self.surrogate_func_id = 0
            self.args[2] = surrogate_function.alpha
        elif isinstance(surrogate_function, surrogate.ATan):
            self.surrogate_func_id = 1
            self.args[2] = surrogate_function.alpha
        else:
            raise f"IFNode Surrogate: Sigmoid | ATan, alpha: {self.args[2]}"

        self.sn_apply = IFNodeCUDA.apply

    def forward(self, x):
        with DeviceEnvironment(x.get_device()):
            spike_seq, v_seq = self.sn_apply(
                x, self.args, self.hard_reset, self.detach_reset, self.surrogate_func_id)

        if self.store_v_seq:
            return spike_seq, v_seq
        else:
            return spike_seq

    def extra_repr(self):
        return f"IFNode C++ CUDA:  v_threshold={self.v_th}, v_reset={self.v_reset}, " \
               f"detach_reset={self.detach_reset}, store_v_seq={self.store_v_seq}"


class LIFNodeCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, param_args, hard_reset, decay_input, detach_reset, surrogate_func_id):
        inputs = inputs.contiguous()

        spike_seq, h_seq, v_seq = spike_cuda_op.LIFNodeForward(
            inputs, param_args, bool(hard_reset), bool(decay_input), 256)

        if inputs.requires_grad:
            ctx.save_for_backward(h_seq)
            ctx.param_args = param_args
            ctx.hard_reset = hard_reset
            ctx.decay_input = decay_input
            ctx.detach_reset = detach_reset
            ctx.surrogate_func_id = surrogate_func_id

        return spike_seq, v_seq

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()

        h_seq, = ctx.saved_tensors

        grad_x_seq = spike_cuda_op.LIFNodeBackward(
            grad_spike_seq, grad_v_seq, h_seq, ctx.param_args,
            bool(ctx.hard_reset), bool(ctx.decay_input),
            bool(ctx.detach_reset), int(ctx.surrogate_func_id), 256
        )

        return grad_x_seq[0], None, None, None, None, None


class LIFNode(nn.Module):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, store_v_seq: bool = False):
        super().__init__()
        self.tau = tau
        self.decay_input = decay_input
        self.v_th = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.store_v_seq = store_v_seq

        # v_th v_reset decay [alpha, ...]
        self.args = torch.tensor([v_threshold, 0, 1. / tau, 0], dtype=torch.float32, requires_grad=False, device="cpu")

        self.hard_reset = True
        if v_reset is None:
            self.hard_reset = False
            self.args[1] = 0.
        else:
            self.args[1] = v_reset

        self.alpha = 0
        self.surrogate_func_id = -1
        if isinstance(surrogate_function, surrogate.Sigmoid):
            self.surrogate_func_id = 0
            self.args[3] = surrogate_function.alpha
        elif isinstance(surrogate_function, surrogate.ATan):
            self.surrogate_func_id = 1
            self.args[3] = surrogate_function.alpha
        else:
            raise f"Surrogate: Sigmoid | ATan, alpha: {self.args[3]}"

        self.sn_apply = LIFNodeCUDA.apply

    def forward(self, x):
        with DeviceEnvironment(x.get_device()):
            spike_seq, v_seq = self.sn_apply(
                x, self.args, self.hard_reset, self.decay_input, self.detach_reset, self.surrogate_func_id)

        if self.store_v_seq:
            return spike_seq, v_seq
        else:
            return spike_seq

    def extra_repr(self):
        return f"LIFNode C++ CUDA: tau={self.tau}, decay_input={self.decay_input}, v_threshold={self.v_th}, " \
               f"v_reset={self.v_reset}, detach_reset={self.detach_reset}, store_v_seq={self.store_v_seq}"


class PLIFNodeCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, decay, param_args, hard_reset, decay_input, detach_reset, surrogate_func_id):
        inputs = inputs.contiguous()

        spike_seq, h_seq, v_seq = spike_cuda_op.PLIFNodeForward(
            inputs, param_args, bool(hard_reset), bool(decay_input), 256)

        ctx.save_for_backward(h_seq, v_seq)
        ctx.param_args = param_args
        ctx.hard_reset = hard_reset
        ctx.decay_input = decay_input
        ctx.detach_reset = detach_reset
        ctx.surrogate_func_id = surrogate_func_id

        return spike_seq, v_seq

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_seq):
        h_seq, v_seq = ctx.saved_tensors

        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()

        grad = spike_cuda_op.PLIFNodeBackward(
            grad_spike_seq, grad_v_seq, h_seq, v_seq, ctx.param_args,
            bool(ctx.hard_reset), bool(ctx.decay_input),
            bool(ctx.detach_reset), int(ctx.surrogate_func_id), 256
        )

        return grad[0], grad[1], None, None, None, None, None


class ParametricLIFNode(nn.Module):
    def __init__(self, init_tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, store_v_seq: bool = False):
        super().__init__()
        assert isinstance(init_tau, float) and init_tau > 1.

        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w, dtype=torch.float32), requires_grad=True)

        self.decay_input = decay_input
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.store_v_seq = store_v_seq

        # v_th v_reset decay [alpha, ...]
        self.args = torch.tensor([v_threshold, 0, self.w.sigmoid().detach().cpu().item(), 0],
                                 dtype=torch.float32, requires_grad=False, device="cpu")

        self.hard_reset = True
        if v_reset is None:
            self.hard_reset = False
            self.args[1] = 0.
        else:
            self.args[1] = v_reset

        self.alpha = 0
        self.surrogate_func_id = -1
        if isinstance(surrogate_function, surrogate.Sigmoid):
            self.surrogate_func_id = 0
            self.args[3] = surrogate_function.alpha
        elif isinstance(surrogate_function, surrogate.ATan):
            self.surrogate_func_id = 1
            self.args[3] = surrogate_function.alpha
        else:
            raise f"Surrogate: Sigmoid | ATan, alpha: {self.args[3]}"

        self.sn_apply = PLIFNodeCUDA.apply

    def forward(self, x):
        with DeviceEnvironment(x.get_device()):
            decay = self.w.sigmoid()
            self.args[2] = decay.detach().cpu().item()

            spike_seq, v_seq = self.sn_apply(
                x, decay, self.args, self.hard_reset, self.decay_input,
                self.detach_reset, self.surrogate_func_id)

        if self.store_v_seq:
            return spike_seq, v_seq
        else:
            return spike_seq

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()

        return f"ParametricLIFNode C++ CUDA: tau={tau}, decay_input={self.decay_input}, v_threshold={self.v_th}, " \
               f"v_reset={self.v_reset}, detach_reset={self.detach_reset}, store_v_seq={self.store_v_seq}"
