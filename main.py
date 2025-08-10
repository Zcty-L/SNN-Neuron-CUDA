# -*- coding: utf-8 -*-
# @Time    : 2025/7/10 10:23
# @Author  : if
# @File    : main.py
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import torch.profiler
from spikingjelly.activation_based import neuron, surrogate, functional, layer
from spikingjelly.activation_based.model import sew_resnet

import neuron_cupy
import neuron_cuda


def if_test():
    device = torch.device("cuda:0")

    x = torch.randn([4, 4, 32, 80, 80], dtype=torch.float32, requires_grad=True).to(device)
    # x = torch.randn([2, 1, 2, 8, 8], dtype=torch.float32, requires_grad=True).to(device)
    x = x * 4

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()

    v_th = 1.
    v_reset = 0.
    surrogate_func = surrogate.ATan()
    detach_reset = True

    print("----------------------- Torch -----------------------")
    sn_torch = neuron.IFNode(
        v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
        step_mode="m", backend="torch", store_v_seq=False
    ).to(device)
    y_torch = sn_torch(x_torch)
    loss_torch = y_torch.mean()
    loss_torch.backward()
    print("loss torch: ", loss_torch)

    print("----------------------- CuPy -----------------------")
    sn_cupy = neuron.IFNode(
        v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
        step_mode="m", backend="cupy", store_v_seq=False
    ).to(device)
    y_cupy = sn_cupy(x_cupy)
    loss_cupy = y_cupy.mean()
    loss_cupy.backward()
    print(" loss cupy: ", loss_cupy)

    print("----------------------- CUDA -----------------------")
    sn_cuda = neuron_cupy.IFNode(
        v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
    ).to(device)
    y_cuda = sn_cuda(x_cuda)
    loss_cuda = y_cuda.mean()
    print(" loss cuda: ", loss_cuda)
    loss_cuda.backward()

    # 以torch计算为基准
    print("----------------------- CMP -----------------------")
    print("Loss torch: ", loss_torch)
    print("Loss  cupy: ", loss_cupy)
    print("Loss  cuda: ", loss_cuda)
    print()

    print("SUM<Spike Cupy - Torch>: ", (y_cupy - y_torch).sum())
    diff_mask = y_cupy != y_torch
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)

    print("SUM<Spike Cuda - Torch>: ", (y_cuda - y_torch).sum())
    diff_mask = y_cuda != y_torch
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)

    print("SUM<Spike Cuda -  Cupy>: ", (y_cuda - y_cupy).sum())
    diff_mask = y_cuda != y_cupy
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)
    print()

    print("SUM<Grad Cupy - Torch>: ", (x_cupy.grad - x_torch.grad).sum())
    print("SUM<Grad Cuda - Torch>: ", (x_cuda.grad - x_torch.grad).sum())
    print("SUM<Grad Cuda -  Cupy>: ", (x_cupy.grad - x_cuda.grad).sum())
    print(x_torch.grad[0][0][0][0][:4])
    print(x_cupy.grad[0][0][0][0][:4])
    print(x_cuda.grad[0][0][0][0][:4])
    print(x_torch.grad[:, 0, 0, 0, 0])
    print(x_cupy.grad[:, 0, 0, 0, 0])
    print(x_cuda.grad[:, 0, 0, 0, 0])
    print()

    return

    warp_up_steps = 10
    forward_steps = 30

    def save_trace_torch(prof_):
        prof_.export_chrome_trace(f"log-torch.json")

    def save_trace_cupy(prof_):
        prof_.export_chrome_trace(f"log-cupy.json")

    def save_trace_cuda(prof_):
        prof_.export_chrome_trace(f"log-cuda.json")

    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]

    # print("----------------------- Torch -----------------------")
    # with torch.profiler.profile(
    #         activities=activities,
    #         schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
    #         # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
    #         on_trace_ready=save_trace_torch,
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    #     for i in range(forward_steps + warp_up_steps + 5):
    #         functional.reset_net(sn_torch)
    #         x_torch.grad.zero_()
    #
    #         with torch.profiler.record_function("torch_f"):
    #             y = sn_torch(x_torch)
    #
    #         # with torch.profiler.record_function("torch_loss"):
    #         #     loss = y.mean()
    #         #
    #         # with torch.profiler.record_function("torch_b"):
    #         #     loss.backward()
    #
    #         prof.step()
    # # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("----------------------- CuPy-----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            # on_trace_ready=save_trace_cupy,
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            functional.reset_net(sn_cupy)
            x_cupy.grad.zero_()

            with torch.profiler.record_function("cupy_f"):
                y = sn_cupy(x_cupy)

            with torch.profiler.record_function("cupy_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cupy_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("----------------------- CUDA -----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            # on_trace_ready=save_trace_cuda,
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            x_cuda.grad.zero_()

            with torch.profiler.record_function("cuda_f"):
                y = sn_cuda(x_cuda)

            with torch.profiler.record_function("cuda_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cuda_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print(f"Max memory allocated on CUDA: {max_mem:.2f} MB")


def lif_test():
    device = torch.device("cuda:0")

    x = torch.randn([4, 4, 32, 80, 80], dtype=torch.float32, requires_grad=True).to(device)
    x = torch.randn([2, 1, 2, 8, 8], dtype=torch.float32, requires_grad=True).to(device)
    x = x * 4

    # image = cv2.imread("000000000009.jpg")
    # image = cv2.resize(image, (224, 224))
    # image_tensor = torch.tensor(image, dtype=torch.float32, requires_grad=False).to(device)
    # image_tensor = image_tensor / 255.0  # [h, w, c]
    # image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # [c, h, w] [3, 224, 224]
    # x = image_tensor.repeat(4, 4, 1, 1, 1).contiguous()  # [4, 4, 3, 224, 224]
    # x = x * 10

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()

    tau = 2.0
    decay_input = True
    v_th = 1.
    v_reset = 0.
    surrogate_func = surrogate.ATan()
    detach_reset = True

    print("----------------------- Torch -----------------------")
    sn_torch = neuron.LIFNode(
        tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
        step_mode="m", backend="torch", store_v_seq=False
    ).to(device)
    y_torch = sn_torch(x_torch)
    loss_torch = y_torch.mean()
    loss_torch.backward()
    print("loss torch: ", loss_torch)

    print("----------------------- CuPy -----------------------")
    sn_cupy = neuron.LIFNode(
        tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
        step_mode="m", backend="cupy", store_v_seq=False
    ).to(device)
    y_cupy = sn_cupy(x_cupy)
    loss_cupy = y_cupy.mean()
    loss_cupy.backward()
    print(" loss cupy: ", loss_cupy)

    print("----------------------- CUDA -----------------------")
    sn_cuda = neuron_cuda.LIFNode(
        tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
    ).to(device)
    y_cuda = sn_cuda(x_cuda)
    loss_cuda = y_cuda.mean()
    print(" loss cuda: ", loss_cuda)
    loss_cuda.backward()

    # 以torch计算为基准
    print("----------------------- CMP -----------------------")
    print("Loss torch: ", loss_torch)
    print("Loss  cupy: ", loss_cupy)
    print("Loss  cuda: ", loss_cuda)
    print()

    print("SUM<Spike Cupy - Torch>: ", (y_cupy - y_torch).sum())
    diff_mask = y_cupy != y_torch
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)

    print("SUM<Spike Cuda - Torch>: ", (y_cuda - y_torch).sum())
    diff_mask = y_cuda != y_torch
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)

    print("SUM<Spike Cuda -  Cupy>: ", (y_cuda - y_cupy).sum())
    diff_mask = y_cuda != y_cupy
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)
    print()

    print("SUM<Grad Cupy - Torch>: ", (x_cupy.grad - x_torch.grad).sum())
    print("SUM<Grad Cuda - Torch>: ", (x_cuda.grad - x_torch.grad).sum())
    print("SUM<Grad Cuda -  Cupy>: ", (x_cupy.grad - x_cuda.grad).sum())
    print(x_torch.grad[0][0][0][0][:4])
    print(x_cupy.grad[0][0][0][0][:4])
    print(x_cuda.grad[0][0][0][0][:4])
    print(x_torch.grad[:, 0, 0, 0, 0])
    print(x_cupy.grad[:, 0, 0, 0, 0])
    print(x_cuda.grad[:, 0, 0, 0, 0])
    print()

    return

    warp_up_steps = 10
    forward_steps = 30

    def save_trace_torch(prof_):
        prof_.export_chrome_trace(f"log-torch.json")

    def save_trace_cupy(prof_):
        prof_.export_chrome_trace(f"log-cupy.json")

    def save_trace_cuda(prof_):
        prof_.export_chrome_trace(f"log-cuda.json")

    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]

    # print("----------------------- Torch -----------------------")
    # with torch.profiler.profile(
    #         activities=activities,
    #         schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
    #         # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
    #         on_trace_ready=save_trace_torch,
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    #     for i in range(forward_steps + warp_up_steps + 5):
    #         functional.reset_net(sn_torch)
    #         x_torch.grad.zero_()
    #
    #         with torch.profiler.record_function("torch_f"):
    #             y = sn_torch(x_torch)
    #
    #         # with torch.profiler.record_function("torch_loss"):
    #         #     loss = y.mean()
    #         #
    #         # with torch.profiler.record_function("torch_b"):
    #         #     loss.backward()
    #
    #         prof.step()
    # # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("----------------------- CuPy-----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            # on_trace_ready=save_trace_cupy,
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            functional.reset_net(sn_cupy)
            x_cupy.grad.zero_()

            with torch.profiler.record_function("cupy_f"):
                y = sn_cupy(x_cupy)

            with torch.profiler.record_function("cupy_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cupy_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("----------------------- CUDA -----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            # on_trace_ready=save_trace_cuda,
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            x_cuda.grad.zero_()

            with torch.profiler.record_function("cuda_f"):
                y = sn_cuda(x_cuda)

            with torch.profiler.record_function("cuda_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cuda_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print(f"Max memory allocated on CUDA: {max_mem:.2f} MB")


def plif_test():
    device = torch.device("cuda:0")

    x = torch.randn([4, 4, 32, 80, 80], dtype=torch.float32, requires_grad=True).to(device)
    # x = torch.randn([2, 1, 2, 8, 8], dtype=torch.float32, requires_grad=True).to(device)
    x = x * 4

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()

    init_tau = 2.0
    decay_input = True
    v_th = 1.
    v_reset = 0.
    surrogate_func = surrogate.ATan()
    detach_reset = True

    print("----------------------- Torch -----------------------")
    sn_torch = neuron.ParametricLIFNode(
        init_tau=init_tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
        step_mode="m", backend="torch", store_v_seq=False
    ).to(device)
    y_torch = sn_torch(x_torch)
    loss_torch = y_torch.mean()
    loss_torch.backward()
    print("loss torch: ", loss_torch)

    print("----------------------- CuPy -----------------------")
    sn_cupy = neuron.ParametricLIFNode(
        init_tau=init_tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
        step_mode="m", backend="cupy", store_v_seq=False
    ).to(device)
    y_cupy = sn_cupy(x_cupy)
    loss_cupy = y_cupy.mean()
    loss_cupy.backward()
    print(" loss cupy: ", loss_cupy)

    print("----------------------- CUDA -----------------------")
    sn_cuda = neuron_cupy.ParametricLIFNode(
        init_tau=init_tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
        surrogate_function=surrogate_func, detach_reset=detach_reset,
    ).to(device)
    y_cuda = sn_cuda(x_cuda)
    loss_cuda = y_cuda.mean()
    print(" loss cuda: ", loss_cuda)
    loss_cuda.backward()

    # 以torch计算为基准
    print("----------------------- CMP -----------------------")
    print("SUM<Spike Torch>: ", y_torch.sum())
    print("SUM<Spike  Cupy>: ", y_cupy.sum())
    print("SUM<Spike  CUDA>: ", y_cuda.sum())
    print("SUM<Spike Cupy - Torch>: ", (y_cupy - y_torch).sum())
    diff_mask = y_cupy != y_torch
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)

    print("SUM<Spike Cuda - Torch>: ", (y_cuda - y_torch).sum())
    diff_mask = y_cuda != y_torch
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)

    print("SUM<Spike Cuda -  Cupy>: ", (y_cuda - y_cupy).sum())
    diff_mask = y_cuda != y_cupy
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    print(diff_indices.shape)
    print()

    print("SUM<Grad Cupy - Torch>: ", (x_cupy.grad - x_torch.grad).sum())
    print("SUM<Grad Cuda - Torch>: ", (x_cuda.grad - x_torch.grad).sum())
    print("SUM<Grad Cuda -  Cupy>: ", (x_cupy.grad - x_cuda.grad).sum())
    print(x_torch.grad[0][0][0][0][:4])
    print(x_cupy.grad[0][0][0][0][:4])
    print(x_cuda.grad[0][0][0][0][:4])
    print(x_torch.grad[:, 0, 0, 0, 0])
    print(x_cupy.grad[:, 0, 0, 0, 0])
    print(x_cuda.grad[:, 0, 0, 0, 0])
    print(sn_torch.w.grad)
    print(sn_cupy.w.grad)
    print(sn_cuda.w.grad)
    print()

    # return

    warp_up_steps = 10
    forward_steps = 100

    def save_trace_torch(prof_):
        prof_.export_chrome_trace(f"log-torch.json")

    def save_trace_cupy(prof_):
        prof_.export_chrome_trace(f"log-cupy.json")

    def save_trace_cuda(prof_):
        prof_.export_chrome_trace(f"log-cuda.json")

    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]

    print("----------------------- Torch -----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            on_trace_ready=save_trace_torch,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            functional.reset_net(sn_torch)
            x_torch.grad.zero_()

            with torch.profiler.record_function("torch_f"):
                y = sn_torch(x_torch)

            with torch.profiler.record_function("torch_loss"):
                loss = y.mean()

            with torch.profiler.record_function("torch_b"):
                loss.backward()

            prof.step()
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("----------------------- CuPy-----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            # on_trace_ready=save_trace_cupy,
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            functional.reset_net(sn_cupy)
            x_cupy.grad.zero_()

            with torch.profiler.record_function("cupy_f"):
                y = sn_cupy(x_cupy)

            with torch.profiler.record_function("cupy_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cupy_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("----------------------- CUDA -----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
            # on_trace_ready=save_trace_cuda,
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            x_cuda.grad.zero_()

            with torch.profiler.record_function("cuda_f"):
                y = sn_cuda(x_cuda)

            with torch.profiler.record_function("cuda_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cuda_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print(f"Max memory allocated on CUDA: {max_mem:.2f} MB")


def module_test():
    device = torch.device("cuda:0")

    x = torch.randn([4, 4, 16, 320, 320], dtype=torch.float32, requires_grad=True).to(device)
    # x = x * 10

    # image = cv2.imread("000000000009.jpg")
    # image = cv2.resize(image, (224, 224))
    # image_tensor = torch.tensor(image, dtype=torch.float32, requires_grad=False).to(device)
    # image_tensor = image_tensor / 255.0  # [h, w, c]
    # image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # [c, h, w] [3, 224, 224]
    # x = image_tensor.repeat(4, 4, 1, 1, 1).contiguous()  # [4, 4, 3, 224, 224]
    # x = image_tensor.repeat(1, 4, 1, 1, 1).contiguous()  # [1, 4, 3, 224, 224]
    # x = image_tensor.repeat(4, 1, 1, 1).contiguous()  # [4, 4, 3, 224, 224]
    # x = x * 10

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float16).requires_grad_()

    x_torch = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cupy = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()
    x_cuda = torch.as_tensor(x.data, device=device, dtype=torch.float32).requires_grad_()

    tau = 2.0
    decay_input = True
    v_th = 1.
    v_reset = 0.
    surrogate_func = surrogate.Sigmoid()
    detach_reset = True
    step_mode = 'm'

    print("----------------------- Torch -----------------------")
    # sn_torch = neuron.LIFNode(
    #     tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
    #     surrogate_function=surrogate_func, detach_reset=detach_reset,
    #     step_mode="m", backend="torch", store_v_seq=False
    # ).to(device)

    sn_torch = nn.Sequential(
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.LIFNode(
            tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
            surrogate_function=surrogate_func, detach_reset=detach_reset,
            step_mode=step_mode, backend="torch", store_v_seq=False
        ),
        # layer.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
    ).to(device)
    functional.set_step_mode(sn_torch, step_mode)
    # sn_torch.half()

    y_torch = sn_torch(x_torch)
    loss_torch = y_torch.mean()
    loss_torch.backward()
    print("loss torch: ", loss_torch)

    print("----------------------- CuPy -----------------------")
    # sn_cupy = neuron.LIFNode(
    #     tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
    #     surrogate_function=surrogate_func, detach_reset=detach_reset,
    #     step_mode="m", backend="cupy", store_v_seq=False
    # ).to(device)

    sn_cupy = nn.Sequential(
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.LIFNode(
            tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
            surrogate_function=surrogate_func, detach_reset=detach_reset,
            step_mode=step_mode, backend="cupy", store_v_seq=False
        ),
    ).to(device)
    functional.set_step_mode(sn_cupy, step_mode)
    # sn_cupy.half()
    sn_cupy[0].weight = sn_torch[0].weight
    # sn_cupy[1].weight = sn_torch[1].weight

    y_cupy = sn_cupy(x_cupy)
    loss_cupy = y_cupy.mean()
    loss_cupy.backward()
    print(" loss cupy: ", loss_cupy)

    print("----------------------- CUDA -----------------------")
    # sn_cuda = neuron_cupy.LIFNode(
    #     tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
    #     surrogate_function=surrogate_func, detach_reset=detach_reset,
    # ).to(device)

    sn_cuda = nn.Sequential(
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron_cuda.LIFNode(
            tau=tau, decay_input=decay_input, v_threshold=v_th, v_reset=v_reset,
            surrogate_function=surrogate_func, detach_reset=detach_reset,
        ),
    ).to(device)
    functional.set_step_mode(sn_cuda, step_mode)
    # sn_cuda.half()
    sn_cuda[0].weight = sn_torch[0].weight
    # sn_cuda[1].weight = sn_torch[1].weight

    y_cuda = sn_cuda(x_cuda)
    loss_cuda = y_cuda.mean()
    print(" loss cuda: ", loss_cuda)
    loss_cuda.backward()

    # 以torch计算为基准
    print("----------------------- CMP -----------------------")
    print("Loss torch: ", loss_torch)
    print("Loss  cupy: ", loss_cupy)
    print("Loss  cuda: ", loss_cuda)
    print()

    # print("Spike Cupy - Torch: ", (y_cupy - y_torch).max())
    # diff_mask = y_cupy != y_torch
    # diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    # print(diff_indices.shape)
    #
    # print("Spike Cuda - Torch: ", (y_cuda - y_torch).max())
    # diff_mask = y_cuda != y_torch
    # diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    # print(diff_indices.shape)
    #
    # print("Spike Cuda -  Cupy: ", (y_cuda - y_cupy).max())
    # diff_mask = y_cuda != y_cupy
    # diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    # print(diff_indices.shape)
    # print()
    #
    # print("Grad Cupy - Torch: ", (x_cupy.grad - x_torch.grad).max())
    # print("Grad Cuda - Torch: ", (x_cuda.grad - x_torch.grad).max())
    # print("Grad Cuda -  Cupy: ", (x_cupy.grad - x_cuda.grad).max())
    # print(x_torch.grad[0][0][0][0][:4])
    # print(x_cupy.grad[0][0][0][0][:4])
    # print(x_cuda.grad[0][0][0][0][:4])
    # print(x_torch.grad[:, 0, 0, 0, 0])
    # print(x_cupy.grad[:, 0, 0, 0, 0])
    # print(x_cuda.grad[:, 0, 0, 0, 0])
    print()

    warp_up_steps = 10
    forward_steps = 30

    def save_trace_torch(prof_):
        prof_.export_chrome_trace(f"log-torch.json")

    def save_trace_cupy(prof_):
        prof_.export_chrome_trace(f"log-cupy.json")

    def save_trace_cuda(prof_):
        prof_.export_chrome_trace(f"log-cuda.json")

    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
    # print("----------------------- Torch -----------------------")
    # with torch.profiler.profile(
    #         activities=activities,
    #         schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
    #         # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_cupy'),
    #         # on_trace_ready=save_trace_torch,
    #         on_trace_ready=None,
    #         record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True
    # ) as prof:
    #     for i in range(forward_steps + warp_up_steps + 5):
    #         functional.reset_net(sn_torch)
    #         x_torch.grad.zero_()
    #
    #         with torch.profiler.record_function("torch_f"):
    #             y = sn_torch(x_torch)
    #
    #         with torch.profiler.record_function("torch_loss"):
    #             loss = y.mean()
    #
    #         with torch.profiler.record_function("torch_b"):
    #             loss.backward()
    #
    #         prof.step()
    # # print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("----------------------- CuPy-----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            functional.reset_net(sn_cupy)
            x_cupy.grad.zero_()

            with torch.profiler.record_function("cupy_f"):
                y = sn_cupy(x_cupy)

            with torch.profiler.record_function("cupy_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cupy_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("----------------------- CUDA -----------------------")
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            on_trace_ready=None,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for i in range(forward_steps + warp_up_steps + 5):
            x_cuda.grad.zero_()

            with torch.profiler.record_function("cuda_f"):
                y = sn_cuda(x_cuda)

            with torch.profiler.record_function("cuda_loss"):
                loss = y.mean()

            with torch.profiler.record_function("cuda_b"):
                loss.backward()

            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    max_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print(f"Max memory allocated on CUDA: {max_mem:.2f} MB")


def test_model():
    device = torch.device("cuda:0")

    image = cv2.imread("000000000009.jpg")
    image = cv2.resize(image, (224, 224))
    image_tensor = torch.tensor(image, dtype=torch.float32, requires_grad=False).to(device)
    image_tensor = image_tensor / 255.0  # [h, w, c]
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # [c, h, w] [3, 224, 224]
    image_tensor = image_tensor.repeat(4, 4, 1, 1, 1).contiguous()  # [4, 4, 3, 224, 224]
    print(image_tensor.shape)

    warp_up_steps = 10
    forward_steps = 100
    activities = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
    profile_memory = False
    record_shapes = False

    def save_trace_torch(prof_):
        prof_.export_chrome_trace(f"log-torch.json")

    def save_trace_cupy(prof_):
        prof_.export_chrome_trace(f"log-cupy.json")

    def save_trace_cuda(prof_):
        prof_.export_chrome_trace(f"log-cuda.json")

    #
    # model = sew_resnet.sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    # model.to(device)
    # functional.set_backend(model, "torch")
    # functional.set_step_mode(model, "m")
    #
    # opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # total_time = 0
    # f_t = 0
    # b_t = 0
    # for i in range(1, forward_cnt + 1):
    #     # inputs = torch.randn([4, 4, 3, 224, 224]).to(device)
    #     inputs = torch.as_tensor(image_tensor.data, device=device, dtype=torch.float32).requires_grad_()
    #
    #     functional.reset_net(model)
    #
    #     st = time.time()
    #     outputs = model(inputs)  # [4, 4, 1000]
    #     f_t += (time.time() - st)
    #
    #     # print(outputs.shape)
    #     loss = outputs.mean()
    #     opt.zero_grad()
    #
    #     bt = time.time()
    #     loss.backward()
    #     b_t += (time.time() - bt)
    #
    #     opt.step()
    #
    #     delta_t = time.time() - st
    #     total_time += delta_t
    #
    #     if i % 10 == 0:
    #         print(
    #             f"Torch {i:4d}: | t: {delta_t:.6f} | T: {total_time:.4f} | FT: {f_t:.4f} | BT: {b_t:.4f} | "
    #             f"GPU Mem: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB"
    #         )

    # print("------------- CuPy -------------")
    # model = sew_resnet.sew_resnet18(cnf="ADD", spiking_neuron=neuron.LIFNode, detach_reset=True)
    # model.to(device)
    # functional.set_backend(model, "cupy")
    # functional.set_step_mode(model, "m")
    #
    # opt = torch.optim.SGD(model.parameters(), lr=0.01)
    # with torch.profiler.profile(
    #         activities=activities,
    #         schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
    #         on_trace_ready=None,
    #         record_shapes=record_shapes,
    #         profile_memory=profile_memory,
    #         with_stack=True
    # ) as prof:
    #     for i in range(1, forward_steps + warp_up_steps + 5 + 1):
    #         # inputs = torch.randn([4, 4, 3, 224, 224]).to(device)
    #         # inputs = torch.as_tensor(image_tensor.data, device=device, dtype=torch.float32).requires_grad_()
    #         inputs = torch.as_tensor(image_tensor.data).to(device)
    #
    #         functional.reset_net(model)
    #
    #         with torch.profiler.record_function("cupy_f"):
    #             outputs = model(inputs)  # [4, 4, 1000]
    #
    #         loss = outputs.mean()
    #         opt.zero_grad()
    #
    #         with torch.profiler.record_function("cupy_b"):
    #             loss.backward()
    #
    #         opt.step()
    #         prof.step()
    #
    #         if i % 20 == 0:
    #             print(f"CuPy {i:4d}: | GPU Mem: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("------------- CUDA -------------")
    model = sew_resnet.sew_resnet18(cnf="ADD", spiking_neuron=neuron_cuda.LIFNode, detach_reset=True)
    model.to(device)
    functional.set_backend(model, "cupy")
    functional.set_step_mode(model, "m")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=5, warmup=warp_up_steps, active=forward_steps, repeat=1),
            on_trace_ready=None,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=True
    ) as prof:
        for i in range(1, forward_steps + warp_up_steps + 5 + 1):
            # inputs = torch.randn([4, 4, 3, 224, 224]).to(device)
            # inputs = torch.as_tensor(image_tensor.data, device=device, dtype=torch.float32).requires_grad_()
            inputs = torch.as_tensor(image_tensor.data).to(device)

            with torch.profiler.record_function("cuda_f"):
                outputs = model(inputs)  # [4, 4, 1000]

            loss = outputs.mean()
            opt.zero_grad()

            with torch.profiler.record_function("cuda_b"):
                loss.backward()

            opt.step()
            prof.step()

            if i % 20 == 0:
                print(f"CUDA {i:4d}: | GPU Mem: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    if_test()
    # lif_test()
    # plif_test()
    # module_test()
    # test_model()


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()
