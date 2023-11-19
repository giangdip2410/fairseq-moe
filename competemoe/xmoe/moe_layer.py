# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module, ModuleList
import random
from .global_groups import get_all2all_group, get_moe_group
import torch.nn.functional as F

try:
    from fairseq.modules.moe import MOELayer

    has_fairseq = True
    Base = MOELayer
except ModuleNotFoundError:
    Base = Module
    has_fairseq = False

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except ModuleNotFoundError:
    has_tutel, fused_cumsum_sub_one = False, lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)


# def kl_loss_sym(logits1, logits2):

#     kl_loss = nn.KLDivLoss(reduction="sum")
    
#     loss = kl_loss(F.log_softmax(logits1, dim=1), F.softmax(logits2, dim=1)) + kl_loss(F.log_softmax(logits2, dim=1), F.softmax(logits1, dim=1))
#     print("Compete Loss: ", loss.detach().cpu().numpy())
#     return loss.mean(-1)

def kl_divergence(softmax_1, softmax_2):
    kl_divergence = nn.KLDivLoss(reduction='mean').cuda()
    compete_loss = kl_divergence(softmax_1, softmax_2)
    print("Compete Loss: ", compete_loss.detach().cpu().numpy())
    return compete_loss





# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))




class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate, experts, args):
        if has_fairseq:
            super(Base, self).__init__()
        else:
            super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        _, self.expert_group = get_moe_group(args.moe_expert_count)
        self.all2all_group = get_all2all_group(args.moe_expert_count)
        self.world_size = dist.get_world_size(group=self.expert_group)
        self.all2all_size = dist.get_world_size(group=self.all2all_group)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.num_local_experts = len(self.experts)
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert (
            len(input.shape) == 3
        ), "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert (
                len(input_padding_mask.shape) == 2
            ), "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        expected_bsz = (
            getattr(self.args, "batch_size", 0)
            if self.training
            else getattr(self.args, "batch_size_valid", 0)
        )
        # This indicates that --batch-size or --max-sentences is not specified
        if expected_bsz is None:
            expected_bsz = 0
        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        if (
            not self.in_generation
            and expected_bsz != 0
            and input_shape[0] != expected_bsz
        ):
            logger.warning(
                f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
            )
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: input_shape[0], :, :] = input
            input = padded_input

            padded_input_padding_mask = torch.ones(
                (
                    expected_bsz,
                    input_shape[1],
                ),
                dtype=torch.bool,
                device=input.device,
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[: input_shape[0], :] = input_padding_mask
            else:
                padded_input_padding_mask[: input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask

        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        with torch.no_grad():
            reshaped_input_padding_mask = (
                input_padding_mask.reshape(-1) if input_padding_mask is not None else None
            )

            # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
            # Pro of --max-tokens: more flexible for MT variable sequence lengths
            # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
            if expected_bsz == 0:
                expected_dim = reshaped_input_shape[0] * torch.ones(
                    (1,), dtype=torch.long, device=input.device
                )
                dist.all_reduce(expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX)
                expected_dim = int(expected_dim.item())
                padded_input = torch.zeros(
                    (expected_dim, reshaped_input_shape[1]),
                    dtype=input.dtype,
                    layout=input.layout,
                    device=input.device,
                )
                padded_input[: reshaped_input_shape[0], :] = reshaped_input
                reshaped_input = padded_input

                padded_input_padding_mask = torch.ones(
                    (expected_dim,), dtype=torch.bool, device=padded_input.device
                )
                if reshaped_input_padding_mask is not None:
                    padded_input_padding_mask[
                        : reshaped_input_shape[0]
                    ] = reshaped_input_padding_mask
                else:
                    padded_input_padding_mask[: reshaped_input_shape[0]] = False
                reshaped_input_padding_mask = padded_input_padding_mask

            # if has_tutel:
            #     l_aux, self.metadata, C, E, indices_, locations_, gates_, gates = self.gate(
            #         reshaped_input, reshaped_input_padding_mask
            #     )
            #     S, M = reshaped_input.size(0), reshaped_input.size(1)

            #     if not hasattr(self, "_tutel_dispatcher"):
            #         self._tutel_dispatcher = tutel_moe.fast_dispatcher(
            #             E, C, M, dispatch_dtype=reshaped_input.dtype
            #         )
            #     self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            #     dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
            # else:
            #     l_aux, combine_weights, dispatch_mask, self.metadata, gates = self.gate(
            #         reshaped_input, reshaped_input_padding_mask
            #     )

            #     dispatch_mask = dispatch_mask.to(input.dtype).permute(
            #         1, 2, 0
            #     )  # S,E,C -> E,C,S
            #     E, C, S = dispatch_mask.size()
            #     M = reshaped_input.size(1)
            #     assert reshaped_input.size() == (S, M)
            #     # einsum("sec,sm->ecm")
            #     # dispatched_input = torch.mm(
            #     #     dispatch_mask.view(E * C, S), reshaped_input
            #     # )  # -> (E*C),M
            dispatched_input = reshaped_input

            if self.all2all_size > 1:
                dispatched_input = self.all_to_all_wrapper(dispatched_input)

            # Re-shape after all-to-all: ecm -> gecm
            dispatched_input = dispatched_input.reshape(
                self.all2all_size, self.num_local_experts, -1, d_model
            )
            chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs += [expert(chunk)]
            expert_output = torch.cat(expert_outputs, dim=1)
            ip_shape = reshaped_input_shape[0] * reshaped_input_shape[1]
            tt_exp_shape = expert_output.shape[0] * expert_output.shape[1] * expert_output.shape[2] * expert_output.shape[3]
            if tt_exp_shape % ip_shape == 0:
                expert_output_opt = expert_output.reshape(reshaped_input_shape[0], reshaped_input_shape[1], -1)
            else:
                v = tt_exp_shape // ip_shape
                expert_output_opt = torch.flatten(expert_output)[:v*ip_shape].reshape(reshaped_input_shape[0], reshaped_input_shape[1], -1)
            expert_output_opt = torch.norm(expert_output_opt, dim=2)
            #calculate gate
            # Reshape into S tokens by dropping sequence dimension.
            reshaped_input = expert_output_opt.reshape(-1, d_model)
            reshaped_input_shape = reshaped_input.shape
            reshaped_input_padding_mask = (
                input_padding_mask.reshape(-1) if input_padding_mask is not None else None
            )

            # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
            # Pro of --max-tokens: more flexible for MT variable sequence lengths
            # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
            if expected_bsz == 0:
                expected_dim = reshaped_input_shape[0] * torch.ones(
                    (1,), dtype=torch.long, device=input.device
                )
                dist.all_reduce(expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX)
                expected_dim = int(expected_dim.item())
                padded_input = torch.zeros(
                    (expected_dim, reshaped_input_shape[1]),
                    dtype=input.dtype,
                    layout=input.layout,
                    device=input.device,
                )
                padded_input[: reshaped_input_shape[0], :] = reshaped_input
                reshaped_input = padded_input

                padded_input_padding_mask = torch.ones(
                    (expected_dim,), dtype=torch.bool, device=padded_input.device
                )
                if reshaped_input_padding_mask is not None:
                    padded_input_padding_mask[
                        : reshaped_input_shape[0]
                    ] = reshaped_input_padding_mask
                else:
                    padded_input_padding_mask[: reshaped_input_shape[0]] = False
                reshaped_input_padding_mask = padded_input_padding_mask

            if has_tutel:
                l_aux, self.metadata, C, E, indices_, locations_, gates_, gate_opt = self.gate(
                    reshaped_input, reshaped_input_padding_mask
                )

            else:
                l_aux, combine_weights, dispatch_mask, self.metadata, gate_opt = self.gate(
                    reshaped_input, reshaped_input_padding_mask
                )
        #normal operation
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = (
            input_padding_mask.reshape(-1) if input_padding_mask is not None else None
        )

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        if expected_bsz == 0:
            expected_dim = reshaped_input_shape[0] * torch.ones(
                (1,), dtype=torch.long, device=input.device
            )
            dist.all_reduce(expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX)
            expected_dim = int(expected_dim.item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            padded_input[: reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[
                    : reshaped_input_shape[0]
                ] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[: reshaped_input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask

        if has_tutel:
            l_aux, self.metadata, C, E, indices_, locations_, gates_, gates = self.gate(
                reshaped_input, reshaped_input_padding_mask
            )
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, "_tutel_dispatcher"):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(
                    E, C, M, dispatch_dtype=reshaped_input.dtype
                )
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            l_aux, combine_weights, dispatch_mask, self.metadata, gates = self.gate(
                reshaped_input, reshaped_input_padding_mask
            )

            dispatch_mask = dispatch_mask.to(input.dtype).permute(
                1, 2, 0
            )  # S,E,C -> E,C,S
            E, C, S = dispatch_mask.size()
            M = reshaped_input.size(1)
            assert reshaped_input.size() == (S, M)
            # einsum("sec,sm->ecm")
            dispatched_input = torch.mm(
                dispatch_mask.view(E * C, S), reshaped_input
            )  # -> (E*C),M

        #optimal losscalculate loss
        compete_loss = kl_divergence(gates, gate_opt)
        # combine with gate loss
        l_aux += compete_loss

        # norm flow
        if self.all2all_size > 1:
            dispatched_input = self.all_to_all_wrapper(dispatched_input)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(
            self.all2all_size, self.num_local_experts, -1, d_model
        )
        chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs += [expert(chunk)]
        expert_output = torch.cat(expert_outputs, dim=1)

        if self.all2all_size > 1:
            expert_output = self.all_to_all_wrapper(expert_output)

        # Re-shape back: gecm -> ecm
        expert_output = expert_output.reshape(
            self.all2all_size * self.num_local_experts, -1, d_model
        )

        if has_tutel:
            combined_output = self._tutel_dispatcher.decode(
                expert_output.view(E * C, M)
            )
        else:
            # einsum("sec,ecm->sm")
            combined_output = combine_weights.view(S, E * C).mm(
                expert_output.view(E * C, M)
            )

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[: reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[: input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, l_aux

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, "dummy_a2a", False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += cpu_end - cpu_start
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, "record_a2a_perf_stats", False)
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []


# # einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# # See https://arxiv.org/pdf/2006.16668.pdf for details.

# # Based on https://github.com/pytorch/pytorch/pull/40762
# class _AllToAll(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
#         ctx.group = group
#         input = input.contiguous()
#         output = torch.empty_like(input)
#         if torch.distributed.is_initialized():
#             dist.all_to_all_single(output, input, group=group)
#         else:
#             assert group is None
#             output = input
#         return output

#     @staticmethod
#     def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
#         return (None, _AllToAll.apply(ctx.group, *grad_output))




# class MOELayer(Base):
#     """MOELayer module which implements MixtureOfExperts as described in Gshard_.
#     ::

#         gate = Top2Gate(model_dim, num_experts)
#         moe = MOELayer(gate, expert)
#         output = moe(input)
#         l_aux = moe.l_aux

#     .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

#     Args:
#         gate (torch.nn.Module):
#             gate network
#         expert (torch.nn.Module):
#             expert network
#     """

#     def __init__(self, gate, experts, args):
#         if has_fairseq:
#             super(Base, self).__init__()
#         else:
#             super().__init__()
#         self.gate = gate
#         if type(experts) == ModuleList:
#             self.experts = cast(ModuleList, experts)
#         else:
#             self.experts = ModuleList([experts])
#         _, self.expert_group = get_moe_group(args.moe_expert_count)
#         self.all2all_group = get_all2all_group(args.moe_expert_count)
#         self.world_size = dist.get_world_size(group=self.expert_group)
#         self.all2all_size = dist.get_world_size(group=self.all2all_group)
#         for p in experts.parameters():
#             p.expert = True  # type: ignore
#         self.num_local_experts = len(self.experts)
#         self.args = args
#         self.in_generation = False
#         self.a2a_cuda_event_intervals = []
#         self.a2a_cpu_time_ms = 0.0

#     def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
#         assert len(input) == 1, "only single input Tensor supported"
#         input = input[0]
#         assert (
#             len(input.shape) == 3
#         ), "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
#         if input_padding_mask is not None:
#             assert (
#                 len(input_padding_mask.shape) == 2
#             ), "input Tensor must have dimensions: (s)equence, (t)oken"
#             assert input_padding_mask.shape[0] == input.shape[0]
#             assert input_padding_mask.shape[1] == input.shape[1]
#         # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

#         # Implement Algorithm 2 from GShard paper.
#         d_model = input.shape[2]
#         # Pad to expected batch size
#         input_shape = list(input.shape)
#         expected_bsz = (
#             getattr(self.args, "batch_size", 0)
#             if self.training
#             else getattr(self.args, "batch_size_valid", 0)
#         )
#         # This indicates that --batch-size or --max-sentences is not specified
#         if expected_bsz is None:
#             expected_bsz = 0
#         # Note: Padding is not necessary at generation time at present
#         # because all DDP workers process the same batch. Also, batch size at generation time
#         # can be different from that present in the checkpoint state
#         if (
#             not self.in_generation
#             and expected_bsz != 0
#             and input_shape[0] != expected_bsz
#         ):
#             logger.warning(
#                 f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
#             )
#             assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
#             padded_input = torch.zeros(
#                 (expected_bsz, input_shape[1], input_shape[2]),
#                 dtype=input.dtype,
#                 layout=input.layout,
#                 device=input.device,
#             )
#             padded_input[: input_shape[0], :, :] = input
#             input = padded_input

#             padded_input_padding_mask = torch.ones(
#                 (
#                     expected_bsz,
#                     input_shape[1],
#                 ),
#                 dtype=torch.bool,
#                 device=input.device,
#             )
#             if input_padding_mask is not None:
#                 padded_input_padding_mask[: input_shape[0], :] = input_padding_mask
#             else:
#                 padded_input_padding_mask[: input_shape[0], :] = False
#             input_padding_mask = padded_input_padding_mask

#         # Reshape into S tokens by dropping sequence dimension.
#         reshaped_input = input.reshape(-1, d_model)
#         reshaped_input_shape = reshaped_input.shape
#         reshaped_input_padding_mask = (
#             input_padding_mask.reshape(-1) if input_padding_mask is not None else None
#         )

#         # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
#         # Pro of --max-tokens: more flexible for MT variable sequence lengths
#         # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
#         if expected_bsz == 0:
#             expected_dim = reshaped_input_shape[0] * torch.ones(
#                 (1,), dtype=torch.long, device=input.device
#             )
#             dist.all_reduce(expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX)
#             expected_dim = int(expected_dim.item())
#             padded_input = torch.zeros(
#                 (expected_dim, reshaped_input_shape[1]),
#                 dtype=input.dtype,
#                 layout=input.layout,
#                 device=input.device,
#             )
#             padded_input[: reshaped_input_shape[0], :] = reshaped_input
#             reshaped_input = padded_input

#             padded_input_padding_mask = torch.ones(
#                 (expected_dim,), dtype=torch.bool, device=padded_input.device
#             )
#             if reshaped_input_padding_mask is not None:
#                 padded_input_padding_mask[
#                     : reshaped_input_shape[0]
#                 ] = reshaped_input_padding_mask
#             else:
#                 padded_input_padding_mask[: reshaped_input_shape[0]] = False
#             reshaped_input_padding_mask = padded_input_padding_mask

#         #find optimal routing
#         flip_ = random.random()

#         if flip_ > 1:
#             #top2gate
#             if has_tutel:
#                 l_aux, self.metadata, C, E, indices_, locations_, gates_, gates = self.gate(
#                     reshaped_input, reshaped_input_padding_mask
#                 )
#                 S, M = reshaped_input.size(0), reshaped_input.size(1)

#                 if not hasattr(self, "_tutel_dispatcher"):
#                     self._tutel_dispatcher = tutel_moe.fast_dispatcher(
#                         E, C, M, dispatch_dtype=reshaped_input.dtype
#                     )
#                 self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
#                 dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
#             else:
#                 l_aux, combine_weights, dispatch_mask, self.metadata, gates = self.gate(
#                     reshaped_input, reshaped_input_padding_mask
#                 )

#                 dispatch_mask = dispatch_mask.to(input.dtype).permute(
#                     1, 2, 0
#                 )  # S,E,C -> E,C,S
#                 E, C, S = dispatch_mask.size()
#                 M = reshaped_input.size(1)
#                 assert reshaped_input.size() == (S, M)
#                 # einsum("sec,sm->ecm")
#                 dispatched_input = torch.mm(
#                     dispatch_mask.view(E * C, S), reshaped_input
#                 )  # -> (E*C),M

#             if self.all2all_size > 1:
#                 dispatched_input = self.all_to_all_wrapper(dispatched_input)

#             # Re-shape after all-to-all: ecm -> gecm
#             dispatched_input = dispatched_input.reshape(
#                 self.all2all_size, self.num_local_experts, -1, d_model
#             )
#             chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
#             expert_outputs = []
#             for chunk, expert in zip(chunks, self.experts):
#                 expert_outputs += [expert(chunk)]
#             expert_output = torch.cat(expert_outputs, dim=1)
#             if self.all2all_size > 1:
#                 expert_output = self.all_to_all_wrapper(expert_output)

#             # Re-shape back: gecm -> ecm
#             expert_output = expert_output.reshape(
#                 self.all2all_size * self.num_local_experts, -1, d_model
#             )

#             if has_tutel:
#                 combined_output = self._tutel_dispatcher.decode(
#                     expert_output.view(E * C, M)
#                 )
#             else:
#                 # einsum("sec,ecm->sm")
#                 combined_output = combine_weights.view(S, E * C).mm(
#                     expert_output.view(E * C, M)
#                 )

#         else:
#             #optimal gate
#             if has_tutel:
#                 l_aux, self.metadata, C, E, indices_, locations_, gates_, gates = self.gate(
#                     reshaped_input, reshaped_input_padding_mask
#                 )
#                 S, M = reshaped_input.size(0), reshaped_input.size(1)

#                 # if not hasattr(self, "_tutel_dispatcher"):
#                 #     self._tutel_dispatcher = tutel_moe.fast_dispatcher(
#                 #         E, C, M, dispatch_dtype=reshaped_input.dtype
#                 #     )
#                 # self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
#                 # dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
#             else:
#                 l_aux, combine_weights, dispatch_mask, self.metadata, gates = self.gate(
#                     reshaped_input, reshaped_input_padding_mask
#                 )

#                 dispatch_mask = dispatch_mask.to(input.dtype).permute(
#                     1, 2, 0
#                 )  # S,E,C -> E,C,S
#                 E, C, S = dispatch_mask.size()
#                 M = reshaped_input.size(1)
#                 # assert reshaped_input.size() == (S, M)
#                 # # einsum("sec,sm->ecm")
#                 # dispatched_input = torch.mm(
#                 #     dispatch_mask.view(E * C, S), reshaped_input
#                 # )  # -> (E*C),M
#             dispatched_input = reshaped_input
#             if self.all2all_size > 1:
#                 dispatched_input = self.all_to_all_wrapper(dispatched_input)

#             # Re-shape after all-to-all: ecm -> gecm
#             dispatched_input = dispatched_input.reshape(
#                 self.all2all_size, self.num_local_experts, -1, d_model
#             )
#             chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
#             expert_outputs = []
#             for chunk, expert in zip(chunks, self.experts):
#                 expert_outputs += [expert(chunk)]
#             expert_output = torch.cat(expert_outputs, dim=1)
#             print(expert_output.shape)
#             #find optimal routing
#             with torch.no_grad():
#                 expert_output_opt = expert_output.detach().clone()
#                 num_token = reshaped_input_shape[0]
#                 expert_output_opt = expert_output_opt.reshape(num_token, self.all2all_size *self.num_local_experts, -1)
#                 expert_output_opt = torch.norm(expert_output_opt, dim=2)
#                 expert_output_score = F.softmax(expert_output_opt, dim=1)
            
#             #calculate loss
#             compete_loss = kl_divergence(gates, expert_output_score)
#             # combine with gate loss
#             l_aux += compete_loss
            

#             if self.all2all_size > 1:
#                 expert_output = self.all_to_all_wrapper(expert_output)

#             # Re-shape back: gecm -> ecm
#             expert_output = expert_output.reshape(
#                 self.all2all_size * self.num_local_experts, -1, d_model
#             )
#             combined_output = expert_output.view(S, M)
#             # if has_tutel:
#             #     combined_output = self._tutel_dispatcher.decode(
#             #         expert_output.view(E * C, M)
#             #     )
#             # else:
#             #     # einsum("sec,ecm->sm")
#             #     combined_output = combine_weights.view(S, E * C).mm(
#             #         expert_output.view(E * C, M)
#             #     )

#         # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
#         combined_output = combined_output[: reshaped_input_shape[0], :]
#         combined_output = combined_output.reshape(input.shape)
#         combined_output = combined_output[: input_shape[0], :, :]

#         self.record_all_to_all_stats()

#         return combined_output, l_aux

#     def prepare_for_inference_(self):
#         self.in_generation = True

#     def all_to_all_wrapper(self, input: Tensor):
#         dummy_a2a = getattr(self.args, "dummy_a2a", False)
#         if dummy_a2a:
#             input = input.contiguous()
#             output = input.detach().clone()
#             return input
#         # always record times, since it is not a lot of overhead
#         # if we do not log it we simply clear it off in record_all_to_all_stats
#         cuda_start = torch.cuda.Event(enable_timing=True)
#         cuda_end = torch.cuda.Event(enable_timing=True)
#         cpu_start = time.time() * 1000
#         cuda_start.record()
#         output = _AllToAll.apply(self.all2all_group, input)
#         cuda_end.record()
#         cpu_end = time.time() * 1000
#         self.a2a_cpu_time_ms += cpu_end - cpu_start
#         self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
#         return output

#     def record_all_to_all_stats(self):
#         # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
#         record_a2a_perf_stats = getattr(self.args, "record_a2a_perf_stats", False)
#         if record_a2a_perf_stats:
#             torch.cuda.synchronize()
#             self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
#             a2a_cuda_time_ms = 0.0
#             for ev_start, ev_end in self.a2a_cuda_event_intervals:
#                 a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
#             self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
#         # reset stats
#         self.a2a_cpu_time_ms = 0.0
#         self.a2a_cuda_event_intervals = []


# # # einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# # # See https://arxiv.org/pdf/2006.16668.pdf for details.

# # # Based on https://github.com/pytorch/pytorch/pull/40762
# # class _AllToAll(torch.autograd.Function):
# #     @staticmethod
# #     def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
# #         ctx.group = group
# #         input = input.contiguous()
# #         output = torch.empty_like(input)
# #         if torch.distributed.is_initialized():
# #             dist.all_to_all_single(output, input, group=group)
# #         else:
# #             assert group is None
# #             output = input
# #         return output

# #     @staticmethod
# #     def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
# #         return (None, _AllToAll.apply(ctx.group, *grad_output))




# # class MOELayer(Base):
# #     """MOELayer module which implements MixtureOfExperts as described in Gshard_.
# #     ::

# #         gate = Top2Gate(model_dim, num_experts)
# #         moe = MOELayer(gate, expert)
# #         output = moe(input)
# #         l_aux = moe.l_aux

# #     .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

# #     Args:
# #         gate (torch.nn.Module):
# #             gate network
# #         expert (torch.nn.Module):
# #             expert network
# #     """

# #     def __init__(self, gate, experts, args):
# #         if has_fairseq:
# #             super(Base, self).__init__()
# #         else:
# #             super().__init__()
# #         self.gate = gate
# #         if type(experts) == ModuleList:
# #             self.experts = cast(ModuleList, experts)
# #         else:
# #             self.experts = ModuleList([experts])
# #         _, self.expert_group = get_moe_group(args.moe_expert_count)
# #         self.all2all_group = get_all2all_group(args.moe_expert_count)
# #         self.world_size = dist.get_world_size(group=self.expert_group)
# #         self.all2all_size = dist.get_world_size(group=self.all2all_group)
# #         for p in experts.parameters():
# #             p.expert = True  # type: ignore
# #         self.num_local_experts = len(self.experts)
# #         self.args = args
# #         self.in_generation = False
# #         self.a2a_cuda_event_intervals = []
# #         self.a2a_cpu_time_ms = 0.0
# #         self.use_tutel=getattr(args, 'use_tutel', False) and has_tutel
# #         self.use_tutel_all2all=getattr(args, 'use_tutel_all2all', False) and has_tutel


# #     def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
# #         assert len(input) == 1, "only single input Tensor supported"
# #         input = input[0]
# #         assert (
# #             len(input.shape) == 3
# #         ), "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
# #         if input_padding_mask is not None:
# #             assert (
# #                 len(input_padding_mask.shape) == 2
# #             ), "input Tensor must have dimensions: (s)equence, (t)oken"
# #             assert input_padding_mask.shape[0] == input.shape[0]
# #             assert input_padding_mask.shape[1] == input.shape[1]
# #         # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

# #         # Implement Algorithm 2 from GShard paper.
# #         d_model = input.shape[2]
# #         # Pad to expected batch size
# #         input_shape = list(input.shape)
# #         expected_bsz = (
# #             getattr(self.args, "batch_size", 0)
# #             if self.training
# #             else getattr(self.args, "batch_size_valid", 0)
# #         )
# #         # This indicates that --batch-size or --max-sentences is not specified
# #         if expected_bsz is None:
# #             expected_bsz = 0
# #         # Note: Padding is not necessary at generation time at present
# #         # because all DDP workers process the same batch. Also, batch size at generation time
# #         # can be different from that present in the checkpoint state
# #         if (
# #             not self.in_generation
# #             and expected_bsz != 0
# #             and input_shape[0] != expected_bsz
# #         ):
# #             logger.warning(
# #                 f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})"
# #             )
# #             assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
# #             padded_input = torch.zeros(
# #                 (expected_bsz, input_shape[1], input_shape[2]),
# #                 dtype=input.dtype,
# #                 layout=input.layout,
# #                 device=input.device,
# #             )
# #             padded_input[: input_shape[0], :, :] = input
# #             input = padded_input

# #             padded_input_padding_mask = torch.ones(
# #                 (
# #                     expected_bsz,
# #                     input_shape[1],
# #                 ),
# #                 dtype=torch.bool,
# #                 device=input.device,
# #             )
# #             if input_padding_mask is not None:
# #                 padded_input_padding_mask[: input_shape[0], :] = input_padding_mask
# #             else:
# #                 padded_input_padding_mask[: input_shape[0], :] = False
# #             input_padding_mask = padded_input_padding_mask

# #         # Reshape into S tokens by dropping sequence dimension.
# #         reshaped_input = input.reshape(-1, d_model)
# #         reshaped_input_shape = reshaped_input.shape
# #         reshaped_input_padding_mask = (
# #             input_padding_mask.reshape(-1) if input_padding_mask is not None else None
# #         )

# #         # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
# #         # Pro of --max-tokens: more flexible for MT variable sequence lengths
# #         # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
# #         if expected_bsz == 0:
# #             expected_dim = reshaped_input_shape[0] * torch.ones(
# #                 (1,), dtype=torch.long, device=input.device
# #             )
# #             dist.all_reduce(expected_dim, group=dist.group.WORLD, op=dist.ReduceOp.MAX)
# #             expected_dim = int(expected_dim.item())
# #             padded_input = torch.zeros(
# #                 (expected_dim, reshaped_input_shape[1]),
# #                 dtype=input.dtype,
# #                 layout=input.layout,
# #                 device=input.device,
# #             )
# #             padded_input[: reshaped_input_shape[0], :] = reshaped_input
# #             reshaped_input = padded_input

# #             padded_input_padding_mask = torch.ones(
# #                 (expected_dim,), dtype=torch.bool, device=padded_input.device
# #             )
# #             if reshaped_input_padding_mask is not None:
# #                 padded_input_padding_mask[
# #                     : reshaped_input_shape[0]
# #                 ] = reshaped_input_padding_mask
# #             else:
# #                 padded_input_padding_mask[: reshaped_input_shape[0]] = False
# #             reshaped_input_padding_mask = padded_input_padding_mask

# #         #calculate thread_hold
# #         flip_num = random.random()
       
# #         if has_tutel:
# #             l_aux, self.metadata, C, E, indices_, locations_, gates_, gates = self.gate(
# #                 reshaped_input, reshaped_input_padding_mask
# #             )
# #             S, M = reshaped_input.size(0), reshaped_input.size(1)

# #             if not hasattr(self, "_tutel_dispatcher"):
# #                 self._tutel_dispatcher = tutel_moe.fast_dispatcher(
# #                     E, C, M, dispatch_dtype=reshaped_input.dtype
# #                 )
# #             self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
# #             dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
# #         else:
# #             l_aux, combine_weights, dispatch_mask, self.metadata, gates = self.gate(
# #                 reshaped_input, reshaped_input_padding_mask
# #             )

# #             dispatch_mask = dispatch_mask.to(input.dtype).permute(
# #                 1, 2, 0
# #             )  # S,E,C -> E,C,S
# #             E, C, S = dispatch_mask.size()
# #             M = reshaped_input.size(1)
# #             assert reshaped_input.size() == (S, M)
# #             # einsum("sec,sm->ecm")
# #             dispatched_input = torch.mm(
# #                 dispatch_mask.view(E * C, S), reshaped_input
# #             )  # -> (E*C),M

# #         if self.all2all_size > 1:
# #             dispatched_input = self.all_to_all_wrapper(dispatched_input)

# #         # Re-shape after all-to-all: ecm -> gecm
# #         dispatched_input = dispatched_input.reshape(
# #             self.all2all_size, self.num_local_experts, -1, d_model
# #         )
# #         chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
# #         expert_outputs = []
# #         for chunk, expert in zip(chunks, self.experts):
# #             expert_outputs += [expert(chunk)]
# #         expert_output = torch.cat(expert_outputs, dim=1)

# #         #find the optimal routing
# #         if flip_num > 0.2:
# #             try:
# #                 with torch.no_grad():
# #                     expert_output_opt = expert_output.clone()
# #                     num_token = reshaped_input_shape[0]
# #                     expert_output_opt = expert_output_opt.reshape(num_token, self.all2all_size *self.num_local_experts, -1)
# #                     expert_output_opt = torch.norm(expert_output_opt, dim=2)
# #                 #calculate loss
# #                 compete_loss = kl_loss_sym(gates, expert_output_opt)
# #                 # combine with gate loss
# #                 l_aux += compete_loss
# #             except:
# #                 pass


# #         if self.all2all_size > 1:
# #             expert_output = self.all_to_all_wrapper(expert_output)

# #         # Re-shape back: gecm -> ecm
# #         expert_output = expert_output.reshape(
# #             self.all2all_size * self.num_local_experts, -1, d_model
# #         )

# #         if has_tutel:
# #             combined_output = self._tutel_dispatcher.decode(
# #                 expert_output.view(E * C, M)
# #             )
# #         else:
# #             # einsum("sec,ecm->sm")
# #             combined_output = combine_weights.view(S, E * C).mm(
# #                 expert_output.view(E * C, M)
# #             )

# #         # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
# #         combined_output = combined_output[: reshaped_input_shape[0], :]
# #         combined_output = combined_output.reshape(input.shape)
# #         combined_output = combined_output[: input_shape[0], :, :]

# #         self.record_all_to_all_stats()

# #         return combined_output, l_aux
     

# #     def prepare_for_inference_(self):
# #         self.in_generation = True

# #     def all_to_all_wrapper(self, input: Tensor):
# #         dummy_a2a = getattr(self.args, "dummy_a2a", False)
# #         if dummy_a2a:
# #             input = input.contiguous()
# #             output = input.detach().clone()
# #             return input
# #         # always record times, since it is not a lot of overhead
# #         # if we do not log it we simply clear it off in record_all_to_all_stats
# #         cuda_start = torch.cuda.Event(enable_timing=True)
# #         cuda_end = torch.cuda.Event(enable_timing=True)
# #         cpu_start = time.time() * 1000
# #         cuda_start.record()
# #         output = _AllToAll.apply(self.all2all_group, input)
# #         cuda_end.record()
# #         cpu_end = time.time() * 1000
# #         self.a2a_cpu_time_ms += cpu_end - cpu_start
# #         self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
# #         return output

# #     def record_all_to_all_stats(self):
# #         # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
# #         record_a2a_perf_stats = getattr(self.args, "record_a2a_perf_stats", False)
# #         if record_a2a_perf_stats:
# #             torch.cuda.synchronize()
# #             self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
# #             a2a_cuda_time_ms = 0.0
# #             for ev_start, ev_end in self.a2a_cuda_event_intervals:
# #                 a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
# #             self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
# #         # reset stats
# #         self.a2a_cpu_time_ms = 0.0
# #         self.a2a_cuda_event_intervals = []