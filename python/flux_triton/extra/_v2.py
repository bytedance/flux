################################################################################
#
# Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import os

import triton
import triton.language as tl
from triton.language import core

__path__ = os.path.dirname(os.path.abspath(__file__))

tl.pointer_type.__hash__ = lambda x: hash(str(x))


def patch_triton_module(func):
    func.__module__ = f"triton.core.{func.__module__}"
    return func


# copy of tl.core.extern_elementwise here. a little trick to support pointer type
def extern_elementwise(
    lib_name: str,
    lib_path: str,
    args: list,
    arg_type_symbol_dict: dict,
    is_pure: bool,
    _builder=None,
):
    """
    Dispatch an elementwise function to a library
    :param lib_name: the name of the library
    :param lib_path: the path of the library
    :param args: the arguments of the function
    :param arg_type_symbol_dict: the type of the arguments
    :param is_pure: whether the function is pure
    :param _builder: the builder
    :return: the return value of the function
    """
    dispatch_args = args.copy()
    all_scalar = True
    ret_shape = None
    arg_types = []
    for i in range(len(dispatch_args)):
        dispatch_args[i] = tl.core._to_tensor(dispatch_args[i], _builder)
        arg_types.append(dispatch_args[i].dtype)
        if dispatch_args[i].type.is_block():
            all_scalar = False
    if len(arg_types) > 0:
        arg_types = tuple(arg_types)
        arithmetic_check = True
        # If there's a type tuple that is not supported by the library, we will do arithmetic check
        if arg_types in arg_type_symbol_dict:
            arithmetic_check = False
        broadcast_arg = dispatch_args[0]
        # Get the broadcast shape over all the arguments
        for i, item in enumerate(dispatch_args):
            _, broadcast_arg = tl.core.semantic.binary_op_type_checking_impl(
                item,
                broadcast_arg,
                _builder,
                allow_lhs_ptr=True,  # allow pointer here
                allow_rhs_ptr=True,  # allow pointer here
                arithmetic_check=arithmetic_check,
            )
        # Change the shape of each argument based on the broadcast shape
        for i in range(len(dispatch_args)):
            dispatch_args[i], _ = tl.core.semantic.binary_op_type_checking_impl(
                dispatch_args[i],
                broadcast_arg,
                _builder,
                allow_lhs_ptr=True,
                allow_rhs_ptr=True,
                arithmetic_check=arithmetic_check,
            )
        if not all_scalar:
            ret_shape = broadcast_arg.shape
    func = getattr(_builder, "create_extern_elementwise")
    return tl.core.dispatch(
        func, lib_name, lib_path, dispatch_args, arg_type_symbol_dict, ret_shape, is_pure, _builder
    )


@core.extern
def __tid__(axis: tl.constexpr = "x", _builder=None):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [],
        {
            tuple(): (f"tid{axis.value}", core.dtype("int32")),
        },
        is_pure=True,
        _builder=_builder,
    )


@patch_triton_module
@triton.jit
def tid(axis: tl.constexpr, _builder=None):
    """
    :param axis should be one of 0,1,2
    :return blockDim.x, blockDim.y, blockDim.z
    """
    if axis == 0:
        return __tid__("x")
    elif axis == 1:
        return __tid__("y")
    elif axis == 2:
        return __tid__("z")
    else:
        tl.static_assert(False, "axis should be one of 0,1,2", _builder=_builder)


@core.extern
def __ntid__(axis: tl.constexpr = "x", _builder=None):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [],
        {
            tuple(): (f"ntid{axis.value}", core.dtype("int32")),
        },
        is_pure=True,
        _builder=_builder,
    )


@patch_triton_module
@triton.jit
def ntid(axis: tl.constexpr, _builder=None):
    """
    :param axis should be one of 0,1,2
    :return blockDim.x, blockDim.y, blockDim.z
    """
    if axis == 0:
        return __ntid__("x")
    elif axis == 1:
        return __ntid__("y")
    elif axis == 2:
        return __ntid__("z")
    else:
        tl.static_assert(False, "axis should be one of 0,1,2", _builder=_builder)


@core.extern
def __syncthreads(_builder=None):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [],
        {
            tuple(): ("syncthreads", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def ld_acquire(ptr, scope: tl.constexpr = "gpu", _builder=None):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [
            ptr,
        ],
        {
            (tl.pi32_t,): (f"ld_acquire_{scope.value}", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def ld(ptr, scope: tl.constexpr = "gpu", _builder=None):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [
            ptr,
        ],
        {
            (tl.pointer_type(tl.int16),): (f"ld_{scope.value}_i16", core.dtype("int16")),
            (tl.pointer_type(tl.int32),): (f"ld_{scope.value}_i32", core.dtype("int32")),
            (tl.pointer_type(tl.int64),): (f"ld_{scope.value}_i64", core.dtype("int64")),
        },
        is_pure=False,
        _builder=_builder,
    )


@core.extern
def red_release(ptr, value, scope: tl.constexpr = "gpu", _builder=None):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [
            ptr,
            value,
        ],
        {
            (
                tl.pi32_t,
                core.dtype("int32"),
            ): (f"red_release_{scope.value}", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
    )


@patch_triton_module
@triton.jit
def wait_eq(barrier_ptr, thread_idx, value, scope: tl.constexpr):
    if thread_idx == 0:
        while ld_acquire(barrier_ptr, scope) != value:
            pass
    __syncthreads()


@patch_triton_module
@triton.jit
def arrive_inc(barrier_ptr, thread_idx, value, scope: tl.constexpr):
    __syncthreads()
    if thread_idx == 0:
        red_release(barrier_ptr, value, scope)


@core.extern
def __atomic_add(
    ptr, value, scope: tl.constexpr = "gpu", semantic: tl.constexpr = "relaxed", _builder=None
):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [
            ptr,
            value,
        ],
        {
            (
                tl.pi32_t,
                core.dtype("int32"),
            ): (f"atomic_add_{semantic.value}_{scope.value}", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
    )


@patch_triton_module
@triton.jit
def atomic_add(barrier_ptr, value, scope: tl.constexpr, semantic: tl.constexpr):
    """custom atomic_add implementation using extern_elementwise

    :param scope: one of "gpu", "sys". default to "gpu"
    :param semantic: one of "release", "acquire", "relaxed", "acq_rel". default to "relaxed"
    :returns: the result of atomic_add
    :rtype: int
    """
    return __atomic_add(barrier_ptr, value, scope, semantic)


@core.extern
def __shfl_sync_with_mode_i32(
    mask: tl.core.uint32,
    value: tl.core.int32,
    delta: tl.core.uint32,
    mode: tl.constexpr = "up",
    _builder=None,
):
    return extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [
            mask,
            value,
            delta,
        ],
        {
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): (f"__shfl_{mode.value}_sync_i32", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
    )


@triton.jit
def __shfl_sync_i32(mask, value, laneid):
    return __shfl_sync_with_mode_i32(mask, value, laneid, "idx")


@triton.jit
def __shfl_up_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "up")


@triton.jit
def __shfl_down_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "down")


@core.extern
def __ballot_sync(
    mask: tl.core.uint32,
    predicate: tl.core.int32,
    _builder=None,
):
    return tl.core.extern_elementwise(
        "cuda_extra",
        os.path.join(__path__, "cuda_extra.bc"),
        [
            mask,
            predicate,
        ],
        {
            (
                core.dtype("uint32"),
                core.dtype("uint32"),
            ): (f"__ballot_sync", core.dtype("int32")),
        },
        is_pure=False,
        _builder=_builder,
    )


from triton.language.math import ffs

__all__ = [
    "__syncthreads",
    "tid",
    "ntid",
    "wait_eq",
    "arrive_inc",
    "red_release",
    "ld_acquire",
    "atomic_add",
    "__shfl_sync_i32",
    "__shfl_up_sync_i32",
    "__shfl_down_sync_i32",
    "__ballot_sync",
    "ld",
    "ffs",
]
