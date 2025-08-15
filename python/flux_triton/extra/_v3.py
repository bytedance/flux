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

import triton
import triton.language as tl
from triton.language import core


def patch_triton_module(func):
    func.__module__ = f"triton.{func.__module__}"
    return func


@patch_triton_module
@core.extern
def __syncthreads(_builder=None):
    return core.inline_asm_elementwise(
        asm="""
        bar.sync 0;
        mov.u32 $0, 0;
        """,
        constraints="=r",  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=False,  # no optimize this!
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@tl.core.extern
def __tid__(axis: tl.constexpr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %tid.{axis.value};",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def tid(axis: tl.constexpr):
    if axis == 0:
        return __tid__("x")
    elif axis == 1:
        return __tid__("y")
    elif axis == 2:
        return __tid__("z")
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


@patch_triton_module
@tl.core.extern
def __ntid__(_builder=None):
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %ntid.x;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def ntid(axis: tl.constexpr, _builder=None):
    if axis == 0:
        return __ntid__("x", _builder=_builder)
    elif axis == 1:
        return __ntid__("y", _builder=_builder)
    elif axis == 2:
        return __ntid__("z", _builder=_builder)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2", _builder=_builder)


@patch_triton_module
@tl.core.extern
def red_release(barrier_ptr, value, scope: tl.constexpr = "gpu", _builder=None):
    tl.inline_asm_elementwise(
        asm=f"""{{
        mov.u32         $0, %tid.x;
        red.release.{scope.value}.global.add.s32 [$1], $2;
        }}""",
        constraints=("=r," "l,r"),  # no use output, which is threadId.x
        args=[barrier_ptr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def arrive_inc(barrier_ptr, thread_idx, value, scope: tl.constexpr):
    __syncthreads()
    if thread_idx == 0:
        red_release(barrier_ptr, value, scope)


@patch_triton_module
@tl.core.extern
def arrive_inc_asm(barrier_ptr, thread_idx, value, scope: tl.constexpr = "gpu", _builder=None):
    tl.inline_asm_elementwise(
        asm=f"""{{
        bar.sync        0;
        mov.u32         $0, %tid.x;
        setp.eq.s32     %p1, $2, 0;
        @%p1            red.release.{scope.value}.global.add.s32 [$1], $3;
        }}""",
        constraints=("=r," "l,r,r"),  # no use output
        args=[barrier_ptr, thread_idx, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@tl.core.extern
def ld_acquire(barrier_ptr, scope: tl.constexpr = "gpu", _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"""{{
        ld.global.acquire.{scope.value}.b32 $0, [$1];
        }}
        """,
        constraints=("=r,l"),
        args=[barrier_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@tl.core.extern
def __atomic_add(
    ptr,
    value,
    scope: tl.constexpr = "gpu",
    semantic: tl.constexpr = "relaxed",
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.add.s32 $0, [$1], $2;",
        constraints=("=r,l,r"),
        args=[
            ptr,
            value,
        ],
        is_pure=False,
        pack=1,
        dtype=tl.int32,
        _builder=_builder,
    )


@triton.jit
def atomic_add(barrier_ptr, value, scope: tl.constexpr, semantic: tl.constexpr):
    """custom atomic_add implementation using extern_elementwise

    :param scope: one of "gpu", "sys". default to "gpu"
    :param semantic: one of "release", "acquire", "relaxed", "acq_rel". default to "relaxed"
    :returns: the result of atomic_add
    :rtype: int
    """
    return __atomic_add(barrier_ptr, value, scope, semantic)


@triton.jit
def wait_eq(barrier_ptr, thread_idx, value, scope: tl.constexpr):
    if thread_idx == 0:
        while ld_acquire(barrier_ptr, scope) != value:
            pass
    __syncthreads()


@patch_triton_module
@tl.core.extern
def __shfl_sync_with_mode_i32(
    mask: tl.core.uint32,
    value: tl.core.int32,
    delta: tl.core.uint32,
    mode: tl.constexpr = "up",
    c: tl.constexpr = 31,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"shfl.sync.{mode.value}.b32 $0, $1, $2, {c.value}, $3;",
        constraints="=r,r,r,r",
        args=[value, delta, mask],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def __shfl_sync_i32(mask, value, laneid):
    return __shfl_sync_with_mode_i32(mask, value, laneid, "idx", 31)


@triton.jit
def __shfl_up_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "up", 0)


@triton.jit
def __shfl_down_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "down", 0)


@patch_triton_module
@tl.core.extern
def __ballot_sync(
    mask: tl.core.uint32,
    predicate: tl.core.int32,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm="{.reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2;}",
        constraints="=r,r,r",
        args=[predicate, mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@patch_triton_module
@tl.core.extern
def __ld(ptr, scope: tl.constexpr = "gpu", nbit: tl.constexpr = 32, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"ld.global.relaxed.{scope.value}.b{nbit.value} $0, [$1];",
        constraints="=r,l",
        args=[ptr],
        dtype=tl.int32 if nbit.value == 32 else (tl.int64 if nbit.value == 64 else tl.int16),
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def ld(ptr, scope: tl.constexpr = "gpu"):
    if ptr.dtype == tl.pointer_type(tl.int32):
        return __ld(ptr, scope, 32)
    elif ptr.dtype == tl.pointer_type(tl.int64):
        return __ld(ptr, scope, 64)
    elif ptr.dtype == tl.pointer_type(tl.int16):
        return __ld(ptr, scope, 16)
    else:
        tl.static_assert(False, "unsupported dtype")


from triton.language.extra.cuda.libdevice import ffs

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
