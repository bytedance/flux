; Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;    http://www.apache.org/licenses/LICENSE-2.0
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.

; How to generate cuda_extra.bc: ~/.triton/llvm/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release/bin/llvm-as ./cuda_extra.ll -o ./cuda_extra.bc
; How to write asm in LLVM IR: https://llvm.org/docs/LangRef.html#inline-asm-constraint-string
; How to write asm in LLVM IR: https://mcyoung.xyz/2023/08/01/llvm-ir/
; How to inspect a *.bc file: https://releases.llvm.org/13.0.0/docs/CommandGuide/llvm-bcanalyzer.html
;   llvm-bcanalyzer -dump /usr/local/cuda-12.4/nvvm/libdevice/libdevice.10.bc

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "nvptx64-nvidia-cuda"


define i32 @tid() #0 {
  %1 = call i32 asm "mov.u32 $0, %tid.x;", "=r"() nounwind
  ret i32 %1
}

define i32 @tidx() #0 {
  %1 = call i32 asm "mov.u32 $0, %tid.x;", "=r"() nounwind
  ret i32 %1
}

define i32 @tidy() #0 {
  %1 = call i32 asm "mov.u32 $0, %tid.y;", "=r"() nounwind
  ret i32 %1
}

define i32 @tidz() #0 {
  %1 = call i32 asm "mov.u32 $0, %tid.z;", "=r"() nounwind
  ret i32 %1
}

define i32 @ntidx() #0 {
  %1 = call i32 asm "mov.u32 $0, %ntid.x;", "=r"() nounwind
  ret i32 %1
}

define i32 @ntidy() #0 {
  %1 = call i32 asm "mov.u32 $0, %ntid.y;", "=r"() nounwind
  ret i32 %1
}

define i32 @ntidz() #0 {
  %1 = call i32 asm "mov.u32 $0, %ntid.z;", "=r"() nounwind
  ret i32 %1
}

define void @syncthreads() #0 {
  call void asm sideeffect "bar.sync 0;", ""() nounwind
  ret void
}

define void @red_release_gpu(i64 %ptr, i32 %value) #0 {
  call void asm sideeffect "red.release.gpu.global.add.s32 [$0], $1;", "l,r"(i64 %ptr, i32 %value)
  ret void
}

define i32 @ld_acquire_gpu(i64 %ptr) #0 {
  %1 = call i32 asm sideeffect "ld.global.acquire.gpu.b32 $0, [$1];", "=r,l"(i64 %ptr) nounwind
  ret i32 %1
}

define void @red_release_sys(i64 %ptr, i32 %value) #0 {
  call void asm sideeffect "red.release.sys.global.add.s32 [$0], $1;", "l,r"(i64 %ptr, i32 %value)
  ret void
}

define i32 @ld_acquire_sys(i64 %ptr) #0 {
  %1 = call i32 asm sideeffect "ld.global.acquire.sys.b32 $0, [$1];", "=r,l"(i64 %ptr) nounwind
  ret i32 %1
}

define i32 @atomic_add_release_sys(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.release.sys.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_release_gpu(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.release.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_relaxed_sys(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.relaxed.sys.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_relaxed_gpu(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.relaxed.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_acquire_sys(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.acquire.sys.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_acquire_gpu(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.acquire.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_acq_rel_sys(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.acq_rel.sys.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @atomic_add_acq_rel_gpu(i64 %ptr, i32 %value) #0 {
  %1 = call i32 asm sideeffect "atom.acq_rel.gpu.global.add.s32 $0, [$1], $2;", "=r,l,r"(i64 %ptr, i32 %value)
  ret i32 %1
}

define i32 @__shfl_idx_sync_i32(i32 %mask, i32 %var, i32 %laneid) #0 {
  %1 = call i32 asm sideeffect "shfl.sync.idx.b32 $0, $1, $2, 31, $3;", "=r,r,r,r"(i32 %var, i32 %laneid, i32 %mask)
  ret i32 %1
}

define i32 @__shfl_up_sync_i32(i32 %mask, i32 %var, i32 %delta) #0 {
  %1 = call i32 asm sideeffect "shfl.sync.up.b32 $0, $1, $2, 0, $3;", "=r,r,r,r"(i32 %var, i32 %delta, i32 %mask)
  ret i32 %1
}

define i32 @__shfl_down_sync_i32(i32 %mask, i32 %var, i32 %delta) #0 {
  %1 = call i32 asm sideeffect "shfl.sync.down.b32 $0, $1, $2, 0, $3;", "=r,r,r,r"(i32 %var, i32 %delta, i32 %mask)
  ret i32 %1
}

define i32 @__ballot_sync(i32 %mask, i32 %predicate) #0 {
  %1 = call i32 asm sideeffect "{.reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2;}", "=r,r,r"(i32 %predicate, i32 %mask)
  ret i32 %1
}

define i64 @ld_gpu_i64(i64 %ptr) #0 {
  %1 = call i64 asm sideeffect "ld.global.relaxed.gpu.b64 $0, [$1];", "=l,l"(i64 %ptr) nounwind
  ret i64 %1
}

define i32 @ld_gpu_i32(i64 %ptr) #0 {
  %1 = call i32 asm sideeffect "ld.global.relaxed.gpu.b32 $0, [$1];", "=r,l"(i64 %ptr) nounwind
  ret i32 %1
}

define i16 @ld_gpu_i16(i64 %ptr) #0 {
  %1 = call i16 asm sideeffect "ld.global.relaxed.gpu.b16 $0, [$1];", "=h,l"(i64 %ptr) nounwind
  ret i16 %1
}

define i64 @ld_sys_i64(i64 %ptr) #0 {
  %1 = call i64 asm sideeffect "ld.global.relaxed.sys.b64 $0, [$1];", "=l,l"(i64 %ptr) nounwind
  ret i64 %1
}

define i32 @ld_sys_i32(i64 %ptr) #0 {
  %1 = call i32 asm sideeffect "ld.global.relaxed.sys.b32 $0, [$1];", "=r,l"(i64 %ptr) nounwind
  ret i32 %1
}

define i16 @ld_sys_i16(i64 %ptr) #0 {
  %1 = call i16 asm sideeffect "ld.global.relaxed.sys.b16 $0, [$1];", "=h,l"(i64 %ptr) nounwind
  ret i16 %1
}

attributes #0 = { alwaysinline nounwind }
