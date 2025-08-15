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
import binascii
import hashlib
from pathlib import Path
from typing import Any, Dict, List

import triton
from packaging.version import Version

_TRITON_VER = Version(triton.__version__)
_IS_NEW_TRITON = _TRITON_VER.major >= 3 and _TRITON_VER.minor >= 2


"""
some API changes:
 triton.compiler.ASTSource
    - signature:
        Dict[int, anotation] for 3.0.0
        Dict[name, anotation] for 3.2.0
    - constants: Dict[int, Any]
        Dict[int, anotation] for 3.0.0
        Dict[name, anotation] for 3.2.0
    - attrs: triton.compiler.AttrsDescriptor
"""


def hash_signature(signature: List[str]):
    m = hashlib.sha256()
    m.update(" ".join(signature).encode())
    return m.hexdigest()[:8]


def _meta_sig(num_stages: int, num_warps: int) -> str:
    meta_sig = f"warps{num_warps}xstages{num_stages}"
    return meta_sig


def constexpr(s):
    try:
        ret = int(s)
        return ret
    except ValueError:
        pass
    try:
        if s.lower() in ["true", "false"]:
            return 1 if s.lower() == "true" else 0
    except ValueError:
        pass
    try:
        ret = float(s)
        return ret
    except ValueError:
        pass
    return None


def make_ast_source_legacy(
    kernel: triton.JITFunction, signature: str
) -> triton.compiler.CompiledKernel:
    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), signature.split(",")))

    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {i: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constants}

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    attrs = triton.compiler.AttrsDescriptor(divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    for i in equal_to_1:
        constants.update({i: 1})
    src = triton.compiler.ASTSource(
        fn=kernel, constants=constants, signature=signature, attrs=attrs
    )
    return src


def make_ast_source_new(
    kernel: triton.JITFunction,
    signature: str,
) -> triton.compiler.CompiledKernel:
    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), signature.split(",")))

    hints = {(i,): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}

    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = "constexpr"

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"

    attrs = triton.backends.compiler.AttrsDescriptor.from_hints(hints)
    for p, v in attrs.get_constants().items():
        constants.update({kernel.arg_names[p[0]]: v})
    src = triton.compiler.ASTSource(
        fn=kernel, constexprs=constants, signature=signature, attrs=attrs
    )
    return src


def make_ast_source(kernel: triton.JITFunction, signature: str) -> triton.compiler.CompiledKernel:
    if _IS_NEW_TRITON:
        return make_ast_source_new(kernel, signature)
    else:
        return make_ast_source_legacy(kernel, signature)


def kernel_name_suffix(
    src: triton.compiler.ASTSource,
    num_stages: int,
    num_warps: int,
):
    from triton.compiler.code_generator import kernel_suffix

    suffix = kernel_suffix(src.signature.values(), src.attrs)
    return f"{src.hash()[:8]}_{suffix}"


def _indexed_constants(src: triton.compiler.ASTSource) -> Dict[int, str]:
    if not _IS_NEW_TRITON:
        return src.constants
    constants = {}
    for i, params in enumerate(src.fn.params):
        if params.is_constexpr:
            constants[i] = src.constants[params.name]
    return constants


def _indexed_signature(src: triton.compiler.ASTSource) -> Dict[int, str]:
    if not _IS_NEW_TRITON:
        return src.signature
    signature = {}
    for i, params in enumerate(src.fn.params):
        signature[i] = src.signature[params.name]
    return signature


def _equal_to_1(src: triton.compiler.ASTSource) -> Dict[int, str]:
    if not _IS_NEW_TRITON:
        return src.equal_to_1
    equal_to_1 = [x[0] for x in src.equal_to_1]
    return equal_to_1


def _make_const_sig(src: triton.compiler.ASTSource) -> str:
    constants = []
    indexed_constants = _indexed_constants(src)
    for i, params in enumerate(src.fn.params):
        if params.is_constexpr:
            constants.append(indexed_constants[i])
    return "x".join([str(v) for v in constants])


def materialize_c_params(
    kernel,
    ccinfo,
    kernel_name,
    out_name,
    num_warps,
    num_stages,
    grid: List[int],
):
    from triton.backends.nvidia.driver import ty_to_cpp

    src = ccinfo.src

    func_name = f"{out_name}_{kernel_name_suffix(src, num_stages, num_warps)}"
    attrs = src.attrs
    constants = _indexed_constants(src)  # key from int to str
    signature = _indexed_signature(src)
    equal_to_1 = _equal_to_1(attrs)
    const_sig = _make_const_sig(src)

    doc_string = [f"{kernel.arg_names[i]}={constants[i]}" for i in constants]
    doc_string += [f"num_warps={num_warps}", f"num_stages={num_stages}"]

    assert len(grid) == 3, f"{grid}"
    arg_names = []
    arg_types = []
    for i in signature.keys():
        if i not in equal_to_1 and not src.fn.params[i].is_constexpr:
            arg_names += [kernel.arg_names[i]]
            arg_types += [signature[i]]

    # dump C stub code
    hex_ = str(binascii.hexlify(ccinfo.asm["cubin"]))[2:-1]
    params = {
        "kernel_name": func_name,
        "triton_kernel_name": kernel_name,
        "bin_size": len(hex_),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join(
            [f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]
        ),
        "full_signature": ", ".join(
            [
                f"{ty_to_cpp(signature[i])} {kernel.arg_names[i]}"
                for i in signature.keys()
                if not src.fn.params[i].is_constexpr
            ]
        ),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args": len(arg_names),
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": num_warps,
        "algo_info": "_".join([const_sig, _meta_sig(num_stages, num_warps)]),
        "gridX": grid[0],
        "gridY": grid[1],
        "gridZ": grid[2],
        "_placeholder": "",
    }
    return params


def dump_c_code(out_path: Path, params):
    for ext in ["h", "c"]:
        template_path = Path(__file__).parent / f"compile.{ext}"
        with out_path.with_suffix(f".{ext}").open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))


if __name__ == "__main__":
    from typing import Any, Dict, List

    from flux_triton.kernels.moe_gather_rs import moe_gather_rs_grouped_gemm_kernel, signature

    def _materialize_constexpr(
        signature: str,
        grid: List[str],
        kernel: triton.JITFunction,
        triton_algo_infos: List[Dict[str, Any]],
    ):
        # replace %{constexpr} with concrete values
        for param in kernel.params:
            if param.is_constexpr:
                assert (
                    param.name in triton_algo_infos
                ), f"constexpr {param.name} not found in triton_algo_infos: {triton_algo_infos}"
                signature = signature.replace(f"%{param.name}", str(triton_algo_infos[param.name]))
                grid = [
                    x.replace(f"%{param.name}", str(triton_algo_infos[param.name])) for x in grid
                ]
        return signature, grid

    signature = signature.format(input_dtype="i8", output_dtype="bf16")
    _grid = [
        "((M + %BLOCK_SIZE_M - 1) / %BLOCK_SIZE_M) * ((N + %BLOCK_SIZE_N - 1) / %BLOCK_SIZE_N)",
        "1",
        "1",
    ]
    N_SPLITS = [1, 2, 4]  # TODO(houqi.1993) add more
    num_warps = 4
    num_stages = 4
    opts = {"num_warps": num_warps, "num_stages": num_stages}
    kernel = moe_gather_rs_grouped_gemm_kernel
    triton_algo_infos = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "num_warps": 4,
        "num_stages": 4,
        "N_SPLIT": 4,
    }
    signature, grid = _materialize_constexpr(signature, _grid, kernel, triton_algo_infos)
    src = make_ast_source(kernel, signature)
    ccinfo = triton.compile(src, options=opts)

    params = materialize_c_params(
        kernel,
        ccinfo,
        "output",
        "moe_gather_rs_grouped_gemm",
        num_warps,
        num_stages,
        grid,
    )
    func_kernel_suffix = kernel_name_suffix(src, num_stages, num_warps)
    dump_c_code(Path(f"output.{func_kernel_suffix}.c"), params)
