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

import argparse
import logging
import pathlib
import re
from pathlib import Path
from pprint import pprint

cpp_license_pattern = """//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//"""

py_license_pattern = """################################################################################
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
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Check license")
    parser.add_argument("--home", default=".")
    parser.add_argument("--fix", action="store_true", default=False)
    return parser.parse_args()


def has_correct_cpp_license(infile):
    """
    relax_length_check: don't force header with correct len
    """
    with open(infile, "r") as f:
        content = f.read()

    content = content.lstrip()
    lines = content.split("\n")
    header = lines[0]
    content = "\n".join(lines[1:])
    header_pattern = r"//===- {} -* ?C\+\+ -+=+//".format(Path(infile).name)

    return content.startswith(cpp_license_pattern) and re.match(header_pattern, header)


def fix_cpp_license(infile):
    def _is_cpp_license(content):
        return content.strip().startswith("//===-")

    def _make_header():
        header_part0 = "//===- {} ".format(Path(infile).name)
        header_part2 = "- C++ ---===//"
        header_part1_len = 80 - len(header_part0) - len(header_part2)
        if len(header_part0) + len(header_part2) > 80:
            logging.warning(f"C++ file may have illegal format license, skip fix {infile}")
            return False
        else:
            header_part1 = "-" * header_part1_len
            header = header_part0 + header_part1 + header_part2
        return header

    content = infile.read_text()
    header = _make_header()
    if not _is_cpp_license(content) and header:
        content = header + "\n" + cpp_license_pattern + "\n\n" + content
        logging.info(f"auto fix C++ license for {infile}")
        infile.write_text(content)
        return True
    else:
        logging.warning(f"C++ file may have illegal format license, skip fix {infile}")
        return False


def has_correct_py_license(infile):
    with open(infile, "r") as f:
        content = f.read()

    content = content.lstrip()
    return content.startswith(py_license_pattern)


def fix_py_license(infile: Path):
    content = infile.read_text()
    if not content.strip().startswith("#" * 80):
        content = py_license_pattern + "\n" + content
        logging.info(f"auto fix Python license for {infile}")
        infile.write_text(content)
    else:
        logging.warning("Python file may have license")


def _in_cpp_whilelist(infile):
    cpp_whitelist = [
        "sm89_all_gather_fp8_gemm_with_absmax.hpp",
        "default_epilogue_tensor_op.h",
        "sm90_tile_scheduler_group_ag_scatter.hpp",
        "gemm_with_groupwise_scaling.h",
    ]
    return (
        infile.name in cpp_whitelist
        or str(infile).find("cutlass_impls") >= 0
        or str(infile).find("workspace") >= 0
        or str(infile).find("triton_aot_generated") >= 0
    )


def _in_py_whitelist(infile):
    py_whilelist = ["version.py"]
    return infile.name in py_whilelist


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    home = Path(args.home)
    cpp_extensions = [".hpp", ".h", ".cc", ".cpp", ".cu"]
    py_extensions = [".py", "pyi"]
    subdirs = ["include", "src", "test", "python/flux"]
    cpp_files = []
    for cpp_ext in cpp_extensions:
        for subdir in subdirs:
            path = pathlib.Path(home) / subdir
            cpp_files.extend(path.glob(f"**/*{cpp_ext}"))
    cpp_files = [x for x in cpp_files if not _in_cpp_whilelist(x)]
    failed_cpp_files = [cpp_file for cpp_file in cpp_files if not has_correct_cpp_license(cpp_file)]
    if failed_cpp_files:
        print(
            f"those C++ files don't have correct license:\n",
            "\n".join([" " + str(x) for x in failed_cpp_files]),
        )
        if not args.fix or any([not fix_cpp_license(x) for x in failed_cpp_files]):
            raise Exception("Failed to check cpp license")

    py_files = []
    for py_ext in py_extensions:
        for subdir in subdirs:
            path = pathlib.Path(home) / subdir
            py_files.extend(path.glob(f"**/*{py_ext}"))
    py_files = [x for x in py_files if not _in_py_whitelist(x)]
    failed_py_files = [py_file for py_file in py_files if not has_correct_py_license(py_file)]
    if failed_py_files:
        print(
            f"those Python files don't have correct license:\n",
            "\n".join([" " + str(x) for x in failed_py_files]),
        )
        if not args.fix or any([not fix_py_license(x) for x in failed_py_files]):
            raise Exception("Failed to check py license")
