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
import datetime
import logging
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

"""
CREATE TABLE flux_perf(
  id BIGINT UNSIGNED NOT NULL auto_increment PRIMARY KEY COMMENT 'the key',
  op_name VARCHAR(32) not null COMMENT 'such as "gemm_rs"/"ag_gemm"',
  exp_name VARCHAR(64) not null COMMENT 'such as "flux #rank0", "torch #rank0"',
  url VARCHAR(256) not null COMMENT 'such as "123456"',
  compile_args VARCHAR(1024) not null COMMENT 'nvshmem/ipc',
  hardware VARCHAR(64) not null COMMENT 'Nvidia_L20. from nvidia-smi',
  runtime_args VARCHAR(256) not null COMMENT 'such as "py311_cu124_th24"',
  shape VARCHAR(256) not null COMMENT 'such as "M_8192_N_8192_K_8192"',
  dtype VARCHAR(256) not null COMMENT 'such as "float16"',
  extra_args VARCHAR(1024) not null COMMENT 'from parse_args + world_size from env',
  timestamp DATETIME not null COMMENT 'perf timestamp',
  duration numeric(8, 4) not null COMMENT 'the duration',
  commit_args JSON not null COMMENT 'code branch, commit id, author or so',
  compile_args_detail JSON not null COMMENT 'with nvshmem or not',
  runtime_args_detail JSON not null COMMENT 'such as container',
  environment_variables JSON not null COMMENT 'such as CUDA_MAX_CONNECTIONS',
  extra_args_detail JSON not null COMMENT 'from parse_args + world_size from env'
)
ENGINE=InnoDB DEFAULT CHARSET=utf8 comment 'perf database for FLUX'
"""

_DB_ENGINE = None
_DB_PSM = "toutiao.mysql.flux_ci_perf_staging_write"
_DB_TABLE_BYTED = "flux_perf"  #  Don't pollute this RDS table. only for gitlab CI.
_DB_TABLE_GITHUB = "flux_perf_github"  # Don't pollute this RDS table. only for github CI.
_DB_TABLE_TEST = "flux_perf_test"  # flux_perf_test: Do anything you like to this
_DB_TABLE = os.getenv("FLUX_PERF_TABLE", _DB_TABLE_TEST)
_OP_NAME = None
_SHAPE_ARGS, _DTYPE_ARGS, _EXTRA_ARGS, _FULL_ARGS = None, None, None, None


def _sort_dict(dict_):
    return dict(sorted(dict_.items()))


def _get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))


def set_global_args(op_name, args):
    "to move shape_args and dtype_args from log_perf interface and save some dup code"
    global _OP_NAME, _SHAPE_ARGS, _DTYPE_ARGS, _EXTRA_ARGS, _FULL_ARGS
    assert (
        _OP_NAME is None
        and _SHAPE_ARGS is None
        and _DTYPE_ARGS is None
        and _EXTRA_ARGS is None
        and _FULL_ARGS is None
    ), "set_global_args() can only be called once"
    SHAPE_FIELDS = [
        "M",
        "N",
        "K",  # for dense & Moe Gather+RS
        "B",
        "S",
        "H",
        "ffn_hidden_size",  # for Moe AG+Scatter
        "G",  # num experts
        "topk",  # topk
        "E",  # EP size
        "T",  # TP size
        "input_groups",  # for MOE Gather+RS
        "weight_groups",  # for MOE AG+Scatter
    ]
    DTYPE_FIELDS = ["dtype"]
    FILTER_FIELDS = [
        "profile",
        "dist",
        "iters",
        "warmup",
        "triton",
        "lego",
        "triton_ony",
        "debug",
        "tune",
    ]  # those fields should never affect the performance
    args = args.__dict__
    _SHAPE_ARGS = _sort_dict({k: v for k, v in args.items() if k in SHAPE_FIELDS})
    _DTYPE_ARGS = _sort_dict({k: v for k, v in args.items() if k in DTYPE_FIELDS})
    _EXTRA_ARGS = _sort_dict(
        {k: v for k, v in args.items() if k not in DTYPE_FIELDS + SHAPE_FIELDS + FILTER_FIELDS}
    )
    _FULL_ARGS = _sort_dict(dict(args.items()))
    _EXTRA_ARGS.update({"TP": _get_world_size()})
    _OP_NAME = op_name
    assert _OP_NAME in ["gemm_rs", "ag_gemm", "moe_gather_rs", "moe_ag_scatter"]


def _db_engine():
    global _DB_ENGINE
    from bytedmysql import sqlalchemy_init
    from bytedmysql.sqlalchemy.dialect import BytedMySQLDialect
    from sqlalchemy import create_engine

    BytedMySQLDialect.supports_statement_cache = True

    if _DB_ENGINE is None:
        sqlalchemy_init()
        _DB_ENGINE = create_engine(f"mysql+pymysql://:@/?charset=utf8mb4&&db_psm={_DB_PSM}")
    return _DB_ENGINE


def should_log_to_rds():
    if os.getenv("CUDA_LAUNCH_BLOCKING", "0") == "1":
        logging.warning(f"with CUDA_LAUNCH_BLOCKING=1, no data with be logged to RDS")
    return os.getenv("FLUX_LOG_TO_RDS", "0") == "1"


def _commit_args() -> Dict[str, str]:
    return {
        "CI_ACTOR": os.getenv("CI_ACTOR"),
        "CI_HEAD_BRANCH": os.getenv("CI_HEAD_BRANCH"),
        "CI_HEAD_SHA": os.getenv("CI_HEAD_SHA"),
        "CI_JOB_ID": os.getenv("CI_JOB_ID"),
        "CI_JOB_IMAGE": os.getenv("CI_JOB_IMAGE"),
        "CI_JOB_NAME": os.getenv("CI_JOB_NAME"),
        "CI_PIPELINE_NAME": os.getenv("CI_PIPELINE_NAME"),
        "CI_PIPELINE_RUN_ID": os.getenv("CI_PIPELINE_RUN_ID"),
        "CI_REPO_NAME": os.getenv("CI_REPO_NAME"),
        "CI_REPO_WORKSPACE": os.getenv("CI_REPO_WORKSPACE"),
        "CI_STEP_ID": os.getenv("CI_STEP_ID"),
        "CI_STEP_NAME": os.getenv("CI_STEP_NAME"),
        "CI_EVENT_CHANGE_URL": os.getenv("CI_EVENT_CHANGE_URL", ""),
    }


def _cuda_version() -> Tuple[int, ...]:
    """CUDA Toolkit version as a (major, minor) by nvcc --version"""

    # Try finding NVCC
    nvcc_bin: Optional[Path] = None
    if nvcc_bin is None and os.getenv("CUDA_HOME"):
        # Check in CUDA_HOME
        cuda_home = Path(os.getenv("CUDA_HOME"))
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if nvcc_bin is None:
        # Check if nvcc is in path
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin is not None:
            nvcc_bin = Path(nvcc_bin)
    if nvcc_bin is None:
        # Last-ditch guess in /usr/local/cuda
        cuda_home = Path("/usr/local/cuda")
        nvcc_bin = cuda_home / "bin" / "nvcc"
    if not nvcc_bin.is_file():
        raise FileNotFoundError(f"Could not find NVCC at {nvcc_bin}")

    # Query NVCC for version info
    output = subprocess.run(
        [nvcc_bin, "-V"],
        capture_output=True,
        check=True,
        universal_newlines=True,
    )
    match = re.search(r"release\s*([\d.]+)", output.stdout)
    version = match.group(1).split(".")
    return tuple(int(v) for v in version)


def _hardware() -> str:
    return torch.cuda.get_device_name().replace(" ", "_").lower()


def _is_perf_related_envs(key):
    return (key.startswith("CUDA_") and key not in ["CUDA_HOME", "CUDA_PATH"]) or (
        key.startswith("FLUX_")
        and key not in ["FLUX_LOG_TO_RDS", "FLUX_PERF_TABLE", "FLUX_EXTRA_TORCHRUN_ARGS"]
    )


@lru_cache
def _environment_variables():
    envs = _sort_dict(
        {key: value for key, value in os.environ.items() if _is_perf_related_envs(key)}
    )
    return envs


@lru_cache
def _runtime_args_detail():
    return {
        "torch": torch.__version__,
        "cuda": ".".join([str(x) for x in _cuda_version()]),
        "python": sys.version.split(" ")[0],  # return things like "3.11.2"
    }


@lru_cache
def _runtime_args():
    args = _runtime_args_detail()
    return "py{}_cu{}_th{}".format(
        "".join(args["python"].split(".")[:2]),
        "".join(args["cuda"].split(".")[:2]),
        "".join(args["torch"].split(".")[:2]),
    )


def _compile_with_nvshmem() -> bool:
    ci_job_name = os.getenv("CI_JOB_NAME", "")
    if ci_job_name.lower().find("nvshmem") >= 0:
        return True
    if ci_job_name.lower().find("ipc") >= 0:
        return False
    return True


def _compile_args_detail():
    return {
        "BUILD_WITH_NVSHMEM": _compile_with_nvshmem(),
    }


def _compile_args():
    if _compile_with_nvshmem():
        return "nvshmem"
    else:
        return "ipc"


def _extract_perf_result(perf_result) -> Tuple[str, float]:
    perf_result = perf_result.__dict__
    name = perf_result.get("name", "unknown")
    duration = perf_result.get("total_ms", 0.0)
    return name.replace(" ", ""), float(duration)


def _as_string(dict):
    def _prettify(v):
        if isinstance(v, bool):
            return "1" if v else "0"
        return str(v)

    return "_".join([f"{k}={_prettify(v)}" for k, v in dict.items()])


def _table_cls(table_name):
    from sqlalchemy import DATETIME, INT, JSON, NUMERIC, VARCHAR, Column
    from sqlalchemy.orm import declarative_base

    Base = declarative_base()

    # 定义User对象:
    class PerfItem(Base):
        # 表的名字:
        __tablename__ = _DB_TABLE

        id = Column(INT, primary_key=True)
        op_name = Column(VARCHAR(32))  # such as "gemm_rs"/"ag_gemm"
        exp_name = Column(VARCHAR(64))  # such as "flux #rank0", "torch #rank0"
        url = Column(VARCHAR(256))
        compile_args = Column(VARCHAR(1024))
        hardware = Column(VARCHAR(64))
        runtime_args = Column(VARCHAR(256))
        shape = Column(VARCHAR(256))
        dtype = Column(VARCHAR(256))
        extra_args = Column(VARCHAR(1024))
        timestamp = Column(DATETIME)
        duration = Column(NUMERIC(18, 2))
        # details fields: for API query
        commit_args = Column(JSON)
        compile_args_detail = Column(JSON)
        runtime_args_detail = Column(JSON)
        environment_variables = Column(JSON)
        extra_args_detail = Column(JSON)

        def __repr__(self) -> str:
            return (
                f"op_name: {self.op_name}, "
                f"exp_name: {self.exp_name}, "
                f"url: {self.url}, "
                f"compile_args: {self.compile_args}, "
                f"hardware: {self.hardware}, "
                f"runtime_args: {self.runtime_args}, "
                f"shape: {self.shape}, "
                f"dtype: {self.dtype}, "
                f"extra_args: {self.extra_args}, "
                f"timestamp: {self.timestamp}, "
                f"duration: {self.duration}, "
            )

    return PerfItem


def upload_to_rds(perf_result):
    from sqlalchemy.orm import sessionmaker

    PerfItem = _table_cls(_DB_TABLE)
    name, duration = _extract_perf_result(perf_result)
    item = PerfItem(
        op_name=_OP_NAME,
        exp_name=name,
        url=_commit_args()["CI_EVENT_CHANGE_URL"],
        compile_args=_compile_args(),
        hardware=_hardware(),
        runtime_args=_runtime_args(),
        shape=_as_string(_SHAPE_ARGS),
        dtype=_as_string(_DTYPE_ARGS),
        extra_args=_as_string(_EXTRA_ARGS),
        timestamp=datetime.datetime.now(),
        duration=duration,
        commit_args=_commit_args(),
        compile_args_detail=_compile_args_detail(),
        runtime_args_detail=_runtime_args_detail(),
        environment_variables=_environment_variables(),
        extra_args_detail=_FULL_ARGS,
    )
    logging.debug(item)

    engine = _db_engine()
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    session.add(item)
    session.commit()
    session.close()


def query(delete: bool = False):
    """sample if you want to query and delete some records"""
    from sqlalchemy.orm import sessionmaker

    PerfItem = _table_cls(_DB_TABLE)
    engine = _db_engine()
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    query = (
        session.query(PerfItem)
        .filter(PerfItem.environment_variables["CUDA_LAUNCH_BLOCKING"].as_string() == "1")
        .filter(PerfItem.duration == 3.0)
    )
    session.commit()
    for out in query.all():
        print(out)
    if delete:
        query.delete()
        session.commit()
    session.close()


def log_perf_stdout(perf_result):
    print(perf_result)


def log_perf_rds(perf_result):
    logging.debug(f"log to RDS")
    upload_to_rds(perf_result)


def log_perf(perf_result):
    log_perf_stdout(perf_result)
    if should_log_to_rds():
        try:
            log_perf_rds(perf_result)
        except Exception as e:
            logging.exception(f"Failed to upload perf result to RDS: {e}")


class PerfResult:
    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        gathered_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
        time3: str = "gemm_only",
        gemm_only_time_ms: float = 0,
    ) -> None:
        self.name = name
        self.output = output
        self.gathered_output = gathered_output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
        self.time3 = time3
        self.gemm_only_time_ms = gemm_only_time_ms

    def __repr__(self) -> str:
        if self.gemm_only_time_ms == 0.0:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms"
            )
        else:
            return (
                f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms, {self.time3} {self.gemm_only_time_ms:.3f} ms"
            )


def _mock_ag_gemm_args():
    args = argparse.Namespace()
    setattr(args, "dtype", "s8")
    setattr(args, "M", 512)
    setattr(args, "N", 1024)
    setattr(args, "K", 8192)
    setattr(args, "iters", 10)
    setattr(args, "warmup", 5)
    setattr(args, "iters", 100)
    setattr(args, "transpose_weight", True)
    setattr(args, "has_bias", True)
    setattr(args, "fastacc", False)
    setattr(args, "ring_mode", "auto")
    setattr(args, "verify", False)
    setattr(args, "use_cuda_core_local", True)
    setattr(args, "use_cuda_core_ag", True)
    setattr(args, "triton", True)
    setattr(args, "gather_input", True)
    setattr(args, "debug", False)
    setattr(args, "profile", False)
    return args


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--op", type=str, choices=["query", "log_perf"], default="log_perf")
        parser.add_argument("--delete", action="store_true", default=False)
        return parser.parse_args()

    args = parse_args()

    if "query" == args.op:
        query(args.delete)
    elif args.op == "log_perf":
        set_global_args("ag_gemm", _mock_ag_gemm_args())
        perf_res = PerfResult("flux #rank 0", None, None, 3.0, "gemm", 0.0, "comm", 0.0)
        log_perf(perf_res)
