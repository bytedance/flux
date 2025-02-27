import argparse
import datetime
import itertools
import json
import logging
import pathlib
import time
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

import flux
from flux.testing import get_device_shared_memory_size, get_dram_gbps, get_tflops
from flux.triton.gemm_grouped_only import GemmGrouped

print = partial(print, flush=True)

SEARCH_SPACE_COMPUTE_BOUND = [
    (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps)
    for (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps) in itertools.product(
        [32, 64, 128, 256],  # BLOCK_SIZE_M
        [32, 64, 128, 256],  # BLOCK_SIZE_N
        [32, 64, 128, 256],  # BLOCK_SIZE_K
        [3, 4],  # num_stages
        [4, 8],  # num_warps
    )
]

SEARCH_SPACE_IO_BOUND = [
    (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps)
    for (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps) in itertools.product(
        [16, 32],  # BLOCK_SIZE_M,
        [32, 64, 128, 256],  # BLOCK_SIZE_N
        [32, 64],  # BLOCK_SIZE_K
        [2, 3, 4, 5, 6],  # num_stages
        [2, 4],  # num_warps
    )
    if BLOCK_SIZE_N <= 64 and num_warps == 2 or BLOCK_SIZE_N > 64 and num_warps == 4
]

SEARCH_SPACE_SMALL = [
    (128, 128, 128, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 128, 32, 4, 4),
    (128, 256, 128, 3, 8),
    (128, 256, 32, 3, 8),
    (128, 256, 64, 3, 8),
    (128, 32, 32, 4, 4),
    (128, 32, 64, 4, 4),
    (128, 64, 32, 4, 4),
    (128, 64, 64, 3, 4),
    (128, 64, 64, 4, 4),
    (256, 128, 128, 3, 8),
    (256, 128, 32, 3, 8),
    (256, 64, 128, 4, 4),
    (256, 64, 32, 4, 4),
    (32, 64, 32, 5, 2),
    (64, 128, 32, 4, 4),
    (64, 128, 64, 4, 4),
    (64, 256, 128, 4, 4),
    (64, 256, 32, 4, 4),
    (64, 32, 32, 5, 2),
    (64, 32, 64, 4, 4),
]

SEARCH_SPACE_CORE = [
    (128, 64, 32, 4, 4),
    (128, 64, 64, 3, 4),
    (128, 64, 64, 4, 4),
    (64, 128, 32, 4, 4),
    (64, 128, 64, 4, 4),
    (64, 256, 32, 4, 4),
]

SEARCH_SPACE_FULL = set(
    SEARCH_SPACE_COMPUTE_BOUND + SEARCH_SPACE_IO_BOUND + SEARCH_SPACE_SMALL + SEARCH_SPACE_CORE
)


def _shared_memory_used(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, dtype):
    shm_size_A = BLOCK_SIZE_M * BLOCK_SIZE_K * dtype.itemsize
    shm_size_B = BLOCK_SIZE_N * BLOCK_SIZE_K * dtype.itemsize
    return (shm_size_A + shm_size_B) * (num_stages - 1)


def batched(iter: Iterable, batch_size: int):
    output = []
    for n, value in enumerate(iter):
        output.append(value)
        if (n + 1) % batch_size == 0:
            yield output
            output = []
    if output:
        yield output


class DataBase:
    """

    in file:
        problem_size: f"{M}_{N}_{K}_{dtype}_{device_name}"
        config: f"{BLOCK_SIZE_M}_{BLOCK_SIZE_N}_{BLOCK_SIZE_K}_{num_stages}_{num_warps}"
        duration: float value

    in memory:
        problem_size: (M, N, K, dtype, device_name)
        config: (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps)
        duration: float value

    ```
    {
        "problem_size": {
            "config": duration,
        ...
        }
    }
    ```
    """

    @staticmethod
    def __from_config_key(config_key: str):
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps = map(
            int, config_key.split("_")
        )
        return BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps

    @staticmethod
    def __to_config_key(config: tuple):
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps = config
        return (
            f"{BLOCK_SIZE_M}_{BLOCK_SIZE_N}_{BLOCK_SIZE_K}_{GROUP_SIZE_M}_{num_stages}_{num_warps}"
        )

    @staticmethod
    def __from_problem_key(problem_key: str) -> Tuple[int, int, int, int, bool, bool, str, str]:
        M, N, K, E, trans_a, trans_b, dtype, device_name = problem_key.split("_")
        dtype = dtype.replace("-", "_")
        trans_a: bool = DataBase.__from_trans_key(trans_a)
        trans_b: bool = DataBase.__from_trans_key(trans_b)
        return int(M), int(N), int(K), int(E), trans_a, trans_b, dtype, device_name

    @staticmethod
    def __to_problem_key(problem_key: Tuple[int, int, int, int, bool, bool, str, str]) -> str:
        M, N, K, E, trans_a, trans_b, dtype, device_name = problem_key
        # Let's hope device_name has no `_` charactor
        dtype = dtype.replace("_", "-")
        trans_a = DataBase.__to_trans_key(trans_a)
        trans_b = DataBase.__to_trans_key(trans_b)
        return f"{M}_{N}_{K}_{E}_{trans_a}_{trans_b}_{dtype}_{device_name}"

    @staticmethod
    def load(database: str) -> Dict:
        with open(database, "r") as f:
            result = json.load(f)
        return {
            DataBase.__from_problem_key(problem_key): {
                DataBase.__from_config_key(config_key): duration
                for config_key, duration in perf_result.items()
            }
            for problem_key, perf_result in result.items()
        }

    def __to_trans_key(trans: bool):
        return "T" if trans else "N"

    def __from_trans_key(trans: str) -> bool:
        return {
            "N": False,
            "T": True,
        }[trans]

    @staticmethod
    def save(result: Dict, database: str):
        pathlib.Path.mkdir(pathlib.Path(database).parent, parents=True, exist_ok=True)
        with open(database, "w") as f:
            json.dump(
                {
                    DataBase.__to_problem_key(problem_size): {
                        DataBase.__to_config_key(config): duration
                        for config, duration in perf_result.items()
                    }
                    for problem_size, perf_result in result.items()
                },
                f,
                indent=2,
            )

    @staticmethod
    def merge(root, other, strategy="min"):
        # merge two records
        for problem_key, configs in other.items():
            if problem_key not in root:
                root[problem_key] = configs
            else:
                for config, duration in configs.items():
                    if config not in root[problem_key]:
                        root[problem_key][config] = duration
                    else:
                        if duration is float("inf"):
                            continue
                        duration_old = root[problem_key][config]
                        if duration / duration_old > 1.03 or duration_old / duration > 1.03:
                            logging.warning(
                                f"{problem_key} {config} value not stable: {duration_old:0.3f} vs {duration:0.3f}"
                            )
                            continue
                        root[problem_key][config] = min(duration_old, duration)
        return root


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-commands - commands help")

    perf_parser = subparsers.add_parser("perf", help="perf gemm")
    perf_parser.add_argument("-M", type=int, default=8192)
    perf_parser.add_argument("-N", type=int, default=1024)
    perf_parser.add_argument("-K", type=int, default=8192)
    perf_parser.add_argument("-E", type=int, default=32)
    perf_parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=list(DTYPE_MAP.keys()),
    )
    perf_parser.add_argument("--trans_a", default=False, action="store_true")
    perf_parser.add_argument("--trans_b", default=False, action="store_true")
    perf_parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    perf_parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    perf_parser.add_argument(
        "--space",
        default="small",
        choices=["core", "small", "full"],
        help=f"search space, core search space size {len(SEARCH_SPACE_CORE)};"
        + f" small search space {len(SEARCH_SPACE_SMALL)};"
        + f" full search space {len(SEARCH_SPACE_FULL)}",
    )
    perf_parser.add_argument("--verbose", action="store_true", help="verbose mode", default=False)
    perf_parser.add_argument("--database", type=str, default="merged.json", help="database file")
    perf_parser.add_argument("--workspace", default="workspace")
    perf_parser.add_argument("--cache", default=True, action=argparse.BooleanOptionalAction)
    perf_parser.add_argument("--group_size_m", default=4, type=int)
    perf_parser.add_argument("--print-only", action="store_true", default=True)
    perf_parser.set_defaults(func=perf_main)

    merge_parser = subparsers.add_parser("merge", help="merge all perf results")
    merge_parser.add_argument("--workspace", default="workspace")
    merge_parser.set_defaults(func=merge_main)

    analysis_parser = subparsers.add_parser("analysis", help="analysis perf results")
    analysis_parser.add_argument("--workspace", default="workspace")
    analysis_parser.add_argument("--database", default="merged.json")
    analysis_parser.add_argument(
        "--dtype",
        default="all",
        choices=["all"] + list(DTYPE_MAP.keys()),
    )
    analysis_parser.add_argument(
        "--space",
        default="small",
        choices=["core", "small", "full"],
        help=f"search space, core search space size {len(SEARCH_SPACE_CORE)};"
        + f" small search space {len(SEARCH_SPACE_SMALL)};"
        + f" full search space {len(SEARCH_SPACE_FULL)}",
    )
    analysis_parser.set_defaults(func=analysis_main)
    return parser.parse_args()


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}

OUTPUT_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int32,
    "float8_e4m3fn": torch.bfloat16,
    "float8_e5m2": torch.bfloat16,
}


def _perf_with_config(
    A: torch.Tensor,
    B: torch.Tensor,
    splits: Sequence[int],
    C: torch.Tensor,
    config: Tuple[int, int, int, int, int, int],
    warmup: int,
    iters: int,
    verbose: bool = False,
):
    op = GemmGrouped()
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_stages, num_warps = config
    device = torch.cuda.current_device()
    A = A.to(device)
    B = B.to(device)
    C = C.to(device)
    params = {
        "output": C,  # no memory allocation
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": BLOCK_SIZE_N,
        "BLOCK_SIZE_K": BLOCK_SIZE_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_stages": num_stages,
        "num_warps": num_warps,
    }
    func = lambda: op.forward(A, B, splits, **params)
    _, duration = flux.util.bench_func(func, warmup, iters)
    if verbose:
        logging.info(f"{config}: {duration:0.3f} ms at {torch.cuda.current_device()}")
    return duration


def perf_with_config_at_device(
    device_id: int,
    A: torch.Tensor,
    B: torch.Tensor,
    splits: Sequence[int],
    C: torch.Tensor,
    config: Tuple[int, int, int, int, int, int],
    warmup: int,
    iters: int,
    verbose: bool = False,
):
    time.sleep(1)
    try:
        with torch.cuda.device(device_id):
            duration = _perf_with_config(
                A,
                B,
                splits,
                C,
                config,
                warmup,
                iters,
                verbose,
            )
    except Exception as e:
        if verbose:
            logging.error(f"{config}: {e}")
        duration = float("inf")
    return duration


def perf(
    M: int,
    N: int,
    K: int,
    E: int,
    trans_a: bool,
    trans_b: bool,
    input_dtype: torch.dtype,
    search_spaces: List[Dict],
    warmup: int,
    iters: int,
    verbose: bool = False,
) -> Dict[Tuple[int, int, int, int, int], float]:
    output_dtype = {
        torch.bfloat16: torch.bfloat16,
        torch.float16: torch.float16,
        torch.int8: torch.int32,
        torch.float8_e4m3fn: torch.bfloat16,
        torch.float8_e5m2: torch.bfloat16,
    }[input_dtype]
    A_shape = (M, K) if not trans_a else (K, M)
    B_shape = (E, K, N) if not trans_b else (E, N, K)
    splits = [M // E for _ in range(E)]
    C_shape = (M, N)
    if input_dtype in [torch.float16, torch.bfloat16]:
        A = torch.randn(A_shape, dtype=input_dtype, device="cuda")
        B = torch.randn(B_shape, dtype=input_dtype, device="cuda")
        C = torch.randn(C_shape, dtype=output_dtype, device="cuda")
    elif input_dtype in [torch.int8]:
        A = torch.randint(-127, 127, A_shape, dtype=input_dtype, device="cuda")
        B = torch.randint(-127, 127, B_shape, dtype=input_dtype, device="cuda")
        C = torch.zeros(C_shape, dtype=output_dtype, device="cuda")
    elif input_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        A = torch.randn(A_shape, dtype=torch.float16, device="cuda").to(input_dtype)
        B = torch.randn(B_shape, dtype=torch.float16, device="cuda").to(input_dtype)
        C = torch.randn(C_shape, dtype=output_dtype, device="cuda")
    else:
        assert False, f"not support dtype: {input_dtype}"

    A = A if not trans_a else A.t()
    B = B if not trans_b else B.transpose(1, 2)

    result = {}
    device_count = torch.cuda.device_count()
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=device_count) as executor:
        for configs in batched(search_spaces, device_count):
            duration_list = executor.map(
                lambda x: perf_with_config_at_device(
                    x[0],
                    A,
                    B,
                    splits,
                    C,
                    x[1],
                    warmup,
                    iters,
                    verbose,
                ),
                enumerate(configs),
            )
            for config, duration in zip(configs, duration_list):
                result[config] = duration
    return result


def _calc_problem_sol_time(
    M: int,
    N: int,
    K: int,
    E: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    read_bytes = (M * K + E * K * N) * input_dtype.itemsize
    write_bytes = (M * N) * output_dtype.itemsize
    gflops = M * N * K * 2 / 1e9

    dram_bw = get_dram_gbps()
    tflops_hw = get_tflops(input_dtype)
    time_c, time_r, time_w = (
        gflops / tflops_hw,
        read_bytes / 1e6 / dram_bw,
        write_bytes / 1e6 / dram_bw,
    )
    return time_c, time_r, time_w


def perf_main(args):
    search_space = {
        "core": set(SEARCH_SPACE_CORE),
        "small": set(SEARCH_SPACE_SMALL),
        "full": set(SEARCH_SPACE_FULL),
    }[args.space]

    problem_key = (
        args.M,
        args.N,
        args.K,
        args.E,
        args.trans_a,
        args.trans_b,
        args.dtype,
        torch.cuda.get_device_name(),
    )
    if args.cache:
        try:
            perf_history = DataBase.load(Path(args.workspace) / args.database)
            perf_records = perf_history.get(problem_key, {})
        except Exception as e:
            logging.error(f"Failed to load perf history from {args.database}: {e}")
            perf_history, perf_records = {}, {}
    else:
        perf_history, perf_records = {}, {}
    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = OUTPUT_DTYPE_MAP[args.dtype]
    print(f"search_space size: {len(search_space)}")
    search_space = {
        (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, args.group_size_m, num_stages, num_warps)
        for (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps) in search_space
        if _shared_memory_used(BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, input_dtype)
        <= get_device_shared_memory_size()
    }
    print(f"search space after shared_memory prune: {len(search_space)}")
    search_space = search_space - set(perf_records.keys())

    assert args.M % args.E == 0, f"M must be divisible by E, got {args.M} and {args.E}"

    time_c, time_r, time_w = _calc_problem_sol_time(
        args.M, args.N, args.K, args.E, input_dtype, output_dtype
    )
    print(
        f"""SOL time for GEMM Grouped: MNK: {args.M}x{args.N}x{args.K} with E={args.E}
          TensorCore time: {time_c:0.3f} ms
          read  Memory time: {time_r:0.3f} ms
          write Memory time: {time_w:0.3f} ms"""
    )
    perf_out = perf(
        args.M,
        args.N,
        args.K,
        args.E,
        args.trans_a,
        args.trans_b,
        input_dtype,
        search_space,
        args.warmup,
        args.iters,
        args.verbose,
    )
    perf_records.update(perf_out)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    duration_best = min(list(perf_records.values()))
    config = [config for config in perf_records if perf_records[config] == duration_best][0]
    tflops = args.M * args.N * args.K * 2 / duration_best / 1e9
    bytes_r = (args.K * args.M + args.N * args.K * args.E) * input_dtype.itemsize
    bytes_w = args.M * args.N * output_dtype.itemsize
    memband = (bytes_r + bytes_w) / duration_best / 1e6
    print(
        f"{problem_key} best config: {config} with duration {duration_best:0.3f} ms {tflops:0.1f} TFLOPS. {memband:0.2f} GB/s"
    )

    if not search_space:
        print(f"warning: all config items found in cache. skip update perf database...")
        return

    def _to_trans_attr(trans):
        return "T" if trans else "N"

    trans_tag = f"{_to_trans_attr(args.trans_a)}{_to_trans_attr(args.trans_b)}"
    outfile = f"matmul-perf-{args.M}-{args.N}-{args.K}-{trans_tag}-{args.dtype}-{args.space}-{timestamp}.json"
    # save to seperate database
    DataBase.save({problem_key: perf_records}, Path(args.workspace) / "detail" / outfile)
    # save to merged database
    perf_history = DataBase.merge(perf_history, {problem_key: perf_records})
    DataBase.save(perf_history, Path(args.workspace) / args.database)


def merge_main(args):
    def _stat(record):
        # record: {problem_key: {config: duration}}
        return len(record), sum([len(v) for _, v in record.items()])

    workspace = args.workspace
    merged = {}

    json_files: List[Path] = pathlib.Path(workspace).glob("detail/*.json")
    for json_file in json_files:
        if json_file.name in ["merged.json"]:
            continue
        try:
            logging.info(f"load {json_file}")
            out = DataBase.load(json_file)
            problem_count, record_count = _stat(out)
            problem_count_total, record_count_total = _stat(merged)
            merged = DataBase.merge(merged, out)
            logging.info(
                f"load {json_file} with {problem_count} / {record_count} records to {problem_count_total} / {record_count_total} records"
            )
        except Exception as e:
            logging.exception(f"load {json_file} failed: {e}")

    DataBase.save(merged, Path(workspace) / "merged.json")


def analysis_main(args):
    def _in_space(config):
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, _, num_stages, num_warps = config

        return (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_stages, num_warps) in search_space

    search_space = {
        "core": set(SEARCH_SPACE_CORE),
        "small": set(SEARCH_SPACE_SMALL),
        "full": set(SEARCH_SPACE_FULL),
    }[args.space]

    perf_results = DataBase.load(Path(args.workspace) / args.database)

    groups = set(
        [
            (dtype, trans_a, trans_b, device_name)
            for _, _, _, _, trans_a, trans_b, dtype, device_name in perf_results.keys()
            if (args.dtype == dtype or args.dtype == "all")
        ]
    )
    config_list = None
    for group in groups:
        logging.info(group)
        dtype, trans_a, trans_b, device_name = group
        perf_results_mnk = {
            (M, N, K, E): {
                config: duration for config, duration in value.items() if _in_space(config)
            }
            for (
                M,
                N,
                K,
                E,
                trans_a_,
                trans_b_,
                dtype_,
                device_name_,
            ), value in perf_results.items()
            if (trans_a_, trans_b_, dtype_, device_name_) == (trans_a, trans_b, dtype, device_name)
        }  # (m,n,k) -> ((config) -> duration)

        def _sorted_record(record: Dict[Tuple, float], sorted_keys: List[Tuple]):
            return [record.get(key, float("inf")) for key in sorted_keys]

        def _sort_dict(dict_):
            return dict(sorted(dict_.items()))

        def _config_as_key(config):
            (
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
                GROUP_SIZE_M,
                num_stages,
                num_warps,
            ) = config
            return f"{BLOCK_SIZE_M}x{BLOCK_SIZE_N}x{BLOCK_SIZE_K}@{GROUP_SIZE_M}_{num_stages}stages{num_warps}warps"

        def _problem_as_key(problem):
            M, N, K, E = problem
            return f"{M}x{N}x{K}@{E}"

        perf_out = list(perf_results_mnk.values())[0]
        config_list = sorted(list(perf_out.keys()))
        input_dtype, output_dtype = DTYPE_MAP[dtype], OUTPUT_DTYPE_MAP[dtype]

        perf_results_mnk_rel = {
            _problem_as_key((M, N, K, E)): [
                *_calc_problem_sol_time(M, N, K, E, input_dtype, output_dtype),
                *_sorted_record(value, config_list),
            ]
            for (M, N, K, E), value in perf_results_mnk.items()
        }
        perf_results_mnk_rel = _sort_dict(perf_results_mnk_rel)
        import pandas as pd

        columns = [_config_as_key(x) for x in config_list]
        frame = pd.DataFrame.from_dict(
            {k: v for k, v in perf_results_mnk_rel.items()},
            columns=["compute(ms)", "read(ms)", "write(ms)"] + columns,
            orient="index",
        )
        frame["best"] = frame[columns].min(axis=1)
        frame_rel = frame[columns].div(frame["best"], axis="index")
        frame_rel.rename(columns={k: k + "_rel" for k in columns}, inplace=True)
        frame = pd.concat((frame, frame_rel), axis=1)
        frame = frame.transpose()
        frame["average"] = frame.mean(numeric_only=True, axis=1)
        columns_rel = [x + "_rel" for x in columns]
        index = frame.loc[columns_rel]["average"].argsort()[:30]
        frame = pd.concat(
            (
                frame[:3],
                frame.loc[["best"]],
                frame.loc[columns].iloc[index],
                frame.loc[columns_rel].iloc[index],
            ),
            axis=0,
        )
        print(frame)
        device_name = device_name.replace(" ", "-")
        csv_file = f"gemm_grouped_{dtype}_{trans_a}_{trans_b}_{device_name}.csv"
        frame.to_csv(Path(args.workspace) / csv_file, index=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    args = parse_args()
    args.func(args)
