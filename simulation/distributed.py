"""Small torch.distributed helpers shared by LookStep evaluators."""

import builtins
import datetime
import os

import torch
import torch.distributed as dist


def setup_for_distributed(is_master: bool) -> None:
    """Suppress normal prints on workers while retaining ``force=True``."""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}] ", end="")
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def init_distributed_mode(args) -> None:
    """Initialize from ``torchrun`` environment variables when present."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.local_rank = args.gpu
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)
        args.distributed = False
        return

    args.distributed = True
    use_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl" if use_cuda else "gloo"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, "
        f"backend={args.dist_backend}, local_rank={args.gpu}",
        flush=True,
    )
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(hours=2),
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)
