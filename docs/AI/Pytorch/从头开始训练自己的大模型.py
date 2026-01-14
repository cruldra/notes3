import os
import random

import numpy as np
import torch
import torch.distributed as dist


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    device = None
    local_rank = init_distributed_mode()
    if dist.is_initialized(): device = f"cuda:{local_rank}"
    print(f"local_rank: {local_rank}, device: {device}")
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))