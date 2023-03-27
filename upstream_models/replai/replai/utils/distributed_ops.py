import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def init_distributed_environment(gpu, ngpus_per_node, args):
    if args.environment.dist_url == "env://" and args.environment.rank == -1:
        args.environment.rank = int(os.environ["RANK"])
    if args.environment.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.environment.rank = args.environment.rank * ngpus_per_node + gpu
    dist.init_process_group(
        backend=args.environment.dist_backend,
        init_method=args.environment.dist_url,
        world_size=args.environment.world_size,
        rank=args.environment.rank,
    )

    if args.environment.gpu is not None:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        torch.cuda.set_device(args.environment.gpu)

        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.optim.batch_size = int(args.optim.batch_size / ngpus_per_node)
        args.environment.workers = int(
            (args.environment.workers + ngpus_per_node - 1) / ngpus_per_node
        )
    return args


def send_to_device(module, distributed, device=None):
    if distributed:
        module.cuda(device)
        module = DistributedDataParallel(
            module, device_ids=[device] if device is not None else None
        )
    elif device is not None:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    # else:
    #     module.cuda()
    return module


def shuffle_batch(x1, x2):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    if not dist.is_initialized():
        batch_size_all = x1.shape[0]
        idx_shuffle = torch.randperm(batch_size_all).to(x1.device)
        return x1[idx_shuffle], x2[idx_shuffle]

    else:
        # gather from all gpus
        batch_size_this = x1.shape[0]
        x1_gather = concat_all_gather(x1)
        x2_gather = concat_all_gather(x2)
        batch_size_all = x1_gather.shape[0]

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).to(x1.device)

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        num_gpus = batch_size_all // batch_size_this
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        # shuffle
        return x1_gather[idx_this], x2_gather[idx_this]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: dist.all_gather has no gradient.
    """
    if not dist.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
