# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
from .pos_embed import get_3d_sincos_pos_embed


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        print('use_dist_on_itp')
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print(f"Environment variable initialization: RANK={os.environ['RANK']} WORLD_SIZE={os.environ['WORLD_SIZE']}, LOCAL_RANK={os.environ['LOCAL_RANK']}")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    #elif 'SLURM_PROCID' in os.environ:
    #    print("'SLURM_PROCID' in os.envision")
    #    env = os.environ
    #    print('env:', env)
    #    num_nodes = env["SLURM_JOB_NUM_NODES"]
    #    gpu_per_node = int(env["SLURM_GPUS_PER_NODE"])
    #    node_rank = int(env["SLURM_PROCID"])
    #    args.rank = gpu_per_node * node_rank + args.local_rank
    #    args.gpu = args.rank % torch.cuda.device_count()
    #    #args.world_size = int(env['SLURM_GPUS'])
    #    args.world_size = int(env['SLURM_NPROCS'])
    #    args.dist_url=f"tcp://{env['MASTER_ADDR']}:{env['MASTER_PORT']}"
    #    print(f"RANK and WORLD_SIZE in environ: {args.rank}/{args.world_size}")
    #    #args.rank = int(os.environ['SLURM_PROCID'])
    #    #args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('distributed init (rank {}): {}, gpu {}, world_size {}'.format(
        args.rank, args.dist_url, args.gpu, args.world_size
    ))
    #torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                     world_size=args.world_size, rank=args.rank)
    #torch.distributed.init_process_group(
    #    backend=args.dist_backend,
    #    init_method=args.dist_url,
    #    world_size=args.world_size,
    #    rank=args.rank,
    #)

    torch.distributed.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
    



def merge_vmae_to_avmae(avmae_state_dict, vmae_ckpt):
    #keys_to_copy=['pos_embed','patch_embed']
    #replaced=0

    vmae_ckpt['cls_token'] = vmae_ckpt['cls_token_v']
    vmae_ckpt['mask_token'] = vmae_ckpt['mask_token_v']

    # pos_emb % not trainable, use default
    pos_embed_v = vmae_ckpt['pos_embed_v'] #1,589,768
    pos_embed = pos_embed_v[:,1:,:] #1,588,768
    cls_embed = pos_embed_v[:,0,:].unsqueeze(1)
    pos_embed = pos_embed.reshape(1, 2, 14, 14,768).sum(dim=1) # 1, 14, 14, 768
    print("Position interpolate from 14,14 to 64,8")
    pos_embed = pos_embed.permute(0, 3, 1, 2) # 1, 14,14,768 -> 1,768,14,14
    pos_embed = torch.nn.functional.interpolate(
        pos_embed, size=(64, 8), mode='bicubic', align_corners=False)
    pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2) # 1, 14, 14, 768 => 1, 196,768
    pos_embed = torch.cat((cls_embed, pos_embed), dim=1)
    assert(vmae_ckpt['pos_embed'].shape == pos_embed.shape)
    vmae_ckpt['pos_embed'] = pos_embed 
    # patch_emb
    # aggregate 3 channels in video-rgb ckpt to 1 channel for audio
    v_weight = vmae_ckpt['patch_embed_v.proj.weight'] # 768,3,2,16,16
    new_proj_weight = torch.nn.Parameter(v_weight.sum(dim=2).sum(dim=1).unsqueeze(1))
    assert(new_proj_weight.shape == vmae_ckpt['patch_embed.proj.weight'].shape)
    vmae_ckpt['patch_embed.proj.weight'] = new_proj_weight
    vmae_ckpt['patch_embed.proj.bias'] = vmae_ckpt['patch_embed_v.proj.bias']

    # hack 
    vmae_ckpt['norm.weight'] = vmae_ckpt['norm_v.weight']
    vmae_ckpt['norm.bias'] = vmae_ckpt['norm_v.bias']

    # replace transformer encoder
    for k,v in vmae_ckpt.items():
        if k.startswith('blocks.'):
            kk = k.replace('blocks.','blocks_v.')
            vmae_ckpt[k] = vmae_ckpt[kk]
        elif k.startswith('blocks_v.'):
            pass
        else:
            print(k)
            pass
    print(k)

def merge_mae_avmae(avmae_ckpt, mae_ckpt):
    avmae_ckpt_model = avmae_ckpt['model']
    mae_ckpt_model = mae_ckpt['model']

    keys_to_change=['cls_token_v']
    replaced=0
    for k,v in avmae_ckpt_model.items():
        if k in keys_to_change:
            kk = k.replace('_v','')
            avmae_ckpt_model[k].shape
            mae_ckpt_model[kk].shape
            assert(avmae_ckpt_model[k].shape == mae_ckpt_model[kk].shape)
            avmae_ckpt_model[k] = mae_ckpt_model[kk]
            replaced+=1
        elif k.startswith('blocks_v'):
            kk = k.replace('blocks_v','blocks')
            assert(avmae_ckpt_model[k].shape == mae_ckpt_model[kk].shape)
            avmae_ckpt_model[k] = mae_ckpt_model[kk]
            replaced+=1

    #hacking patch_emb
    
    org_weight = avmae_ckpt_model['patch_embed_v.proj.weight']
    new_weight = mae_ckpt_model['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,2,1,1)
    #new_weight = mae_ckpt_model['patch_embed.proj.weight'].unsqueeze(2)
    #assert(org_weight.shape == new_weight.shape)
    avmae_ckpt_model['patch_embed_v.proj.weight']=new_weight
    avmae_ckpt_model['patch_embed_v.proj.bias']=mae_ckpt_model['patch_embed.proj.bias']

    #hacking pos_emb
    org_pos = avmae_ckpt_model['pos_embed_v']
    #new_pos_final = get_3d_sincos_pos_embed(org_pos.shape[-1], 14, 4, True) # assume 8 frame, tride = 2
    #new_pos_final = torch.from_numpy(new_pos_final).float().unsqueeze(0)
    
    new_pos=mae_ckpt_model['pos_embed']
    new_cls = new_pos[:,:1,:]
    new_pos_ = new_pos[:,1:,:]
    new_pos_final = torch.cat((new_cls,new_pos_,new_pos_,new_pos_,new_pos_), dim=1)
    assert(org_pos.shape == new_pos_final.shape)
    avmae_ckpt_model['pos_embed_v']=new_pos_final


    print(f'overide avmae ckpt with mae for {replaced}+3 modules')
    avmae_ckpt['model']=avmae_ckpt_model
    return avmae_ckpt



block_v_list=[
'blocks_v.11.norm1.weight','blocks_v.11.norm1.bias','blocks_v.11.attn.qkv.weight','blocks_v.11.attn.qkv.bias','blocks_v.11.attn.proj.weight','blocks_v.11.attn.proj.bias','blocks_v.11.norm2.weight','blocks_v.11.norm2.bias','blocks_v.11.mlp.fc1.weight','blocks_v.11.mlp.fc1.bias','blocks_v.11.mlp.fc2.weight','blocks_v.11.mlp.fc2.bias'
]

block_v_list_large=[
'blocks_v.23.norm1.weight','blocks_v.23.norm1.bias','blocks_v.23.attn.qkv.weight','blocks_v.23.attn.qkv.bias','blocks_v.23.attn.proj.weight','blocks_v.23.attn.proj.bias','blocks_v.23.norm2.weight','blocks_v.23.norm2.bias','blocks_v.23.mlp.fc1.weight','blocks_v.23.mlp.fc1.bias','blocks_v.23.mlp.fc2.weight','blocks_v.23.mlp.fc2.bias'
]

def load_mae(mae_ckpt_path, av=False, large=False):
    mae_ckpt = torch.load(mae_ckpt_path, map_location='cpu')
    mae_ckpt_model = mae_ckpt['model']
    from collections import OrderedDict
    new_ckpt_model = OrderedDict()
    for key, value in mae_ckpt_model.items():
        keys = key.split('.')
        new_key = keys[0] + '_v'

        if len (keys) > 1:
            key_other ='.'.join(keys[1:])
            new_key = new_key + '.' + key_other

        # hacking mae to video mae
        if new_key == 'pos_embed_v':
            new_pos = value
            new_cls = new_pos[:,:1,:]
            new_pos_ = new_pos[:,1:,:]
            value = torch.cat((new_cls,new_pos_,new_pos_,new_pos_,new_pos_), dim=1)
        elif new_key == 'patch_embed_v.proj.weight':
            value = value.unsqueeze(2).repeat(1,1,2,1,1)
        elif new_key == 'decoder_pos_embed_v':
            new_pos = value
            new_cls = new_pos[:,:1,:]
            new_pos_ = new_pos[:,1:,:]
            value = torch.cat((new_cls,new_pos_,new_pos_,new_pos_,new_pos_), dim=1)

        if av:
            if not large: # ViT-B
                if new_key in block_v_list:
                    av_key = new_key.replace('_v.11', '_av.0')
                    new_ckpt_model[av_key] = value
                    av_key = new_key.replace('_v.11', '_av.1')
                    new_ckpt_model[av_key] = value
            else: # ViT-L
                if new_key in block_v_list_large:
                    av_key = new_key.replace('_v.23', '_av.0')
                    new_ckpt_model[av_key] = value
                    av_key = new_key.replace('_v.23', '_av.1')
                    new_ckpt_model[av_key] = value


        print(new_key)
        new_ckpt_model[new_key] = value
    return new_ckpt_model

    # mae_ckpt_model = mae_ckpt['model']


    # ### start here 1025
    # state_dict = model.state_dict()

    # if not args.eval:
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]


    # keys_to_change=['cls_token_v']
    # replaced=0
    # for k,v in avmae_ckpt_model.items():
    #     if k in keys_to_change:
    #         kk = k.replace('_v','')
    #         avmae_ckpt_model[k].shape
    #         mae_ckpt_model[kk].shape
            
    #         avmae_ckpt_model[k] = mae_ckpt_model[kk]
    #         replaced+=1
    #     elif k.startswith('blocks_v'):
    #         kk = k.replace('blocks_v','blocks')
            
    #         avmae_ckpt_model[k] = mae_ckpt_model[kk]
    #         replaced+=1

    # #hacking patch_emb
    
    # org_weight = avmae_ckpt_model['patch_embed_v.proj.weight']
    # new_weight = mae_ckpt_model['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,2,1,1)
    # avmae_ckpt_model['patch_embed_v.proj.weight']=new_weight
    # avmae_ckpt_model['patch_embed_v.proj.bias']=mae_ckpt_model['patch_embed.proj.bias']

    # #hacking pos_emb
    # org_pos = avmae_ckpt_model['pos_embed_v']
    # new_pos=mae_ckpt_model['pos_embed']
    # new_cls = new_pos[:,:1,:]
    # new_pos_ = new_pos[:,1:,:]
    # new_pos_final = torch.cat((new_cls,new_pos_,new_pos_,new_pos_,new_pos_), dim=1)
    # avmae_ckpt_model['pos_embed_v']=new_pos_final


    # print(f'overide avmae ckpt with mae for {replaced}+3 modules')
    # new_ckpt['model']=avmae_ckpt_model
    # return new_ckpt