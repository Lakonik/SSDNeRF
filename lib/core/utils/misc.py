import functools
import torch
import torch.distributed as dist

from mmcv.parallel import MMDistributedDataParallel
from collections import defaultdict, abc as container_abcs
from itertools import chain
from functools import partial
from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def optimizer_state_to(state_dict, device=None, dtype=None):
    assert dtype.is_floating_point
    out = dict(state=dict(),
               param_groups=state_dict['param_groups'])
    for key_state_single, state_single in state_dict['state'].items():
        state_single_out = dict()
        for key, val in state_single.items():
            if isinstance(val, torch.Tensor):
                if key != 'step' and val.dtype != dtype:
                    val = val.clamp(min=torch.finfo(dtype).min, max=torch.finfo(dtype).max)
                state_single_out[key] = val.to(
                    device=device, dtype=None if key == 'step' else dtype)
            else:
                state_single_out[key] = val
        out['state'][key_state_single] = state_single_out
    return out


def load_tensor_to_dict(d, key, value, device=None, dtype=None):
    assert dtype.is_floating_point
    if isinstance(value, torch.Tensor):
        if key not in ['density_grid', 'density_bitfield', 'step'] and value.dtype != dtype:
            value = value.clamp(min=torch.finfo(dtype).min, max=torch.finfo(dtype).max)
        if key in d:
            d[key].copy_(value)
        else:
            d[key] = value.to(
                device=device, dtype=None if key in ['density_grid', 'density_bitfield', 'step'] else dtype)
    else:
        d[key] = value


def optimizer_state_copy(d_src, d_dst, device=None, dtype=None):
    d_dst['param_groups'] = d_src['param_groups']
    for key_state_single, state_single in d_src['state'].items():
        if key_state_single not in d_dst['state']:
            d_dst['state'][key_state_single] = dict()
        for key, val in state_single.items():
            load_tensor_to_dict(d_dst['state'][key_state_single], key, val,
                                device=device, dtype=dtype)


def optimizer_set_state(optimizer, state_dict):
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
              zip(chain.from_iterable((g['params'] for g in saved_groups)),
                  chain.from_iterable((g['params'] for g in groups)))}

    def cast(param, value, key=None):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            if key != "step":
                if param.is_floating_point():
                    value = value.to(param.dtype)
                value = value.to(param.device)
            return value
        elif isinstance(value, dict):
            return {k: cast(param, v, key=k) for k, v in value.items()}
        elif isinstance(value, container_abcs.Iterable):
            return type(value)(cast(param, v) for v in value)
        else:
            return value

    state = defaultdict(dict)
    for k, v in state_dict['state'].items():
        if k in id_map:
            param = id_map[k]
            state[param] = cast(param, v)
        else:
            state[k] = v

    optimizer.__setstate__({'state': state})


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if isinstance(obj, MMDistributedDataParallel):
            obj = obj.module
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if isinstance(pre, MMDistributedDataParallel):
        pre = pre.module
    return setattr(pre, post, val)


def rhasattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if isinstance(pre, MMDistributedDataParallel):
        pre = pre.module
    return hasattr(pre, post)


def rdelattr(obj, attr):
    pre, _, post = attr.rpartition('.')
    pre = rgetattr(obj, pre) if pre else obj
    if isinstance(pre, MMDistributedDataParallel):
        pre = pre.module
    return delattr(pre, post)


class module_requires_grad:
    def __init__(self, module, requires_grad=True):
        self.module = module
        self.requires_grad = requires_grad
        self.prev = []

    def __enter__(self):
        for p in self.module.parameters():
            self.prev.append(p.requires_grad)
            p.requires_grad = self.requires_grad

    def __exit__(self, exc_type, exc_value, traceback):
        for p, r in zip(self.module.parameters(), self.prev):
            p.requires_grad = r
