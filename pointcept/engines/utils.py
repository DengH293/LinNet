import torch
from torch_cluster import grid_cluster
from torch_scatter import segment_min_csr

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def voxel_sample_random(coord, offset, grid_size=0.01, train=True):
    coord =coord-coord.min(0)[0]
    size = torch.tensor([grid_size, grid_size, grid_size]).cuda()
    batch = offset2batch(offset)
    batch1 = batch.unsqueeze(1)*(coord.max(0)[0]-coord.min(0)[0])
    center = coord + batch1

    coord_relative_cluster = torch.remainder(coord, size[0])
    coord_relative_cluster = torch.sum(coord_relative_cluster**2, dim=-1)

    # cluster = torch.ops.torch_cluster.grid(center, size, None, None)
    cluster = grid_cluster(center, size)

    sorted_center, index = torch.sort(cluster)
    unique, order, counts = torch.unique(
        sorted_center, sorted=True, return_inverse=True, return_counts=True
    )

    if train:  # train mode
        idx = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])[:-1] + torch.randint(0, counts.max(), counts.size(), device=coord.device) % counts
        idx = index[idx]
        batch = batch[idx]
        offset = batch2offset(batch)
        return idx.contiguous(), offset.contiguous()
    else:  # val mode
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord, idx = segment_min_csr(coord_relative_cluster[index], idx_ptr)
        idx = index[idx]
        batch = batch[idx]
        offset = batch2offset(batch)
        return idx.contiguous(), offset.contiguous()


def linearization_sampling(coord, offset, grid_size=0.01):
    coord = coord-coord.min(0)[0]
    size = torch.tensor([grid_size, grid_size, grid_size], device = coord.device)
    batch = offset2batch(offset)
    batch1 = batch.unsqueeze(1)*(coord.max(0)[0]-coord.min(0)[0])
    center = coord + batch1

    coord_relative_cluster = torch.remainder(coord, size[0])
    coord_relative_cluster = torch.sum(coord_relative_cluster**2, dim=-1)

    # cluster = torch.ops.torch_cluster.grid(center, size, None, None)
    cluster = grid_cluster(center, size)
    sorted_center, index = torch.sort(cluster)
    unique, order, counts = torch.unique(
        sorted_center, sorted=True, return_inverse=True, return_counts=True
    )

    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    coord, idx = segment_min_csr(coord_relative_cluster[index], idx_ptr)
    idx = index[idx]
    batch = batch[idx]
    offset = batch2offset(batch)
    return idx.contiguous(), offset.contiguous()


def pre_sample(input_dict, grid_sizes):
    new_coord = input_dict['coord']
    new_offset = input_dict['offset']
    indices = []
    offset = [new_offset.cuda(non_blocking=True)]
    for gs in grid_sizes:
        index, new_offset = linearization_sampling(new_coord, offset=new_offset, grid_size=gs)
        new_coord = new_coord[index, :]
        offset.append(new_offset.cuda(non_blocking=True))
        indices.append(index.cuda(non_blocking=True))
    input_dict['offset1'] = offset
    input_dict['indices'] = indices
    return input_dict
