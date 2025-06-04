from re import I
from typing import List, Type, Tuple

import logging
import torch
import torch.nn as nn
from pointcept.models.builder import MODELS

from torch.autograd import Function
from pointops._C import knn_query_cuda, random_ball_query_cuda, ball_query_cuda
from linops_cuda import linear_voxel_query_cuda, insert_hashtable_cuda
from torch_scatter import segment_min_csr

def xyz_height(xyz, offset):
    z = xyz[:, 2]
    idx_ptr = torch.cat([offset.new_zeros(1), offset]).long()
    height, idx = segment_min_csr(z, idx_ptr)
    height = height.repeat_interleave(offset2bincount(offset))
    # xyz = torch.cat([xyz, z.unsqueeze(-1)-height.unsqueeze(-1)], dim=-1)
    height = z.unsqueeze(-1)-height.unsqueeze(-1)
    return height 

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


def xyz2key(xyz, depth=8, radius=0.08):
    xyz = torch.div(xyz, radius, rounding_mode='floor').int()
    # B, N, _ = xyz.shape
    x, y, z, = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    key = torch.zeros_like(x)
    for i in range(depth):
        mask = 1 << i
        key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
        )

    return key.view(-1)

    
def counts2pos(counts):
    value = torch.zeros_like(counts, dtype=torch.int64)
    start = torch.cumsum(counts, dim=0)
    start = torch.cat([start.new_zeros(1), start[:-1]])
    value |= (start & ((1 << 32) - 1)) << 0
    value |= (counts & ((1 << 32) - 1)) << 32
    return value


def next_power_of_2(n: int) -> int:
    if n < 1:
        return 1
    n_tensor = torch.tensor([n], dtype=torch.float32)
    log2_value = torch.ceil(torch.log2(n_tensor))
    result = torch.pow(2, log2_value)
    return int(result.item())


class HashQuery(Function):
    """Hash Query.

    Find nearby points in spherical space.
    """

    @staticmethod
    def forward(ctx, nsample, xyz, offset, new_xyz, new_offset, grid_size=0.08):
        """
        input: coords: (b, n, 3), new_xyz: (b, n2, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        assert xyz.is_contiguous()
        xyz = xyz - torch.min(xyz, dim=0)[0].unsqueeze(0)
        new_xyz = new_xyz - torch.min(new_xyz, dim=0)[0].unsqueeze(0)
        grid_id = xyz2grid_id(xyz, grid_size, offset)
        new_grid_id = xyz2grid_id(new_xyz, grid_size, new_offset)

        key, order, counts = torch.unique(
            grid_id, sorted=True, return_inverse=True, return_counts=True
        )
        index = torch.argsort(grid_id)
        sorted_xyz = xyz.view(-1, 3)[index, :]

        value = counts2pos(counts)
        hash_size = next_power_of_2(len(key)*2)

        hash_table = torch.cuda.LongTensor(hash_size*2).zero_()
        insert_hashtable_cuda(len(key), hash_size, hash_table, key, value)

        new_xyz = new_xyz.view(-1, 3)

        n = new_xyz.size(0)
        idx = torch.cuda.IntTensor(n, nsample).zero_()
        dist = torch.cuda.FloatTensor(n, nsample).zero_()

        linear_voxel_query_cuda(
            n, nsample, hash_size, grid_size,
            hash_table,
            sorted_xyz, new_xyz, new_grid_id,
            new_offset.int(),
            idx, dist
        )

        idx = index[idx.long()]
        del hash_table
        return idx, dist

class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, offset, new_xyz=None, new_offset=None):
        """
        input: coords: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample) -1 is placeholder, dist2: (m, nsample)
        """
        if new_xyz is None or new_offset is None:
            new_xyz = xyz
            new_offset = offset
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        knn_query_cuda(
            m, nsample, xyz, new_xyz, offset.int(), new_offset.int(), idx, dist2
        )
        return idx, torch.sqrt(dist2)



knn_query = KNNQuery.apply
hash_query = HashQuery.apply


class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError


def create_linear_block1d(in_channels: int,
                          out_channels: int,
                          bn: bool = True,
                          act: bool = True
                          ) -> nn.Sequential:
    """
    Linear -> [BatchNorm] -> [ReLU]
    """
    layers: List[nn.Module] = [nn.Linear(in_channels, out_channels)]
    if bn:
        layers.append(PointBatchNorm(out_channels))
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def xyz2grid_id(xyz: torch.Tensor, grid_size: float, offset: torch.Tensor) -> torch.Tensor:
    """
    Compute a 64-bit voxel grid ID for each point.
    Layout: [ batch:8 bits | z:18 bits | y:18 bits | x:18 bits ]
    """
    batch = offset2batch(offset)
    coords = torch.floor_divide(xyz, grid_size).to(torch.long)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    key = (x & ((1 << 18) - 1)) \
        | ((y & ((1 << 18) - 1)) << 18) \
        | ((z & ((1 << 18) - 1)) << 36) \
        | ((batch & ((1 << 8) - 1)) << 54)
    return key


class DSA(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expanse=1,
                 ):
        super().__init__()
        self.pre = create_linear_block1d(in_channels, out_channels)
        self.pool = lambda x: torch.max(x, dim=-2, keepdim=False)[0]
        self.bn = nn.BatchNorm1d(out_channels)

        # feed-forward network
        c = [out_channels, out_channels * expanse, out_channels]
        layers = [
            create_linear_block1d(c[i], c[i+1], act=(i < len(c)-2))
            for i in range(len(c)-1)
        ]
        self.ffn = nn.Sequential(*layers)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):
        p, f, pe, knn_index = inputs
        identity = f
        f = self.pre(f)
        f = self.bn(self.pool(pe + f[knn_index]))

        f = self.ffn(f)

        f = identity + f
        f = self.act(f)
        return [p, f, pe, knn_index]


class DownSample(nn.Module):
    """
    Modified Set Abstraction (PointNet++) with DSA layers.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 nsample: int,
                 radius: float
                 ):
        super().__init__()
        self.nsample = nsample
        self.radius = radius

        self.conv1 = create_linear_block1d(in_channels, out_channels)
        # for positional encoding on deltas
        self.conv2 = nn.Sequential(
            create_linear_block1d(3, 32),
            create_linear_block1d(32, 32),
            create_linear_block1d(32, out_channels, act=False)
        )
        self.pool = lambda x: x.max(dim=-2)[0]
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, inputs):
        p, f, index, offset = inputs
        new_p = p[index, :]
        f = self.conv1(f)
        
        knn_idx, _ = knn_query(self.nsample, p, offset[0].contiguous(), new_p, offset[1].contiguous())
        # knn_idx, _ = hash_query(self.nsample, p, offset[0].contiguous(), new_p, offset[1].contiguous(), self.radius*8)
        knn_idx = knn_idx.long()
        
        dp = p[knn_idx] - new_p.unsqueeze(1)
        pe = self.conv2(dp)
        
        f = pe + f[knn_idx]
        f = self.bn(self.pool(f))
        return new_p, f


class Stage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 depth: int,
                 nsample: int,
                 radius: List[float]
                 ):
        super().__init__()
        assert len(radius) == 2
        self.radius = radius
        self.down_sample = DownSample(in_channels, out_channels, nsample, radius[0])
        self.blocks = nn.Sequential(
            *(DSA(out_channels, out_channels) for _ in range(depth - 1))
        )
        # PE encoder
        self.pe = nn.Sequential(
            create_linear_block1d(3, 32),
            create_linear_block1d(32, 32),
            create_linear_block1d(32, out_channels, act=False)
        )


    def forward(self, inputs):
        p, f, index, offset = inputs

        new_p, new_f = self.down_sample([p, f, index, offset])  # SetAbstraction

        # knn_idx, _ = knn_query(self.down_sample.nsample, new_p, offset[1].contiguous(), new_p, offset[1].contiguous())
        knn_idx, _ = hash_query(self.down_sample.nsample, new_p, offset[1].contiguous(), new_p, offset[1].contiguous(), self.radius[1]*4)
        knn_idx = knn_idx.long().contiguous()
        dp = new_p[knn_idx] - new_p.unsqueeze(1)
        pe = self.pe(dp)

        for block in self.blocks:
            new_p, new_f, pe, knn_idx = block([new_p, new_f, pe, knn_idx])
        return new_p, new_f


class FeaturePropogation(nn.Module):
    def __init__(self, mlp, nsample
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        self.f2conv = create_linear_block1d(mlp[1], mlp[0])
        self.pool = lambda x: torch.max(x, dim=-2, keepdim=False)[0]
        self.norm = PointBatchNorm(mlp[0])
        self.nsample = nsample

    def forward(self, pfo1, pfo2=None):
        p1, f1, o1 = pfo1
        p2, f2, o2 = pfo2

        idx, dist = knn_query(self.nsample, p2, o2.contiguous(), p1, o1.contiguous())  #TODO：交换MLP顺序
        f1 = f1 + self.pool(self.f2conv(f2[idx.long(), :]))
        f1 = self.norm(f1)

        return f1
    
    

@MODELS.register_module()
class LinNet(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 num_classes: int = 19,
                 nsample: int = 24,
                 grid_sizes: List[float] = [0.15, 0.375, 0.9375, 2.34375],
                 channels: List[int] = [64, 128, 256, 512, 1024],
                 encoder_depth: List[int] = [4, 4, 7, 4],
                 radius: List[float] = [0.05, 0.15, 0.375, 0.9375, 2.34375],
                 decoder_nsample: int = 1
                 ):
        super().__init__()
        self.emb = create_linear_block1d(in_channels, channels[0])
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(len(encoder_depth)):
            enc = Stage(channels[i], channels[i + 1], depth=encoder_depth[i], nsample=nsample, radius=radius[i: i+2])
            self.enc_stages.append(enc)
            dec = FeaturePropogation(channels[i: i+2], nsample=decoder_nsample)
            self.dec_stages.append(dec)
        self.head = nn.Sequential(
            nn.BatchNorm1d(channels[0], momentum=0.02),
            nn.ReLU(),
            nn.Linear(channels[0], num_classes)
        )

    def forward_seg_feat(self, p0, f0=None):
        feats = self.forward(p0, f0)
        return p0, feats

    def forward(self, data_dict):

        p0 = data_dict["coord"]
        f0 = data_dict["feat"]
        offset = data_dict["offset1"]
        indices = data_dict["indices"]
        p, f = [p0], [f0]
        # height = xyz_height(f0, offset[0])
        # # f0 = torch.cat([f0, height], dim=-1)
        _f = self.emb(f0)
        _p = p0
        f.append(_f)
        p.append(_p)
        for i in range(len(self.enc_stages)):
            _p, _f = self.enc_stages[i]([_p, _f, indices[i], offset[i:i + 2]])
            p.append(_p)
            f.append(_f)

        for i in range(-1, -len(self.dec_stages) - 1, -1):
            f[i - 1] = self.dec_stages[i]([p[i - 1], f[i - 1], offset[i-1]], [p[i], f[i], offset[i]])

        f = self.head(f[1])

        return f
