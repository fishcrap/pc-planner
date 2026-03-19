import torch.nn as nn
import torch
import numpy as np

from tqdm import tqdm
from skimage import measure
import trimesh

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dim=3, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : False,
                'input_dims' : input_dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def get_cuda_ifavailable(torch_obj, device=None):
    # if cuda is available return a cuda obeject
    if (torch.cuda.is_available()):
        return torch_obj.cuda(device=device)
    else:
        return torch_obj

def get_3d_grid(resolution=100, bbox=1.2*np.array([[-1, 1], [-1, 1], [-1, 1]]), device=None, eps=0.1, dtype=np.float16):
    # generate points on a uniform grid within  a given range
    # reimplemented from SAL : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    # and IGR : https://github.com/amosgropp/IGR/blob/master/code/utils/plots.py

    shortest_axis = np.argmin(bbox[:, 1] - bbox[:, 0])
    if (shortest_axis == 0):
        x = np.linspace(bbox[0, 0] - eps,  bbox[0, 1] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(bbox[1, 0] - eps,  bbox[1, 1] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(bbox[2, 0] - eps, bbox[2, 1] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(bbox[2, 0] - eps, bbox[2, 1] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(bbox[0, 0] - eps, bbox[0, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(bbox[1, 0] - eps, bbox[1, 1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x.astype(dtype), y.astype(dtype), z.astype(dtype)) #
    # grid_points = get_cuda_ifavailable(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16),
    #                                          device=device)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float16)
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def implicit2mesh(decoder, latent, path, grid_res, device=None, translate=[0., 0., 0.], scale=1,
                  bbox=np.array([[-1, 1], [-1, 1], [-1, 1]])):
    # compute a mesh from the implicit representation in the decoder.
    # Uses marching cubes.
    # reimplemented from SAL get surface trace function : https://github.com/matanatz/SAL/blob/master/code/utils/plots.py
    print('in implicit2mesh')
    print('grid resolution', grid_res)
    print('translate', translate)
    print('scale', scale)
    print('bbox', bbox)

    grid_dict = get_3d_grid(resolution=grid_res, bbox=bbox, device=device)
    print('Finished getting grid_dict')
    cell_width = grid_dict['xyz'][0][2] - grid_dict['xyz'][0][1]
    pnts = grid_dict["grid_points"]

    z = []
    # for point in tqdm(torch.split(pnts, 100000, dim=0)):
    for point in tqdm(torch.split(pnts, 100000, dim=0)):
        # point: (100000, 3)
        if (not latent is None):
            point = torch.cat([point, latent.unsqueeze(0).repeat(point.shape[0], 1), ], dim=1)
        point = get_cuda_ifavailable(point, device=device)
        z.append(decoder(point.type(torch.float32)).detach().cpu().numpy())
    z = np.concatenate(z, axis=0).reshape(grid_dict['xyz'][1].shape[0], grid_dict['xyz'][0].shape[0],
                                          grid_dict['xyz'][2].shape[0]).transpose([1, 0, 2]).astype(np.float64)
    # import pdb; pdb.set_trace()
    print(z.min(), z.max())

    verts, faces, normals, values = measure.marching_cubes(volume=z, level=0.0,
                                                                   spacing=(cell_width, cell_width, cell_width))

    verts = verts + np.array([grid_dict['xyz'][0][0], grid_dict['xyz'][1][0], grid_dict['xyz'][2][0]])
    verts = verts * (1/scale) - translate

    mesh = trimesh.Trimesh(verts, faces, vertex_normals=normals, vertex_colors=values, validate=True)

    mesh.export(path, vertex_normal=True)
    print(f'mesh export to {path}')
