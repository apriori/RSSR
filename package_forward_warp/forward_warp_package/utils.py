import torch
import meshzoo
import numpy as np
import math

def generate_2D_mesh(H, W):
    # try to circumvent meshzoo license limit
    aspect = H/W

    if aspect > 1.0:
       hh = 10
       ww = 10/aspect
    else:
       ww = 10
       hh = 10*aspect

    print(f"H: {hh}")
    print(f"W: {ww}")
    W = int(ww)
    H = int(hh)

    _, faces = meshzoo.rectangle_tri(
       np.linspace(-1.0, 1.0, int(ww)),
       np.linspace(-1.0, 1.0, int(hh)),
       variant='zigzag')

    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 
        
    return grid, faces
