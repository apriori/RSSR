import torch
import meshzoo
import numpy as np
import math

def generate_2D_mesh(H, W):
    def tri(s, a, b, c, lim):
        xi = s + a
        yi = s + b
        zi = s + c
    
        if xi > lim or yi > lim or zi > lim:
            return None
        return np.array([xi, yi, zi])

    h = H
    w = W
    n = (w-1) * (h-1) // 2
    nr = w * h -1 

    faces = []

    f = [[0, 1, w+1], [0, w+1, w], [0, 1, w], [1, w+1, w]]
    starts = [0, 0, 1, 1]

    for si, (a, b, c) in zip(starts, f):
        s = si * w
        j = 0
        
        while j < w-1:
            entry = tri(s, a, b, c, nr)
            
            if entry is not None:
                faces.append(entry)
                k = len(faces)-1
            s = s + 2 * w
            
            if s >= nr or entry is None: 
                j = j + 1
                s = ((j+si) % 2) * w + j  
    faces = np.array(faces)

    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    grid = torch.stack([xx, yy], dim=0)
    return grid, faces
