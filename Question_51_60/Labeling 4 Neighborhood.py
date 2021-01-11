# -*- coding: utf-8 -*-
import cv2
import numpy as np


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126*r + 0.7152*g + 0.0722*b
    out = out.astype(np.uint8)

    return out

def L4N(img):    
    tag = 0
    grid = img.copy()

    for j in range(H):
        for i in range(W):
            if grid[j, i] == 255:
                tag += 1
                dfs(grid, j, i, tag)
                
    # draw color
    COLORS = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]]
    out = np.zeros((H, W, C), dtype=np.uint8)

    for i in range(tag):
        out[grid == (i+1)] = COLORS[i]            
    
    return out

def dfs(grid, j, i, tag):
    grid[j, i] = tag
    for y, x in [[0, 1], [1, 0], [0, -1]]:
        tmp_j = j + y
        tmp_i = i + x
        if 0 <= tmp_j < H and 0 <= tmp_i < W and grid[tmp_j, tmp_i] == 255:
            dfs(grid, tmp_j, tmp_i, tag)

# Read image
img = cv2.imread("seg.png").astype(np.float32)
H, W, C = img.shape

# Gray scale
gray = BGR2GRAY(img)

out = L4N(gray)

# Show and save image
cv2.imshow("result", out*10)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Mysult/out59.jpg", out)
