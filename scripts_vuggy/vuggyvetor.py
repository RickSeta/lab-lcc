import sys
import os
import numpy as np
from matplotlib import pyplot as plt


poreSizes =  [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]

rock = "EY"

for ndiv in poreSizes:
# ndiv = int(sys.argv[1])
    nx = 600
    ny = 600
    zstart = 0000
    nz = 2 * ndiv + 2
    cx = nx/2
    cy = nx/2
    cz = ndiv + 1
    cr = ndiv
    digits = len(str(ndiv)) + 1
    prefix = ''


    x = np.arange(nx) + 0.5
    y = np.arange(ny) + 0.5
    z = np.arange(nz) + 0.5
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    baseMatrix = ((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) >= cr ** 2
    booleanMatrix = np.zeros_like(baseMatrix, dtype=bool)
    booleanMatrix[0, :, :] = False
    booleanMatrix[-1, :, :] = False
    booleanMatrix[1:-1, :, :] = baseMatrix[1:-1, :, :]

    folderName = f"{rock}_vuggy_r={ndiv}_size={nx}X{ny}_start={zstart}"

    os.mkdir(folderName)

    for num in range(nz):
        if len(str(zstart + num)) < digits:
            zeros = digits - len(str(num))
            prefix = ('0' * zeros)
        else:
            prefix = ''
            
        filename = f"{folderName}/rev_{rock}_BIN_{prefix + str(zstart + num)}.png"
        plt.imsave(filename, booleanMatrix[:, :, num], cmap='Greys_r', format='png', origin='lower')

    plt.axis('off')



