import numpy as np
import time
import tifffile
import matplotlib.pyplot as plt
from xlearn.utils import angles, nor_data
from xlearn.ganrec2 import GANtomo


def main():
    prj = tifffile.imread('../../test/ganrec/shale_prj.tiff')
    nang, px = prj.shape
    ang = angles(nang)
    prj = nor_data(prj)
    start = time.time()
    gan_tomo_object = GANtomo(prj, ang, iter_num=1200)
    rec = gan_tomo_object.recon
    print('Total time for reconstruction is: {}'.format(time.time()-start))
    tifffile.imwrite('../../test/ganrec/shale_recon.tiff', rec)
if __name__ == "__main__":
    main()
