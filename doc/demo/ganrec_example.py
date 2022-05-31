import numpy as np
import tifffile
import matplotlib.pyplot as plt
from xlearn.utils import angles, nor_data
from xlearn.ganrec2 import GANtomo


def main():
    prj = tifffile.imread('../../test/ganrec/shale_prj.tiff')
    plt.imshow(prj)
    nang, px = prj.shape
    ang = angles(nang)
    prj = nor_data(prj)
    gan_tomo_object = GANtomo(prj, ang, iter_num=1200)
    rec = gan_tomo_object.recon
    tifffile.imwrite('../../test/ganrec/shale_recon.tiff', rec)
if __name__ == "__main__":
    main()
