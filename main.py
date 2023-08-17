## 最终版本的会议论文版本

import argparse
import math
import os
# import cv2 as cv
import cv2 as cv
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import skimage
from matplotlib import pyplot as plt
import hashlib
import time
import pywt
from utils import encryption, dencryption
from Analysis_function import mssim, psnr, npcr, uaci, plt_hist, plt_corr


# if __name__ == '__main__':
#     "仿真结果"
#     #
#     # im = cv.imread("date/woman.tif", 0)
#     # cover = cv.imread("date/peppers_gray.tif", 0)
#
#     # im = cv.imread("date/lena_gray_512.tif", 0)
#     # cover = cv.imread("date/lake.tif", 0)
#
#     # im = cv.imread("date/4.2.01.tiff")
#     # cover = cv.imread("date/4.2.05.tiff")
#
#     # im = cv.imread("date/4.1.03.tiff")
#     # cover = cv.imread("date/mandril_gray.tif", 0)
#
#     im = cv.imread("date/5.1.11.tiff", 0)
#     cover = cv.imread("date/4.1.06.tiff")
#
#     sec_key = [0.55, 0.55, 50, 0.9]
#
#     X3, cip, max_x3, min_x3, T, dynamic_key = encryption(im, cover, sec_key)
#
#     rim = dencryption(cip, cover, max_x3, min_x3, T, dynamic_key, im.shape)
#
#     psnr_cover = psnr(cover, cip)
#     mmsim_cover = mssim(cover, cip, [cover.shape[0]-8,cover.shape[1]-8])
#     print('psnr_cover', psnr_cover,'mmsim_cover',mmsim_cover)
#     psnr_rim = psnr(rim, im)
#     mmsim_rim = mssim(rim, im, [im.shape[0] - 8,im.shape[1] - 8])
#     print('psnr_rim', psnr_rim,'mmsim_rim',mmsim_rim)
#
#     plt_hist(np.uint8(X3),11)
#
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('tcip', np.uint8(X3))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('rim', np.uint8(rim))
#     plt.show()
#     cv.waitKey(0)


# if __name__ == '__main__':
#     "载体对重构图像的影响"
#     im = cv.imread("squat/woman.tif", 0)
#
#     # cover = cv.imread("squat/peppers_gray.tif", 0)
#
#     # cover = cv.imread("squat/lena_gray_512.tif", 0)
#
#     # cover = cv.imread("squat/lake.tif", 0)
#
#     cover = cv.imread("squat/mandril_gray.tif", 0)
#
#     sec_key = [0.55, 0.55, 50, 0.9]
#
#     X3, cip, max_x3, min_x3, T, dynamic_key = encryption(im, cover, sec_key, d=-3)
#
#     rim = dencryption(cip, cover, max_x3, min_x3, T, dynamic_key, im.shape)
#
#     psnr_cover = psnr(cover, cip)
#     mmsim_cover = mssim(cover, cip, [cover.shape[0] - 8,cover.shape[1] - 8])
#     print('psnr_cover', psnr_cover, 'mmsim_cover', mmsim_cover)
#     psnr_rim = psnr(rim, im)
#     mmsim_rim = mssim(rim, im, [im.shape[0] - 8,im.shape[1] - 8])
#     print('psnr_rim', psnr_rim, 'mmsim_rim', mmsim_rim)
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('tcip', np.uint8(X3))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('rim', np.uint8(rim))
#
#     plt.show()
#     cv.waitKey(0)

# if __name__ == '__main__':
#     "明文对密文的影响"
#     # im = cv.imread("squat/woman.tif", 0)
#
#     # im = cv.imread("squat/peppers_gray.tif", 0)
#
#     # im = cv.imread("squat/lena_gray_512.tif", 0)
#
#     im = cv.imread("squat/lake.tif", 0)
#
#     cover = cv.imread("squat/mandril_gray.tif", 0)
#
#     sec_key = [0.55, 0.55, 50, 0.9]
#
#     X3, cip, max_x3, min_x3, T, dynamic_key = encryption(im, cover, sec_key, d=-3)
#
#     rim = dencryption(cip, cover, max_x3, min_x3, T, dynamic_key, im.shape)
#
#     psnr_cover = psnr(cover, cip)
#     mmsim_cover = mssim(cover, cip, [cover.shape[0] - 8,cover.shape[1] - 8])
#     print('psnr_cover', psnr_cover, 'mmsim_cover', mmsim_cover)
#     psnr_rim = psnr(rim, im)
#     mmsim_rim = mssim(rim, im, [im.shape[0] - 8,im.shape[1] - 8])
#     print('psnr_rim', psnr_rim, 'mmsim_rim', mmsim_rim)
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('tcip', np.uint8(X3))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('rim', np.uint8(rim))
#
#     plt.show()
#     cv.waitKey(0)

# if __name__ == '__main__':
#     "直方图移动T对密文质量影响"
#     psnr_cover1 = []
#     mmsim_cover1 = []
#     psnr_cover2 = []
#     mmsim_cover2 = []
#     psnr_cover3 = []
#     mmsim_cover3 = []
#     psnr_cover4 = []
#     mmsim_cover4 = []
#     mean_psnr = []
#     mean_mssim = []
#     record_T = []
#     for T in range(-10,10,1):
#         print(T)
#
#         record_T.append(T)
#
#         im1 = cv.imread("squat/woman.tif", 0)
#
#         im2 = cv.imread("squat/peppers_gray.tif", 0)
#
#         im3 = cv.imread("squat/lena_gray_512.tif", 0)
#
#         im4 = cv.imread("squat/lake.tif", 0)
#
#         cover = cv.imread("squat/mandril_gray.tif", 0)
#
#         sec_key = [0.55, 0.55, 50, 0.9]
#
#         X31, cip1, max_x31, min_x31, T1, dynamic_key1 = encryption(im1, cover, sec_key, d=T)
#         p1 = psnr(cover, cip1)
#         m1 = mssim(cover, cip1, [cover.shape[0] - 8,cover.shape[1] - 8])
#         psnr_cover1.append(p1)
#         mmsim_cover1.append(m1)
#
#         X32, cip2, max_x32, min_x32, T2, dynamic_key2 = encryption(im2, cover, sec_key, d=T)
#         p2 = psnr(cover, cip2)
#         m2 = mssim(cover, cip2, [cover.shape[0] - 8,cover.shape[1] - 8])
#         psnr_cover2.append(p2)
#         mmsim_cover2.append(m2)
#
#         X33, cip3, max_x33, min_x33, T3, dynamic_key3 = encryption(im3, cover, sec_key, d=T)
#         p3 = psnr(cover, cip3)
#         m3 = mssim(cover, cip3, [cover.shape[0] - 8,cover.shape[1] - 8])
#         psnr_cover3.append(p3)
#         mmsim_cover3.append(m3)
#
#         X34, cip4, max_x34, min_x34, T4, dynamic_key4 = encryption(im4, cover, sec_key, d=T)
#         p4 = psnr(cover, cip4)
#         m4 = mssim(cover, cip4, [cover.shape[0] - 8,cover.shape[1] - 8])
#         psnr_cover4.append(p4)
#         mmsim_cover4.append(m4)
#
#         mean_psnr.append((p1+p2+p3+p4)/4)
#         mean_mssim.append((m1+m2+m3+m4)/4)
#
#
#     plt.figure(1)
#     plt.plot(record_T, psnr_cover1, 'ro-.')
#     plt.plot(record_T, psnr_cover2, 'cv-.')
#     plt.plot(record_T, psnr_cover3, 'm^-.')
#     plt.plot(record_T, psnr_cover4, 'y1-.')
#     plt.plot(record_T, mean_psnr, 'k*-')
#     plt.legend(('woman','peppers','lena','lake','mean'),loc='best')
#     plt.ylabel('PSNR(dB)')
#     plt.xlabel('M')
#
#
#     plt.figure(2)
#     plt.plot(record_T, mmsim_cover1, 'ro-.')
#     plt.plot(record_T, mmsim_cover2, 'cv-.')
#     plt.plot(record_T, mmsim_cover3, 'm^-.')
#     plt.plot(record_T, mmsim_cover4, 'y1-.')
#     plt.plot(record_T, mean_mssim, 'k*-')
#     plt.legend(('woman','peppers','lena','lake','mean'),loc='best')
#     plt.ylabel('MSSIM')
#     plt.xlabel('M')
#
#     plt.show()
#     cv.waitKey(0)


# if __name__ == '__main__':
#     "密钥敏感性分析"
#
#     im = cv.imread("squat/woman.tif", 0)
#     cover = cv.imread("squat/peppers_gray.tif", 0)
#
#     sec_key = [0.55, 0.55, 50, 0.9]
#
#     X3, cip, max_x3, min_x3, T, dynamic_key = encryption(im, cover, sec_key, d=-3)
#
#     j1_dynamic_key = np.copy(dynamic_key)
#     j1_dynamic_key[0] = j1_dynamic_key[0]+10**(-14)
#     rim1 = dencryption(cip, cover, max_x3, min_x3, T,  j1_dynamic_key, im.shape)
#
#     j2_dynamic_key = np.copy(dynamic_key)
#     j2_dynamic_key[1] = j2_dynamic_key[1] + 10 ** (-14)
#     rim2 = dencryption(cip, cover, max_x3, min_x3, T, j2_dynamic_key, im.shape)
#
#     j3_dynamic_key = np.copy(dynamic_key)
#     j3_dynamic_key[2] = j1_dynamic_key[2] + 10 ** (-14)
#     rim3 = dencryption(cip, cover, max_x3, min_x3, T, j3_dynamic_key, im.shape)
#
#     j4_dynamic_key = np.copy(dynamic_key)
#     j4_dynamic_key[3] = j4_dynamic_key[3] + 10 ** (-14)
#     rim4 = dencryption(cip, cover, max_x3, min_x3, T, j4_dynamic_key, im.shape)
#
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('tcip', np.uint8(X3))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('rim1', np.uint8(rim1))
#     cv.imshow('rim2', np.uint8(rim2))
#     cv.imshow('rim3', np.uint8(rim3))
#     cv.imshow('rim4', np.uint8(rim4))
#     plt.show()
#     cv.waitKey(0)


# if __name__ == '__main__':
#     "明文敏感性"
#
#     # im = cv.imread("squat/woman.tif", 0)
#     # im1 = np.copy(im)
#     # im1[1, 1] = im1[1, 1] + 1
#     # im1[411, 123] = im1[411, 123] - 1
#
#     # im = cv.imread("squat/peppers_gray.tif", 0)
#     # im1 = np.copy(im)
#     # im1[156, 255] = im[332,18]
#     # im1[332, 18] = im[156, 255]
#
#     # im = cv.imread("squat/lena_gray_512.tif", 0)
#     # im1 = np.copy(im)
#     # im1[78, 464] = im1[78, 464] + 1
#     # im1[496, 354] = im1[496, 354] - 1
#
#     im = cv.imread("squat/lake.tif", 0)
#     im1 = np.copy(im)
#     im1[22, 167] = im[501,23]
#     im1[501,23] = im[22, 167]
#
#
#     cover = cv.imread("squat/mandril_gray.tif", 0)
#
#     sec_key = [0.55, 0.55, 50, 0.9]
#     X3, cip, max_x3, min_x3, T, dynamic_key = encryption(im, cover, sec_key, d=-3)
#     X31, cip1, max_x31, min_x31, T1, dynamic_key1 = encryption(im1, cover, sec_key, d=-3)
#
#     np_tc = npcr(X3, X31)
#     np_cip = npcr(cip, cip1)
#
#     mssim_tc = mssim(X3, X31, [X3.shape[0] - 8,X3.shape[1] - 8])
#     mssim_cip = mssim(cip, cip1, [cip.shape[0] - 8,cip.shape[1] - 8])
#
#     print('np_tc', np_tc,'mssim_tc',mssim_tc)
#     print('np_cip', np_cip,'mssim_cip',mssim_cip)


# if __name__ == '__main__':
#     "对比实验自己部分"
#
#     im = cv.imread("date/lena_gray_512.tif", 0)
#     cover = cv.imread("date/mandril_gray.tif", 0)
#
#     # im = cv.imread("date/peppers_gray.tif", 0)
#     # cover = cv.imread("date/woman.tif", 0)
#
#     # im = cv.imread("date/house.tif", 0)
#     # cover = cv.imread("date/lake.tif", 0)
#
#     # im = cv.imread("date/boat.tiff",0)
#     # cover = cv.imread("date/barbara.pgm",0)
#
#     sec_key = [0.55, 0.55, 50, 0.9]
#     X3, cip, max_x3, min_x3, T, dynamic_key = encryption(im, cover, sec_key)
#     rim = dencryption(cip, cover, max_x3, min_x3, T, dynamic_key, im.shape)
#
#     psnr_cover = psnr(cover, cip)
#     mmsim_cover = mssim(cover, cip, [cover.shape[0]-8,cover.shape[1]-8])
#     print('psnr_cover', psnr_cover,'mmsim_cover',mmsim_cover)
#     psnr_rim = psnr(rim, im)
#     mmsim_rim = mssim(rim, im, [im.shape[0] - 8,im.shape[1] - 8])
#     print('psnr_rim', psnr_rim,'mmsim_rim',mmsim_rim)
#
#     cv.imshow('im', np.uint8(im))
#     cv.imshow('cover', np.uint8(cover))
#     cv.imshow('tcip', np.uint8(X3))
#     cv.imshow('cip', np.uint8(cip))
#     cv.imshow('rim', np.uint8(rim))
#     plt.show()
#     cv.waitKey(0)


# if __name__ == '__main__':
# #     "对比实验自己部分"
# #
#     im = cv.imread("c.png", 0)
#     ar,num=np.unique(im,return_counts=True)
#     print(ar,num)
#
#     im[im<128]=0
#     im[im>=128]=255
#     cv.imwrite('b.bmp', np.uint8(im))
#     cv.imshow('im', np.uint8(im))
#     cv.waitKey(0)
