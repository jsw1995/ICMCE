"""
一些图像加密所用到的函数
输入图像的彩色图像必须为长宽高的形式
"""
import argparse
import math
import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from matplotlib import pyplot as plt


def npcr(a, b):
    # 像素变化率
    if a.shape == b.shape:

        if len(a.shape) == 3:
            [rows, columns, pages] = a.shape
            counter = 0
            for k in range(pages):
                for j in range(columns):
                    for i in range(rows):
                        if a[i, j, k] != b[i, j, k]:
                            counter += 1
            c = counter/(rows*columns*pages)
        else:
            [rows, columns] = a.shape
            counter = 0
            for j in range(columns):
                for i in range(rows):
                    if a[i, j] != b[i, j]:
                        counter += 1
            c = counter / (rows * columns)

    else:
        print('两图像必须大小相同')
        c = 0.0

    return c

def uaci(image1, image2):
    # 归一化变化强度
    if image1.shape == image2.shape:
        if len(image1.shape) == 3:
            [rows, columns, pages] = image1.shape
            counter = 0
            for k in range(pages):
                for j in range(columns):
                    for i in range(rows):
                        counter = (math.fabs(int(image1[i, j, k]) - int(image2[i, j, k])))/255 + counter
            c = counter / (rows * columns * pages)
        else:
            [rows, columns] = image1.shape
            counter = 0
            for j in range(columns):
                for i in range(rows):
                    counter = (math.fabs(int(image1[i, j]) - int(image2[i, j])))/255 + counter
            c = counter / (rows * columns)
    else:
        print('两图像必须大小相同')
        c = 0.0

    return c

def calc_ent(x):
    """
        calculate shanno ent of x
    """
    x = x.reshape(-1, )
    x_value_list = set([x[i] for i in range(x.shape[0])])
    # for i in range(x.shape[0])：
    #    x_value_list = set(x[i])

    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]   # x[x == x_value].shape[0] 求取x中等于x_value的个数
        logp = np.log2(p)
        ent -= p * logp

    return ent

def information_entropy(image):

    if len(image)==3:
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]
        B_ENT = calc_ent(B)
        G_ENT = calc_ent(G)
        R_ENT = calc_ent(R)
        T1 = np.array([B_ENT, G_ENT, R_ENT])
    else:
        T1 = calc_ent(image)
    return T1


def mssim(im1,im2,win):
    """
    :param im1: 第一张图像(输入为长宽高形式)
    :param im2: 第二张图像
    :param win: 窗口大小
    :return: 平均结构相似性
    """
    im1 = im1.astype(float)
    im2 = im2.astype(float)
    c1 = 2.55**2
    c2 = 7.65**2
    if len(im1.shape)==3:
        m,n,p=im1.shape
    else:
        im1 = np.expand_dims(im1,axis=2)
        im2 = np.expand_dims(im2,axis=2)
        m, n, p = im1.shape

    sz1 = int(math.floor(win[0] / 2))
    sz2 = int(math.floor(win[1] / 2))

    mer = []

    for i in range(sz1,m-sz1):
        for j in range(sz2,n-sz2):
            six = im1[i-sz1:i+sz1,j-sz2:j+sz2,:].ravel()
            siy = im2[i-sz1:i+sz1,j-sz2:j+sz2,:].ravel()
            meux = np.mean(six)
            meuy = np.mean(siy)
            sigx = np.std(six)
            sigy = np.std(siy)
            sigxy = np.mean((six-meux)*(siy-meuy))
            er = ((2*meux*meuy+c1)*(2*sigxy+c2))/((meux**2+meuy**2+c1)*(sigx**2+sigy**2+c2))
            mer.append(er)

    return sum(mer)/(len(mer))


def psnr(a,b):
    """
    两图的峰值信噪比
    """
    a = a.astype(float).reshape(-1, )
    b = b.astype(float).reshape(-1, )
    mse = np.sum((a-b)**2)
    psnr = 10*math.log10(255**2*a.shape[0]/mse)
    return psnr


def plt_hist(img,a):
    """
    画出直方图
    """
    if len(img.shape) == 3:
        plt.figure(a)
        plt.hist(img[:, :, 0].ravel(), 256, [0, 256], color='blue')
        plt.hist(img[:, :, 1].ravel(), 256, [0, 256], color='green')
        plt.hist(img[:, :, 2].ravel(), 256, [0, 256], color='red')
    else:
        plt.figure(a)
        plt.hist(img.ravel(), 256, [0, 256])


def plt_corr(img,a):
    """
    画相关性
    """
    m = img.shape[0]
    n = img.shape[1]
    tem = np.ones((img.shape[0], img.shape[1]))
    if len(img.shape) == 3:
        fig = plt.figure(a)
        ax = fig.gca(projection='3d')
        ax.plot(tem[:, 0:n-1].ravel(), img[:, 0:n-1, 0].ravel(), img[:, 1:n, 0].ravel(), '.',
                           markersize=0.25, color='blue')
        ax.plot(2 * tem[0:m-1, :].ravel(), img[0:m-1, :, 0].ravel(), img[1:m, :, 0].ravel(), '.',
                           markersize=0.25, color='blue')
        ax.plot(3 * tem[0:m-1, 0:n-1].ravel(), img[0:m-1, 0:n-1, 0].ravel(),
                           img[1:m, 1:n, 0].ravel(), '.',
                           markersize=0.25, color='blue')

        ax.plot(5 * tem[:, 0:n-1].ravel(), img[:, 0:n-1, 1].ravel(), img[:, 1:n, 1].ravel(), '.',
                           markersize=0.25, color='green')
        ax.plot(6 * tem[0:m-1, :].ravel(), img[0:m-1, :, 1].ravel(), img[1:m, :, 1].ravel(), '.',
                           markersize=0.25, color='green')
        ax.plot(7 * tem[0:m-1, 0:n-1].ravel(), img[0:m-1, 0:n-1, 1].ravel(),
                           img[1:m, 1:n, 1].ravel(), '.',
                           markersize=0.25, color='green')

        ax.plot(9 * tem[:, 0:n-1].ravel(), img[:, 0:n-1, 2].ravel(), img[:, 1:n, 2].ravel(), '.',
                           markersize=0.25, color='red')
        ax.plot(10 * tem[0:m-1, :].ravel(), img[0:m-1, :, 2].ravel(), img[1:m, :, 2].ravel(), '.',
                           markersize=0.25, color='red')
        ax.plot(11 * tem[0:m-1, 0:n-1].ravel(), img[0:m-1, 0:n-1, 2].ravel(),
                           img[1:m, 1:n, 2].ravel(), '.',
                           markersize=0.25, color='red')

        plt.xticks([1, 2, 3, 5, 6, 7, 9, 10 ,11],
                   ['Horizontal', 'Vertical', 'Diagonal', 'Horizontal', 'Vertical', 'Diagonal', 'Horizontal', 'Vertical', 'Diagonal'],
                   rotation=20)

        fig = plt.figure(a+a)
        ax = fig.gca(projection='3d')
        ax.plot(tem.ravel(), img[:, :, 0].ravel(), img[:, :, 1].ravel(), '.',
                markersize=0.25, color='yellow')
        ax.plot(2 * tem.ravel(), img[:, :, 0].ravel(), img[:, :, 2].ravel(), '.',
                markersize=0.25, color='magenta')
        ax.plot(3 * tem.ravel(), img[:, :, 1].ravel(),img[:, :, 2].ravel(), '.',
                markersize=0.25, color='cyan')
        plt.xticks([1, 2, 3],
                   ['R_G', 'R_B', 'B_G'],rotation=20)

    else:
        fig = plt.figure(a)
        ax = fig.gca(projection='3d')
        ax.plot(tem[:, 0:n - 1].ravel(), img[:, 0:n - 1].ravel(), img[:, 1:n].ravel(), '.',
                markersize=0.25, color='blue')
        ax.plot(2 * tem[0:m - 1, :].ravel(), img[0:m - 1, :].ravel(), img[1:m, :].ravel(), '.',
                markersize=0.25, color='blue')
        ax.plot(3 * tem[0:m - 1, 0:n - 1].ravel(), img[0:m - 1, 0:n - 1].ravel(),
                img[1:m, 1:n].ravel(), '.',
                markersize=0.25, color='blue')
        plt.xticks([1, 2, 3],
                   ['Horizontal', 'Vertical', 'Diagonal'], rotation=20)