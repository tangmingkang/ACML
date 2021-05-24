import cv2
import numpy as np
from glob import glob

def shade_of_gray_cc(img, power=6, gamma=None):
    img_dtype = img.dtype
    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256, 1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    # print(img_power.shape, img_power[0][0][0])

    rgb_vec = np.power(np.mean(img_power, (0, 1)), 1 / power)
    # print(rgb_vec)

    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))

    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)

    return img.astype(img_dtype)

def max_rgb(img):
    img_dtype = img.dtype
    img = img.astype('float32')
    # print(img.shape)
    r_max = np.max(img[:, :, 0])
    g_max = np.max(img[:, :, 1])
    b_max = np.max(img[:, :, 2])
    max=np.max(img)

    # print(r_max,g_max,b_max,max)

    k=(r_max,g_max,b_max)/max
    res=img

    res[:, :, 0] = img[:, :, 0] / k[0]
    res[:, :, 1] = img[:, :, 1] / k[1]
    res[:, :, 2] = img[:, :, 2] / k[2]

    result = np.clip(res, a_min=0, a_max=255)

    return result.astype(img_dtype)

def max_red(img):
    img_dtype = img.dtype
    img = img.astype('float32')
    b_max = np.max(img[:, :, 0])
    g_max = np.max(img[:, :, 1])
    r_max = np.max(img[:, :, 2])
    k=(b_max,g_max,r_max)/r_max
    res=img

    res[:, :, 0] = img[:, :, 0] / k[0]
    res[:, :, 1] = img[:, :, 1] / k[1]
    res[:, :, 2] = img[:, :, 2] / k[2]

    result = np.clip(res, a_min=0, a_max=255)

    return result.astype(img_dtype)

def QCGP(img):
    img_dtype = img.dtype
    img = img.astype('float32')
    r_max = np.max(img[:, :, 0])
    g_max = np.max(img[:, :, 1])
    b_max = np.max(img[:, :, 2])
    k_max=(r_max+g_max+b_max)/3

    r_avg = np.mean(img[:, :, 0])
    g_avg = np.mean(img[:, :, 1])
    b_avg = np.mean(img[:, :, 2])
    k_avg = (r_avg + g_avg + b_avg) / 3

    r_u, r_v = cal(r_avg, r_max, k_avg, k_max)
    g_u, g_v = cal(g_avg, g_max, k_avg, k_max)
    b_u, b_v = cal(b_avg, b_max, k_avg, k_max)

    res = img

    res[:, :, 0] = (r_u *(img[:, :, 0] ** 2))+(r_v *(img[:, :, 0]))
    res[:, :, 1] = (g_u * (img[:, :, 1] ** 2)) + (g_v * (img[:, :, 1]))
    res[:, :, 2] = (b_u * (img[:, :, 2] ** 2)) + (b_v * (img[:, :, 2]))

    result = np.clip(res, a_min=0, a_max=255)

    return result.astype(img_dtype)

def cal(avg,max,k_avg,k_max):
    a = k_max-(k_avg*(max/avg))
    a=a/(max**2-avg*max)

    b=(k_avg/avg)-a*avg

    return a,b
