import cv2 as cv

import numpy as np



def getHairMask(img):

    img1 = img
    # OTSU阈值分割，得到2值画图像
    t, img2 = cv.threshold(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 形态学操作
    kr1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))  # kernel，长轴和短轴均为40的椭圆，即半径20的圆
    img3 = cv.morphologyEx(img2, cv.MORPH_TOPHAT, kr1)  # 礼帽/顶帽运算
    kr2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))
    img5 = cv.dilate(img3, kr2)  # 膨胀
    return img5

def remove_function_1(img):

    b, g, r = cv.split(img)

    # filter
    b = cv.medianBlur(b, 3)
    g = cv.medianBlur(g, 3)
    r = cv.medianBlur(r, 3)

    # merge
    img1 = cv.merge([b, g, r])

    # create hair mask
    # ------------------------------------------------------------------------------------------------------
    hairmask = getHairMask(img)

    # remove hair
    # ------------------------------------------------------------------------------------------------------
    result_img = cv.inpaint(img1, hairmask, 3, flags=cv.INPAINT_TELEA)
    return result_img


def remove_function_3(img):

    # Convert the original image to grayscale
    grayScale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Kernel for the morphological filtering
    kernel = cv.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)


    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)

    # inpaint the original image depending on the mask
    dst = cv.inpaint(img, thresh2, 1, cv.INPAINT_TELEA)

    return dst







