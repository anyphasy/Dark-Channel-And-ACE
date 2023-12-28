import cv2
import numpy as np
import time


# 实现最小值滤波，用于计算暗通道。
def zmMinFilterGray(src, r=7):
    # 输入参数 src 是源图像，r 是滤波器半径。
    # 最小值滤波，r是滤波器半径
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用 cv2.erode 进行图像的侵蚀操作，得到暗通道图像。


# 实现导向滤波，用于优化图像处理。
def guidedfilter(I, p, r, eps):
    # 输入参数包括 I 和 p 作为图像及其引导图像，r 是滤波器半径，eps 是一个小的正值。
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I
    # 通过 cv2.boxFilter 函数对图像进行滤波处理，得到优化后的结果。
    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


# 计算大气光照和遮罩图像
def Defog(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    # 计算大气遮罩图像V1和光照值A, V1 = 1-t/A
    V1 = np.min(m, 2)  # 得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 7)

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用导向滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制
    return V1, A


# 对图像进行去雾处理
def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - Mask_img) / (1 - Mask_img / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


if __name__ == '__main__':
    start_time = time.time()  # 获取代码开始执行的时间戳
    m = cv2.imread('swan.png') / 255.0

    m_result = deHaze(m) * 255  # 对图像进行归一化处理，然后调用 deHaze 函数进行去雾处理。
    end_time = time.time()  # 获取代码执行结束的时间戳

    elapsed_time = end_time - start_time  # 计算耗时时长
    print(f"代码运行耗时：{elapsed_time} 秒")
    # 通过 cv2.imshow 显示原始图像、暗通道图像和去雾后的结果图像
    # dark_channel = np.min(m, 2)
    # Dark_Channel = zmMinFilterGray(dark_channel, 7)
    cv2.imshow('Original', m)
    # cv2.imshow('DarkChannelImage', Dark_Channel)
    cv2.imshow('AfterDarkChannel', m_result.astype(np.uint8))
    cv2.imwrite('AfterDarkChannel.png', m_result.astype(np.uint8))
    # cv2.imwrite('AfterDarkChannel.jpg', m_result.astype(np.uint8))
    betaBright = 40
    brightened_image = np.clip(m_result.astype(np.float32) + betaBright, 0, 255).astype(np.uint8)

    cv2.imshow('AfterDarkChannelBrightenedImage', brightened_image)
    # cv2.imshow('AfterDarkChannelBrightenedImageConvertScaleAbs', brightened_image1)

    # cv2.imwrite('AfterDarkChannelBrightenedImage.png', brightened_image)
    cv2.imwrite('AfterDarkChannelBrightenedImage.png', brightened_image)

    # cv2.imwrite('AfterACEAndDarkChannel.jpg', m_result.astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
