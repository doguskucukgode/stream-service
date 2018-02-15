# External imports
import cv2
import numpy as np

def enhance_image(image):
    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    Y, Cr, Cb = cv2.split(image_YCrCb)
    Y = cv2.equalizeHist(Y)
    image_YCrCb = cv2.merge([Y, Cr, Cb])
    image = cv2.cvtColor(image_YCrCb, cv2.COLOR_YCR_CB2BGR)
    return image

def calculate_gamma_LUT(gamma=1.0):
    l = [((i / 255.0) ** gamma) * 255 for i in range(1, 256)]
    l.insert(0, 0.0)
    table = np.array(l).astype("uint8")
    return table

def adjust_gamma_channelwise(image):
    print("Adjust gamma called")
    # Gamme Correction
    r_mean = np.mean(image[:,:,0]) / 255.0
    g_mean = np.mean(image[:,:,1]) / 255.0
    b_mean = np.mean(image[:,:,2]) / 255.0
    gamma_r = -0.3 / np.log10(r_mean)
    gamma_g = -0.3 / np.log10(g_mean)
    gamma_b = -0.3 / np.log10(b_mean)
    table_r = calculate_gamma_LUT(gamma=gamma_r)
    table_g = calculate_gamma_LUT(gamma=gamma_g)
    table_b = calculate_gamma_LUT(gamma=gamma_b)
    image[:, :, 0] = cv2.LUT(image[:,:,0], table_r)
    image[:, :, 1] = cv2.LUT(image[:,:,1], table_g)
    image[:, :, 2] = cv2.LUT(image[:,:,2], table_b)
    return image

def prewhiten(x):
    x = x.astype("float32")
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
