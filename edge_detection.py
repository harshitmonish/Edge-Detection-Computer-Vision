# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 23:13:52 2021

@author: harshit monish
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    
    r, c = 3, 3    
    denoise_img = img
    img1 = np.zeros(img.shape)
    
    # Adding zero padding in the image
    img2 = np.pad(denoise_img, pad_width=1, mode='constant')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # fetching the 3x3 image window
            temp = img2[i:i+r, j:j+c]
            
            # sorting the window to get the middle element as median filter
            temp = np.sort(temp.flatten())
            img1[i][j] = temp[4]
                
    denoise_img = img1.astype(np.int32)    
    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    
    #fetching the shape of the kernel
    r, c = kernel.shape
    flip_kernel = np.zeros(kernel.shape)
    conv_img = img
    
    # adding the zero padding of 2 pixel each side of the image
    conv_img = np.zeros(img.shape)
    conv_img_pad = np.pad(img, pad_width=1, mode='constant')
    i = r-1
    j = c-1
    
    #flipping the kernel for convolution
    while (i >= 0):
        while( j >= 0):
            flip_kernel[i][j] = kernel[r-1-i][c-1-j]
            j-=1
        i-=1
        j = c-1
        
    # implementing the convolution operation on the padded image and updating
    # the results in conv_img    
    for i in range(conv_img.shape[0]):
        for j in range(conv_img.shape[1]):
            # fetching the region of interest of the image
            temp = conv_img_pad[i:i+r,j:j+c]
            
            # convolution operation
            conv = np.sum(np.multiply(temp, flip_kernel))
            
            # Updating the convolution output.
            conv_img[i][j] = conv       
    return conv_img

def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image, 
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    
    # computing the convolution using sobel_x, sobel_y filters to get vertical
    # and horizontal images
    edge_x = convolve2d(img, sobel_x)
    edge_y = convolve2d(img, sobel_y)

    # Computing the image magnitude
    edge_mag = np.zeros(edge_x.shape, dtype='int32')
    edge_mag = np.sqrt(edge_x**2 + edge_y**2)
    
    #normalizing edge_x
    edge_x = 255*((edge_x - np.amin(edge_x))/(np.amax(edge_x) - np.amin(edge_x))) 
    # typecasting into int type so that can be saved properly
    edge_x = edge_x.astype(np.int32)
    
    #normalizing edge_y
    edge_y = 255*((edge_y - np.amin(edge_y))/(np.amax(edge_y) - np.amin(edge_y))) 
    # typecasting into int type so that can be saved properly
    edge_y = edge_y.astype(np.int32)
    
    #normalize edge_mag
    edge_mag = 255*((edge_mag - np.amin(edge_mag))/(np.amax(edge_mag) - np.amin(edge_mag)))
    edge_mag = edge_mag.astype(np.int32)
    
    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #raise NotImplementedError
    sobel_135 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(int)
    sobel_45 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).astype(int)
    edge_45 = convolve2d(img, sobel_45)
    edge_135 = convolve2d(img, sobel_135)
    
    #normalizing edge_45
    edge_45 = 255*((edge_45 - np.amin(edge_45))/(np.amax(edge_45) - np.amin(edge_45))) 
    
    # typecasting into int type so that can be saved properly
    edge_45 = edge_45.astype(np.int32)
    
    #normalizing edge_135
    edge_135 = 255*((edge_135 - np.amin(edge_135))/(np.amax(edge_135) - np.amin(edge_135))) 
    
    # typecasting into int type so that can be saved properly
    edge_135 = edge_135.astype(np.int32)    
    print(" Sobel 45 Kernel:")
    print(sobel_45) # print the two kernels you designed here
    print(" \n Sobel 135 Kernel:")
    print(sobel_135)
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)





