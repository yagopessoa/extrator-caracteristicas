#!/usr/bin/python3

import cv2
import numpy as np
import sys
import os
import pickle
import datetime
import base64
import io
from matplotlib import pyplot as plt
from PIL import Image

import extract_feature

# x = np.random.randint(25,100,25)
# y = np.random.randint(175,255,25)
# z = np.hstack((x,y))
# z = z.reshape((50,1))
# z = np.float32(z)
# # plt.hist(z,256,[0,256]),plt.show()

# # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# # Set flags (Just to avoid line break in the code)
# flags = cv2.KMEANS_RANDOM_CENTERS

# # Apply KMeans
# compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)

# A = z[labels==0]
# B = z[labels==1]

# # Now plot 'A' in red, 'B' in blue, 'centers' in yellow
# plt.hist(A,256,[0,256],color = 'r')
# plt.hist(B,256,[0,256],color = 'b')
# plt.hist(centers,32,[0,256],color = 'y')
# plt.show()

# img = cv2.imread('C:\\Users\\yagor\\extrator-caracteristicas\\banco_imagens\\Parthenon\\spencer-davis-1533814-unsplash.jpg', cv2.COLOR_BGR2RGB)

# # blur = cv2.bilateralFilter(img,9,500,500)
# cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# canny = cv2.Canny(cinza, 150,150)

# plt.subplot(121),plt.imshow(img)
# plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(canny)
# plt.title('Imagem filtrada'), plt.xticks([]), plt.yticks([])

# plt.show()

imagens = extract_feature.ler_diretorio_imagens("banco_imagens/Colosseum")
imagens += extract_feature.ler_diretorio_imagens("banco_imagens/Eiffel")
imagens += extract_feature.ler_diretorio_imagens("banco_imagens/Louvre")
imagens += extract_feature.ler_diretorio_imagens("banco_imagens/Parthenon")

size = 300, 300

for imagem in imagens:
    real_img = Image.open(imagem)
    sqr_img = extract_feature.make_square(real_img)
    sqr_img.thumbnail(size, Image.ANTIALIAS)
    sqr_img.save(imagem.replace('banco_imagens', 'banco_imagens_sqr').replace('.jpg', '.png'))
