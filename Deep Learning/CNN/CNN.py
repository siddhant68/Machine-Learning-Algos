import numpy as np 
import cv2
import matplotlib.pyplot as plt

im = cv2.imread("alcohol-beverage-black-background-1028637.jpg")
img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
print(img.shape)
plt.imshow(img, cmap='gray')

# Gaussian blur kernel
kernel = np.array([[-1.0, -1.0, -1.0],
                   [-1.0, 8.0, -1.0],
                   [-1.0, -1.0,-1.0]
                   ])
print(kernel)

new_img = np.zeros((img.shape[0]-kernel.shape[0]+1, img.shape[1]-kernel.shape[1]+1))
plt.imshow(new_img, cmap='gray')

# convolve
for ix in range(new_img.shape[0]):
    for ij in range(new_img.shape[1]):
        
        # patch size
        im_patch = img[ix: ix+kernel.shape[0], ij: ij+kernel.shape[1]]
        k_prod = im_patch*kernel
        
        # pixels range b/w
        new_img[ix, ij] = max(0, k_prod.sum())

plt.imshow(new_img.astype(np.uint8), cmap='gray')