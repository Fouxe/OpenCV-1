import cv2 
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

####################################
# PART 1
# Function to add noise
def snp_noise(img):
    
    # get the rows and columns from the image
    row, col = img.shape
    
    # randomize white spots distribution
    num_of_pixels = random.randint(7500,75000)
    for i in range(num_of_pixels):
        y_cord = random.randint(0, row - 1)
        x_cord = random.randint(0, col - 1)
        
        img[y_cord][x_cord] = 255
    
    # randommize black spot distribution
    num_of_pixels = random.randint(7500,75000)
    for i in range(num_of_pixels):
        y_cord = random.randint(0, row - 1)
        x_cord = random.randint(0, col - 1)
        
        img[y_cord][x_cord] = 0
    return img

# Turn Img to grayscale
img = cv2.imread('a12_Color.png')
grayimg = cv2.imread('a12_Color.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('snp_apple.png',snp_noise(grayimg))

# Apply Median Filtering 
m_blur = cv2.medianBlur(snp_noise(grayimg), 3) # (img source,kernel size)
cv2.imwrite("m_blur.png",m_blur)
####################################
#PART 2
# Robert
roberts_cross_v = np.array( [[10, -5 ],[5,-10 ]] )
roberts_cross_h = np.array( [[ -5, 10 ],[ -10, 5 ]] )
  
img = cv2.imread("m_blur.png",0).astype('float64')
img/=255.0
vertical = ndimage.convolve( img, roberts_cross_v )
horizontal = ndimage.convolve( img, roberts_cross_h )

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img*=255
cv2.imwrite("robert.bmp",edged_img)

####################################
#Part 3
rob = cv2.imread('robert.bmp',0)
kernel = np.ones((5,5), np.uint8)

#Erosion
rob_erosion = cv2.erode(rob, kernel, iterations=1)
cv2.imwrite('Erosion.bmp', rob_erosion)

#Dilation
rob_dilation = cv2.dilate(rob_erosion, kernel, iterations=1)
cv2.imwrite('Dilation.bmp', rob_dilation)

#Opening
rob_opening = cv2.morphologyEx(rob_dilation, cv2.MORPH_OPEN, kernel)
cv2.imwrite('Opening.bmp',rob_opening)

#Closing
rob_closing = cv2.morphologyEx(rob_opening, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('Closing.bmp',rob_closing)

#Fill
th, im_th = cv2.threshold(rob_closing, 50, 225, cv2.THRESH_BINARY_INV)
im_floodfill = im_th.copy()
h, w = im_th.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, mask, (0,0), 255)
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_out = im_th | im_floodfill_inv
cv2.imwrite("ThresholdedImage.bmp",im_th)
cv2.imwrite("FloodfilledImage.bmp",im_floodfill)
cv2.imwrite("InvertedFloodfilledImage.bmp",im_floodfill_inv)
cv2.imwrite("Fill.bmp", im_out)

# plt.show() workarounds
# Take note that plt.show() stores imgs in BGR and nor RGB
# Therefore before we add them we need to  revert img first
fill = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
titles = ['Original Image', 'Grayscaled', 'Noise Application',
            'Median Blurring', 'Robert Thresholding',
            'Dilation', 'Opening', 'Erosion',
            'Closing', 'Fill'] 

images = [img, grayimg, snp_noise(grayimg), m_blur, rob,
          rob_erosion, rob_dilation, rob_opening, rob_closing, fill]
for i in range(10):
    plt.subplot(5,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
