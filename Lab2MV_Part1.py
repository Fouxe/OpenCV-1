import cv2 
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter

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
grayimg = cv2.imread('a12_Color.png',cv2.IMREAD_GRAYSCALE)
cv2.imwrite('snp_apple.png',snp_noise(grayimg))
snpimg = cv2.imread('snp_apple.png')

# Linear Filtering:Convolution
kernel = np.ones((5,5),np.float32)/25
Convo = cv2.filter2D(snp_noise(grayimg),-1,kernel)
cv2.imwrite('Convolution.png',Convo)


# Apply Median Filtering 
m_blur = cv2.medianBlur(snp_noise(grayimg), 3) # (img source,kernel size)
cv2.imwrite("m_blur.png",m_blur)

# Apply Gaussian Blur
g_blur = cv2.GaussianBlur(snp_noise(grayimg),(5,5),0) # (img,kernel,width,length)
cv2.imwrite('gasblur.png',g_blur)

# Adaptive Filter Mean and Gauss
adapt_filter_mean = cv2.adaptiveThreshold(snp_noise(grayimg),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
cv2.imwrite('adaptive_mean_threshold.png',adapt_filter_mean)
adapt_filter_gauss = cv2.adaptiveThreshold(snp_noise(grayimg),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imwrite('adaptive_gauss_threshold.png',adapt_filter_gauss)

# Sobel
sobelx = cv2.Sobel(m_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(m_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(m_blur, ddepth=cv2.CV_64F, dx=2, dy=2, ksize=7)
cv2.imwrite('Sobel X.png',sobelx)
cv2.imwrite('Sobel Y.png',sobely)
cv2.imwrite('Sobel XY.png',sobelxy)

# Prewitt
kernelx = np.array([[8,8,8],[1.05,-0.7,-1],[-8,-8,-8]])
kernely = np.array([[-8.5,-8.5,8.5],[-1,-0.3,1],[-8.5,8.5,8.5]])
img_prewittx = cv2.filter2D(m_blur, -1, kernelx)
img_prewitty = cv2.filter2D(m_blur, -1, kernely)
img_prewittxy = img_prewittx + img_prewitty
cv2.imwrite("Prewitt X.png", img_prewittx)
cv2.imwrite("Prewitt Y.png", img_prewitty)
cv2.imwrite("Prewitt.png", img_prewittxy)

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
realrob = cv2.imread('robert.bmp')
#Canny
edges = cv2.Canny(m_blur, threshold1=15, threshold2=37)
cv2.imwrite("Canny.png",edges)

#Otsu
ret, thresh1 = cv2.threshold(m_blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
cv2.imwrite('Otsu Threshold.png', thresh1) 


# Using robert for Part 3
cv2.imwrite("robert.bmp",edged_img)
rob = cv2.imread('robert.bmp',0)
kernel = np.ones((5,5), np.uint8)

#Erosion
rob_erosion = cv2.erode(rob, kernel, iterations=1)
cv2.imwrite('Erosion.bmp', rob_erosion)
#Dilation
rob_dilation = cv2.dilate(rob, kernel, iterations=1)
cv2.imwrite('Dilation.bmp', rob_dilation)
#Opening
rob_opening = cv2.morphologyEx(rob, cv2.MORPH_OPEN, kernel)
cv2.imwrite('Opening.bmp',rob_opening)
#Closing
rob_closing = cv2.morphologyEx(rob, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('Closing.bmp',rob_opening)

#Fill
th, im_th = cv2.threshold(rob, 50, 225, cv2.THRESH_BINARY_INV)
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


# Work Arounds for plt.show()

# Take note that OpenCV stores imgs in BGR and not RGB
# For fill we need to convert it from BGR to RGB
fill = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
# Due to how sobel syntax works
# We need to imwrite and image then assign imread to a variable
imgSobx = cv2.imread('Sobel X.png')
imgSoby = cv2.imread('Sobel Y.png')
imgSobxy = cv2.imread('Sobel XY.png')
# to fill in matrixes
blanks = np.zeros((100,100,3), dtype=np.uint8)
                                
titles = ['Original Image', 'Grayscaled', 'Noise Application','Median Blurring','Gaussian Blurring', "Convolution",
          'Adaptive Filtering Mean', 'Adaptive Filtering Gaussian',
          'Sobel Detection X', 'Sobel Detection Y', 'Full Sobel Edge Detection',
          'Prewitt X', 'Prewitt Y', 'Full Prewitt Edge Detection',
          'Robert Edge Detection', 
          'Canny Edge Detection',
          'Otsu Threshold',
          'Erosion', 'Dilation', 'Opening', 'Closing', 'Fill',
          'Blank','Blank','Blank'] 

images = [img, grayimg, snp_noise(grayimg), m_blur, g_blur, Convo,
          adapt_filter_mean, adapt_filter_gauss,
          imgSobx, imgSoby, imgSobxy,
          img_prewittx, img_prewitty, img_prewittxy,
          realrob,
          edges,
          thresh1,
          rob_erosion,rob_dilation, rob_opening, rob_closing, fill,
          blanks, blanks, blanks, blanks]
for i in range(25):
    plt.subplot(5,5,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()