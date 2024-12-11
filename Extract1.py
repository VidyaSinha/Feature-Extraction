import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate histogram for each channel (Red, Green, Blue)
r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

# Plot histograms
plt.plot(r_hist, color='red')
plt.plot(g_hist, color='green')
plt.plot(b_hist, color='blue')
plt.title("Color Histogram")
plt.show()





# Read image
image = cv2.imread('image.jpg', 0)  # Grayscale image

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Display the result
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection')
plt.show()









from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte

# Read image and convert to grayscale
image = cv2.imread('image.jpg', 0)

# Apply Local Binary Pattern (LBP)
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

# Display LBP
plt.imshow(lbp, cmap='gray')
plt.title('Local Binary Pattern (Texture)')
plt.show()













from skimage.feature import hog
from skimage import exposure

# Read image and convert to grayscale
image = cv2.imread('image.jpg', 0)

# Compute HOG features
features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Enhance the HOG image for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Display the HOG image
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title('Histogram of Oriented Gradients')
plt.show()

