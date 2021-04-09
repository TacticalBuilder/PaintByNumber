# Paint By Numbers Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
import time
import cv2

# Options to Run Pipeline
image_name = 'test_images/carrot.jpg'			# Name of Image
reshape_image = True						# Whether to reshape image dimensions
reshape_width = 250
reshape_height = 250
color_code = 1 								# Color code to read in (0 = grayscale, 1 = BGR)
num_colors = 3								# Number of colors needed for k-means clustering
median_kernel = 5							# Size of median kernel used for blurring
min_canny = 100								# Thresholds used for Canny edge detection
max_canny = 200
perimeter_or_area_contour = 'perimeter'		# How to filter out contours
contour_threshold = 50						# Threshold to filter contours	
filter_verbose = True						# Print contour info				
use_cuda = True								# Whether to use cuda
test_sample_cuda = False
blur = 'median'
filter_contours = False

# Load imports only if CUDA is enabled
if use_cuda:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule

# Run sample test case for Py CUDA
if test_sample_cuda:
	mod = SourceModule("""
	__global__ void multiply_them(float *dest, float *a, float *b)
	{
	  const int i = threadIdx.x;
	  dest[i] = a[i] * b[i];
	}
	""")
	multiply_them = mod.get_function("multiply_them")
	a = np.random.randn(400).astype(np.float32)
	b = np.random.randn(400).astype(np.float32)
	dest = np.zeros_like(a)
	for i in range(1000):
		multiply_them(drv.Out(dest), drv.In(a), drv.In(b), block=(400,1,1), grid=(1,1))
	print(dest-a*b)
	print(dest)

# Load Image
test_image = cv2.imread(image_name, color_code)

# Reshape Image if Desired; Assign h and w
if reshape_image:
	test_image = cv2.resize(test_image, (reshape_width, reshape_height))
	(h, w) = (reshape_height, reshape_width)
else:
	(h, w) = test_image.shape[:2]
original_image = test_image.copy()

# Color Quantization
quant_start = time.clock()
quantized_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)							# Convert from BGR to LAB
quantized_image = quantized_image.reshape((test_image.shape[0] * test_image.shape[1], 3))		# Flatten h/w dimensions
clusters = MiniBatchKMeans(n_clusters=num_colors)									# Run k-means
labels = clusters.fit_predict(quantized_image)										# Finds clusters and assigns labels to pixels; (711, 1067, 3)->(758637,)
quantized_image = clusters.cluster_centers_.astype("uint8")[labels]					# Make quantized image
quantized_image = quantized_image.reshape((h, w, 3))								# Reshape quantized image
quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)					# Convert quantized image back to BGR
quant_end = time.clock()

# Median step
median_start = time.clock()
if blur == 'gaussian':
	blurred_image = cv2.GaussianBlur(quantized_image, (5,5), 0) 						# Remove noise with median kernel
else: 
	blurred_image = cv2.medianBlur(quantized_image, median_kernel) 						# Remove noise with median kernel
median_end = time.clock()

# Edge Detection Step
edge_start = time.clock()
edges = cv2.Canny(blurred_image, min_canny, max_canny)								# Get edges with Canny
edge_end = time.clock()

# Contour Detection Step
contour_image = blurred_image.copy()
contour_start = time.clock()					# for best accuracy use binary images for contour detection
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)	# CHAIN_APPROX_NONE, RETR_TREE, RETR_LIST, RETR_FLOODFILL
# CHAIN APPROX SIMPLE - saves memory
contour_end = time.clock()	
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges = cv2.bitwise_not(edges)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# a contour can be 2 points

# Filter Contours Step
filter_contour_start = time.clock()
if filter_contours:
	num_shapes = len(contours)
	if filter_verbose:
		print(num_shapes)
	new_contours = []
	for c in contours:
		if filter_verbose:
			print(c.shape)
		if perimeter_or_area_contour == 'perimeter':
			contour_value = cv2.arcLength(c,True)
		else:
			area = cv2.contourArea(c)
		if contour_value > contour_threshold:
			new_contours.append(c)
			if filter_verbose:
				print(contour_value)
else:
	new_contours = contours
filter_contour_end = time.clock()
filtered_image = blurred_image.copy()
cv2.drawContours(filtered_image, new_contours, -1, (0, 255, 0), 2) #2

# Ideas: Convex Hull, Floodfill, draw bounding box, find median/reassign

# Idea floodfill using seed point

# Get final outline
outline = cv2.bitwise_not(np.zeros((h,w,3), np.uint8))
cv2.drawContours(outline, new_contours, -1, (0, 255, 0), 2) #2


# Final Results
row_1 = np.hstack([original_image, quantized_image, blurred_image])
row_2 = np.hstack([edges, contour_image, filtered_image]) 
row_3 = np.hstack([outline, np.zeros((h,w,3), np.uint8), np.zeros((h,w,3), np.uint8)])
images_to_show = np.vstack([row_1, row_2, row_3])
cv2.imshow("Paint By Numbers", images_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print Timing
print('Timing: ')
print('Color Quantization: ' + str(quant_end - quant_start) + ' s')
print('Median Filter: ' + str(median_end - median_start) + ' s')
print('Edge Detection: ' + str(edge_end - edge_start) + ' s')
print('Contour: ' + str(contour_end - contour_start) + ' s')
print('Filter Contour: ' + str(filter_contour_end - filter_contour_start) + ' s')
