# Paint By Numbers Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
import time
import cv2
import warnings

# Options to Run Pipeline
# What impacts runtime? Image shape, number of colors
warnings.filterwarnings("ignore")
image_name = 'test_images/pizza.png'			# Name of Image
reshape_image = True						# Whether to reshape image dimensions
reshape_width = 250
reshape_height = 250
color_code = 1 								# Color code to read in (0 = grayscale, 1 = BGR)
num_colors = 3							# Number of colors needed for k-means clustering
median_kernel = 5							# Size of median kernel used for blurring
min_canny = 100								# Thresholds used for Canny edge detection
max_canny = 200
perimeter_or_area_contour = 'perimeter'		# How to filter out contours
contour_threshold = 50						# Threshold to filter contours	
filter_verbose = False						# Print contour info				
use_cuda = True								# Whether to use cuda
test_sample_cuda = False
blur = 'median'
filter_contours = False
show_results = True
use_custom_rgb_to_lab = True

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


# PIPELINE 
def convert_rgb_to_lab(img):
	assert img.dtype == 'uint8' # only handle this image

	# Convert image to fp and scale b/w 0 and 1
	img = img.astype('float32') * (1.0/255.0)

	func_srgb = lambda x : ((x+0.055)/1.055) ** (2.4) if x > 0.04045 else x / 12.92
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			for k in range(img.shape[2]):
				img[i, j, k] = func_srgb(img[i,j,k])

	transform_matrix = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
	xyz = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			bgr_val = img[i, j, :]
			xyz[i, j, :] = np.matmul(transform_matrix, np.flip(bgr_val))
	xyz[:, :, 0] = xyz[:, :, 0] / (0.950456)
	xyz[:, :, 2] = xyz[:, :, 2] / (1.088754)

	# Get L, a, b
	new_img = np.zeros(img.shape)
	func_L = lambda y : 116*(y**(1.0/3)) - 16 if y > 0.008856 else 903.3*y
	f_t = lambda t : t**(1.0/3) if t > 0.008856 else 7.787*t + (16.0/116)
	delta = 0
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			new_img[i, j, 0] = func_L(xyz[i, j, 1])		# L
			new_img[i, j, 1] = 500 * (f_t(xyz[i,j,0]) - f_t(xyz[i,j,1])) + delta
			new_img[i, j, 2] = 200 * (f_t(xyz[i,j,1]) - f_t(xyz[i,j,2])) + delta

	new_img[:, :, 0] = (new_img[:, :, 0] * (255.0/100)).astype('uint8')
	new_img[:, :, 1] = (new_img[:, :, 1] + 128).astype('uint8')
	new_img[:, :, 2] = (new_img[:, :, 2] + 128).astype('uint8')


	new_img[:, :, :] = new_img[:, :, :] 

	return new_img



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

color_cvt1_start = time.clock()
if use_custom_rgb_to_lab:
	quantized_image = convert_rgb_to_lab(test_image.copy())
else:
	quantized_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)							# Convert from BGR to LAB
color_cvt1_end = time.clock()

color_reshape1_start = time.clock()
quantized_image = quantized_image.reshape((test_image.shape[0] * test_image.shape[1], 3))		# Flatten h/w dimensions
color_reshape1_end = time.clock()

kmeans_start = time.clock()
clusters = MiniBatchKMeans(n_clusters=num_colors)									# Run k-means
kmeans_end = time.clock()

fit_predict_start = time.clock()
labels = clusters.fit_predict(quantized_image)										# Finds clusters and assigns labels to pixels; (711, 1067, 3)->(758637,)
fit_predict_end = time.clock()

assign_clusters_start = time.clock()
quantized_image = clusters.cluster_centers_.astype("uint8")[labels]					# Make quantized image
assign_clusters_end = time.clock()

color_reshape2_start = time.clock()
quantized_image = quantized_image.reshape((h, w, 3))								# Reshape quantized image
color_reshape2_end = time.clock()

color_cvt2_start = time.clock()
quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)					# Convert quantized image back to BGR
color_cvt2_end = time.clock()

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


# Get final outline
outline = cv2.bitwise_not(np.zeros((h,w,3), np.uint8))
cv2.drawContours(outline, new_contours, -1, (0, 255, 0), 2) #2


# Need to make mask from each contour
for c in new_contours:
	mask = np.zeros((h,w,3),np.uint8)
	cv2.drawContours(mask,new_contours, 0, 255, -1)



# FLOODFILL
##colors = []
#color_locs = []
##flood_image = outline.copy()
#for i in range(h):
#	for j in range(w):
#		print(i)
#		print(j)
#		seed_point = (i,j)
#		if str(blurred_image[i,j]) not in colors:
#			print(i)
#			print(j)
#			print(tuple([int(x) for x in blurred_image[i,j]]))
#			cv2.floodFill(flood_image, None, seedPoint=seed_point, newVal=tuple([int(x) for x in blurred_image[i,j]]), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
#			color_locs.append(str([i,j]))
#			colors.append(str(blurred_image[i,j]))
#print(colors)
#print(color_locs)

# Calculate mean color within contour
#seed_point = (100, 77)
#cv2.floodFill(flood_image, None, seedPoint=seed_point, newVal=(255, 0, 0), loDiff=(0, 0, 0, 0), upDiff=(0, 0, 0, 0))
crayola_image = np.zeros((h,w,3), np.uint8)

# Final Results
if show_results:
	row_1 = np.hstack([original_image, quantized_image, blurred_image])
	row_2 = np.hstack([edges, contour_image, filtered_image]) 
	row_3 = np.hstack([outline, crayola_image, np.zeros((h,w,3), np.uint8)])
	images_to_show = np.vstack([row_1, row_2, row_3])
	cv2.imshow("Paint By Numbers", images_to_show)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Print Timing
print('Timing: ')
print('Color Quantization: ' + str(quant_end - quant_start) + ' s')
print('\tFit Predict: ' + str(fit_predict_end - fit_predict_start) + ' s')
print('\tReshape 1: ' + str(color_reshape1_end - color_reshape1_start) + ' s')
print('\tConvert 1: ' + str(color_cvt1_end - color_cvt1_start) + ' s')
print('\tK-Means: ' + str(kmeans_end - kmeans_start) + ' s')
print('\tAssign Clusters: ' + str(assign_clusters_end - assign_clusters_start) + ' s')
print('\tReshape 2: ' + str(color_reshape2_end - color_reshape2_start) + ' s')
print('\tConvert 2: ' + str(color_cvt2_end - color_cvt2_start) + ' s')
print('Median Filter: ' + str(median_end - median_start) + ' s')
print('Edge Detection: ' + str(edge_end - edge_start) + ' s')
print('Contour: ' + str(contour_end - contour_start) + ' s')
print('Filter Contour: ' + str(filter_contour_end - filter_contour_start) + ' s')
