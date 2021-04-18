# Paint By Numbers Pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
import time
import cv2
import warnings
warnings.filterwarnings("ignore")

# Parameterizable Settings

# Image Settings
image_name = 'test_images/crosby.jpg'			# Name of Image
reshape_image = True						# Whether to reshape image dimensions
reshape_width = 256 
reshape_height = 256
assert reshape_width <= 512					# Max threads per block

color_code = 1 								# Color code to read in (0 = grayscale, 1 = BGR)
num_colors = 20								# Number of colors needed for k-means clustering
median_kernel = 5							# Size of median kernel used for blurring
blur = 'median'								# 'median' or 'gaussian'
show_results = True							# Show plots?
use_custom_rgb_to_lab = True				# Use custom RGB to LAB conversion function

# GPU Settings
use_gpu = True								# Whether to use cuda

# Load imports only if CUDA is enabled
if use_gpu:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule

###########################################################################################################################
# FUNCTIONS

# Custom convert RGB to LAB function
def convert_rgb_to_lab(img):
	assert img.dtype == 'uint8' # only handle this image
	img = img.astype('float32') * (1.0/255.0) # Convert to fp b/w 0 and 1

	# Gamma
	func_srgb = lambda x : ((x+0.055)/1.055) ** (2.4) if x > 0.04045 else x / 12.92
	vectorized_func_srgb = np.vectorize(func_srgb)
	img = vectorized_func_srgb(img)

	# Convert to XYZ and scale
	transform_matrix = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
	xyz = np.zeros(img.shape)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			xyz[i, j, :] = np.matmul(transform_matrix, np.flip(img[i, j, :]))
	xyz[:, :, 0] = xyz[:, :, 0] / (0.950456)
	xyz[:, :, 2] = xyz[:, :, 2] / (1.088754)

	# Get L, a, b
	new_img = np.zeros(img.shape)
	func_L = lambda y : 116*(y**(1.0/3)) - 16 if y > 0.008856 else 903.3*y
	vectorized_func_L = np.vectorize(func_L)
	f_t = lambda t : t**(1.0/3) if t > 0.008856 else 7.787*t + (16.0/116)
	vectorized_func_f_t = np.vectorize(f_t)
	delta = 0
	new_img[:, :, 0] = vectorized_func_L(xyz[:, :, 1])
	new_img[:, :, 1] = 500 * (vectorized_func_f_t(xyz[:,:,0]) - vectorized_func_f_t(xyz[:,:,1])) + delta
	new_img[:, :, 2] = 200 * (vectorized_func_f_t(xyz[:,:,1]) - vectorized_func_f_t(xyz[:,:,2])) + delta
	# Scale
	new_img[:, :, 0] = (new_img[:, :, 0] * (255.0/100)).astype('uint8')
	new_img[:, :, 1] = (new_img[:, :, 1] + 128).astype('uint8')
	new_img[:, :, 2] = (new_img[:, :, 2] + 128).astype('uint8')

	return new_img


# This is one of the cuda kernels; does not work yet but in process of making
def convert_rgb_to_lab_gpu(img):
	# Convert to grayscale?
	mod = SourceModule("""
	__global__ void convert_rgb_to_lab(float *out_l, float *out_a, float *out_b, float *b, float *g, float *r)
	{
	  int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	  float local_b = 0;
	  float local_g = 0;
	  float local_r = 0;
	  if (b[index] > 0.04045) {
	      local_b = pow((b[index] + 0.055/1.055), 2.4);
	  }
	  else {
	  	  local_b = b[index] / 12.92;
	  }
	  if (g[index] > 0.04045) {
	      local_g = pow((g[index] + 0.055/1.055), 2.4);
	  }
	  else {
	  	  local_g = g[index] / 12.92;
	  }
	  if (r[index] > 0.04045) {
	      local_r = pow((r[index] + 0.055/1.055), 2.4);
	  }
	  else {
	  	  local_r = r[index] / 12.92;
	  }

	  float temp_sum = 0;
	  const int row_size = blockDim.x;
	  const int col_size = gridDim.x;
	  int i = (index / row_size);
	  int j = (index % row_size); 
	  float local_x = (0.412453 * local_r + 0.357580 * local_g + 0.180423 * local_b) / 0.950456;
	  float local_y = 0.212671 * local_r + 0.715160 * local_g + 0.072169 * local_b;
	  float local_z = (0.019334 * local_r + 0.119193 * local_g + 0.950227 * local_b) / 1.088754;
	  float f_x = 0;
	  float f_y = 100;
	  float f_z = 0;
	  float delta = 0;
	  if (local_x > 0.008856){
	      f_x = cbrtf(local_x);
	  } else {
          f_x = 7.787*local_x + (16.0/116.0);
	  }
	  if (local_y > 0.008856){
	      f_y = cbrtf(local_y);
	  } else {
          f_y = 7.787*local_y + (16.0/116.0);
	  }
	  if (local_z > 0.008856){
	      f_z = cbrtf(local_z);
	  } else {
          f_z = 7.787*local_z + (16.0/116.0);
	  }
	  if (local_y > 0.008856){
	  	  out_l[index] = 116 * cbrtf(local_y) - 16;
	  }
	  else {
	  	  out_l[index] = 903.3 * local_y;
	  }
	  out_a[index] = 500 * (f_x - f_y) + delta;
	  out_b[index] = 200 * (f_y - f_z) + delta;
	}
	""")
	# Add if statement to handle i and j
	convert_rgb_to_lab = mod.get_function("convert_rgb_to_lab") # block dim .x is 256, gridim.x is 256
	img = img.astype('float32') * (1.0/255.0)
	b = img[:, :, 0].flatten().astype('float32')
	g = img[:, :, 1].flatten().astype('float32') # VERY IMPORTANT TO MAKE UINT32
	r = img[:, :, 2].flatten().astype('float32')
	out_l = np.zeros_like(b)
	out_a = np.zeros_like(g)
	out_b = np.zeros_like(r)
	convert_rgb_to_lab(drv.Out(out_l), drv.Out(out_a), drv.Out(out_b), drv.In(b), drv.In(g), drv.In(r), block=(w,1,1), grid=(h,1,1))
	new_img = np.zeros_like(img)
	new_img[:, :, 0] = (np.reshape(out_l, (h, w)) * 255.0/100).astype('uint8')
	new_img[:, :, 1] = (np.reshape(out_a, (h, w)) + 128).astype('uint8')
	new_img[:, :, 2] = (np.reshape(out_b, (h, w)) + 128).astype('uint8')
	return new_img

###########################################################################################################################
# CHECK THIS -> This will remove small edges; need to debug more 
def outline_cpu(img):
	canvas_image = np.ones((h,w,3), np.uint8) * 255
	outline_image = np.ones((h,w), np.uint8) * 255
	for i in range(w):
		canvas_image[0][i] = [0, 0, 0]
		canvas_image[h-1][i] = [0, 0, 0]
	for i in range(h):
		canvas_image[i][0] = [0, 0, 0]
		canvas_image[i][w-1] = [0, 0, 0]
	for i in range(1, h - 1):
		for j in range(1, w - 1):
			pixel_val = blurred_image[i][j] # 123, 4X5, 678
			if not np.array_equal(pixel_val, blurred_image[i-1][j-1]):
				canvas_image[i,j] = [0, 0, 0]   
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, blurred_image[i-1][j]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, blurred_image[i-1][j+1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, blurred_image[i][j-1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, blurred_image[i][j+1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0 
			elif not np.array_equal(pixel_val, blurred_image[i+1][j-1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0 
			elif not np.array_equal(pixel_val, blurred_image[i+1][j]):
				canvas_image[i,j] = [0, 0, 0] 
				outline_image[i,j] = 0
			elif not np.array_equal(pixel_val, blurred_image[i+1][j+1]):
				canvas_image[i,j] = [0, 0, 0]
				outline_image[i,j] = 0
			else: 
				canvas_image[i, j] = blurred_image[i, j]
	return canvas_image, outline_image

def outline_gpu(img):
	# Convert to grayscale?
	mod = SourceModule("""
	__global__ void outline(int *border, int *out_b, int *out_g, int *out_r, int *b, int *g, int *r)
	{
	  int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	  int curr_b = b[index];
	  int curr_g = g[index];
	  int curr_r = r[index];
	  const int row_size = blockDim.x;
	  const int col_size = gridDim.x;
	  int i = (index / row_size);
	  int j = (index % row_size); 
	  if (i >= 1 && j >= 1 && i < col_size - 1 && j < row_size - 1){
	  	  if (curr_b != b[(i-1)*row_size+(j-1)] && curr_g != g[(i-1)*row_size+(j-1)] && curr_r != r[(i-1)*row_size+(j-1)]){ 
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i-1)*row_size+(j)] && curr_g != g[(i-1)*row_size+(j)] && curr_r != r[(i-1)*row_size+(j)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;

	  	  }
	  	  else if (curr_b != b[(i-1)*row_size+(j+1)] && curr_g != g[(i-1)*row_size+(j+1)] && curr_r != r[(i-1)*row_size+(j+1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i)*row_size+(j-1)] && curr_g != g[(i)*row_size+(j-1)] && curr_r != r[(i)*row_size+(j-1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i)*row_size+(j+1)] && curr_g != g[(i)*row_size+(j+1)] && curr_r != r[(i)*row_size+(j+1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i+1)*row_size+(j-1)] && curr_g != g[(i+1)*row_size+(j-1)] && curr_r != r[(i+1)*row_size+(j-1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i+1)*row_size+(j)] && curr_g != g[(i+1)*row_size+(j)] && curr_r != r[(i+1)*row_size+(j)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else if (curr_b != b[(i+1)*row_size+(j+1)] && curr_g != g[(i+1)*row_size+(j+1)] && curr_r != r[(i+1)*row_size+(j+1)]){
	  	  		out_b[index] = 0;
	  	  		out_g[index] = 0;
	  	  		out_r[index] = 0;
	  	  		border[index] = 0;
	  	  }
	  	  else {
	  	  		out_b[index] = curr_b;
	  	  		out_g[index] = curr_g;
	  	  		out_r[index] = curr_r;
	  	  		border[index] = 255;
	  	  }
	  }
	  else {
	  	  border[index] = 0;
	  }	
	}
	""")
	outline = mod.get_function("outline") # block dim .x is 256, gridim.x is 256
	b = img[:, :, 0].flatten().astype('uint32')
	g = img[:, :, 1].flatten().astype('uint32') # VERY IMPORTANT TO MAKE UINT32
	r = img[:, :, 2].flatten().astype('uint32')
	out_b = np.zeros_like(b)
	out_g = np.zeros_like(g)
	out_r = np.zeros_like(r)
	border = np.ones_like(b) * 255
	outline(drv.Out(border), drv.Out(out_b), drv.Out(out_g), drv.Out(out_r), drv.In(b), drv.In(g), drv.In(r), block=(w,1,1), grid=(h,1,1))
	canvas = np.zeros_like(img)
	canvas[:, :, 0] = np.reshape(out_b, (h, w))
	canvas[:, :, 1] = np.reshape(out_g, (h, w))
	canvas[:, :, 2] = np.reshape(out_r, (h, w))
	border = np.reshape(border, (h, w))
	return canvas, border


###########################################################################################################################
# PIPELINE

start_pipeline = time.clock()
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

## Convert RGB to LAB
color_cvt1_start = time.clock()
if use_custom_rgb_to_lab and use_gpu:
	quantized_image = convert_rgb_to_lab_gpu(original_image.copy())
elif use_custom_rgb_to_lab:
	quantized_image = convert_rgb_to_lab(original_image.copy())
else:
	quantized_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)							
color_cvt1_end = time.clock()

## Flatten h/w
color_reshape1_start = time.clock()
quantized_image = quantized_image.reshape((test_image.shape[0] * test_image.shape[1], 3))		
color_reshape1_end = time.clock()

# Initialize k-means
kmeans_start = time.clock()
clusters = MiniBatchKMeans(n_clusters=num_colors)				# TO DO: Check if this can be						
kmeans_end = time.clock()

# Find clusters and assign labels to pixels (711, 1067, 3) -> (758637,)
fit_predict_start = time.clock()
labels = clusters.fit_predict(quantized_image)										
fit_predict_end = time.clock()

# Make quantized image
assign_clusters_start = time.clock()
quantized_image = clusters.cluster_centers_.astype("uint8")[labels]				
assign_clusters_end = time.clock()

# Reshape quantized image
color_reshape2_start = time.clock()
quantized_image = quantized_image.reshape((h, w, 3))								
color_reshape2_end = time.clock()

# Convert quantized image from LAB to BGR
color_cvt2_start = time.clock()
quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)					
color_cvt2_end = time.clock()

quant_end = time.clock()		# End quantization

# Blurring step
median_start = time.clock()
if blur == 'gaussian':
	blurred_image = cv2.GaussianBlur(quantized_image, (5,5), 0) 						# Remove noise with gaussian kernel
else: 
	blurred_image = cv2.medianBlur(quantized_image, median_kernel) 						# Remove noise with median kernel
median_end = time.clock()

# Make Canvas and Outline 
outline_start = time.clock()
if use_gpu:
	canvas_image, outline_image = outline_gpu(blurred_image)
else: 
	canvas_image, outline_image = outline_cpu(blurred_image)
outline_end = time.clock()

# Get Connected Components 
dfs_start = time.clock()
canvas_image_gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
marked_mask = np.zeros((h,w))
dx = [-1, 0, 1, 1, 1, 0, -1, 1]
dy = [1, 1, 1, 0, -1, -1, -1, 0]
component_num = 1

# DFS to assign regions
def dfs(img, x_val, y_val, component_num):
	stack = []
	stack.append((x_val,y_val))
	while not len(stack) == 0:
		(x,y) = stack.pop()
		for i in range(8):
			nx = x + dx[i]
			ny = y + dy[i]
			if img[nx, ny] != 0 and marked_mask[nx, ny] == 0:
				stack.insert(0, (nx, ny))
				marked_mask[nx, ny] = component_num
# Run DFS
for i in range(1, h-1):
	for j in range(1, w-1):
		if canvas_image_gray[i, j] != 0  and marked_mask[i, j] == 0:
			dfs(canvas_image_gray, i, j, component_num)
			component_num = component_num + 1
dfs_end = time.clock()


# Get num components
print('Num of Components: ' + str(component_num))
brendan_image = marked_mask.copy()							# I think this is what you need @Brendan, lmk
print(brendan_image)
marked_mask = marked_mask.astype('uint8') * 20				# This is dummy line to show the 'marked_mask'; not robust
marked_mask = cv2.cvtColor(marked_mask, cv2.COLOR_GRAY2BGR)

end_pipeline = time.clock()

# Final Results
if show_results:
	outline_image = cv2.cvtColor(outline_image.astype('uint8'), cv2.COLOR_GRAY2BGR)
	row_1 = np.hstack([original_image, quantized_image, blurred_image])
	row_2 = np.hstack([outline_image, canvas_image, marked_mask])
	images_to_show = np.vstack([row_1, row_2])
	cv2.imshow("Paint By Numbers", images_to_show)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Print Timing
print('Timing: ')
print('Color Quantization: ' + str(quant_end - quant_start) + ' s')
print('\tFit Predict: ' + str(fit_predict_end - fit_predict_start) + ' s')
print('\tReshape 1: ' + str(color_reshape1_end - color_reshape1_start) + ' s')
print('\tConvert 1: ' + str(color_cvt1_end - color_cvt1_start) + ' s')
print('\tK-Means Init: ' + str(kmeans_end - kmeans_start) + ' s')
print('\tAssign Clusters: ' + str(assign_clusters_end - assign_clusters_start) + ' s')
print('\tReshape 2: ' + str(color_reshape2_end - color_reshape2_start) + ' s')
print('\tConvert 2: ' + str(color_cvt2_end - color_cvt2_start) + ' s')
print('Blur Filter: ' + str(median_end - median_start) + ' s')
print('Outlining: ' + str(outline_end - outline_start) + ' s')
print('Connected Components: ' + str(dfs_end - dfs_start) + ' s')
print('Total Pipeline Time: ' + str(end_pipeline - start_pipeline) + ' s')
