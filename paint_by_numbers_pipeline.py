import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from time import time
import cv2

# Options to Run Pipeline
reshape_image = True
show_original_image = False

median_kernel = 5
reshape_width = 400
reshape_height = 400
min_canny = 100
max_canny = 200
image_name = 'brendan.png'
color_code = 1 # 0 = grayscale 1 = color

num_colors = 100
test_image = cv2.imread(image_name, color_code)
print(test_image.shape)


# Check if want to see original image
if show_original_image:
	cv2.imshow('Original Image', test_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# PIPELINE STEP: Preprocessing- Image Resize
if reshape_image:
	test_image = cv2.resize(test_image, (reshape_width, reshape_height))
	(h, w) = (reshape_height, reshape_width)
else:
	(h, w) = test_image.shape[:2]


# Color Quantization- Convert Image from BGR Format to LAB
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2LAB)
test_image = test_image.reshape((test_image.shape[0] * test_image.shape[1], 3))
clusters = MiniBatchKMeans(n_clusters=num_colors)
print(clusters)
labels = clusters.fit_predict(test_image)
print(labels.shape)	# For a (711, 1067, 3) image this is (758637,); this function finds clusters and assigns labels

quantized_image = clusters.cluster_centers_.astype("uint8")[labels]
quantized_image = quantized_image.reshape((h, w, 3))
original_image = test_image.reshape((h, w, 3))

quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)
original_image = cv2.cvtColor(original_image, cv2.COLOR_LAB2BGR)



# FILTERING STEP: Median Filter to Remove Dots
blurred_image = cv2.medianBlur(quantized_image, median_kernel) # THIS MAY NEED TUNED

# EDGE DETECTION STEP
edges = cv2.Canny(blurred_image, min_canny, max_canny)
print(edges.shape)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges = cv2.bitwise_not(edges)
# Looks like we will have to get borders not edges
# Floodfill? Find contours? draw contours?

# Idea superimpose borders and floodfill them; then assign colors

# SHOW FINAL RESULTS
cv2.imshow("image", np.hstack([original_image, quantized_image, blurred_image, edges]))
cv2.waitKey(0)
