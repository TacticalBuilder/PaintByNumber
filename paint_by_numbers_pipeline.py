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
reshape_width = 500
reshape_height = 300
min_canny = 100
max_canny = 200
image_name = 'coke.jpg'
color_code = 1 # 0 = grayscale 1 = color
contour_threshold = 50
perimeter_or_area_contour = 'perimeter'

num_colors = 5
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
contour_image = blurred_image.copy()
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges = cv2.bitwise_not(edges)
# Looks like we will have to get borders not edges
# Floodfill? Find contours? draw contours?

# Find contours
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Idea remove small contours)
num_shapes = len(contours)
print(num_shapes)
new_contours = []
for c in contours:
	print(c.shape)
	if perimeter_or_area_contour == 'perimeter':
		contour_value = cv2.arcLength(c,True)
	else:
		area = cv2.contourArea(c)
	if contour_value > contour_threshold:
		new_contours.append(c)
		print(contour_value)

filtered_image = blurred_image.copy()
cv2.drawContours(filtered_image, new_contours, -1, (0, 255, 0), 2)
# convex hull

# Idea- draw bounding box around contour, find median, reassign
# flood fill

# SHOW FINAL RESULTS
row_1 = np.hstack([original_image, quantized_image, blurred_image])
row_2 = np.hstack([edges, contour_image, filtered_image])
images_to_show = np.vstack([row_1, row_2])
cv2.imshow("image", images_to_show)
cv2.waitKey(0)
#np.hstack([original_image, quantized_image, blurred_image, edges, contour_image])
