---
layout: post
title: Imagen como grilla de píxeles
category: image-representation-classification
author: aaron
---
{% assign imgUrl = "/assets/1.Image-Representation-Classification/Images as Grids of Pixels_files/" | prepend: site.baseurl%}
# Images as Grids of Pixels


```python
import numpy as np
import matplotlib.image as mpimg # for reading in images
import matplotlib.pyplot as plt
import cv2

%matplotlib qt 
```


```python
# Read in the image
imgUrl = "../../assets/1.Image-Representation-Classification/"
image = mpimg.imread(imgUrl + "tesla-roadster.jpg")

# Print out the image dimensions
print("Image dimensions: ", image.shape)
```

    Image dimensions:  (750, 1500, 3)



```python
# Change from color to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_image, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x7f1b3a0cae80>



<img src="{{ "Images%20as%20Grids%20of%20Pixels_3_1.png" | prepend: imgUrl }}" class="md_image"/>



```python
# Print specific grayscale pixel values

# max_val = np.amax(gray_image)
# min_val = np.amin(gray_image)
x = 190
y = 375
pixel_val = gray_image[y, x]
print(pixel_val)
```

    101


#### RGB Channels


```python
# Isolate RGB channels
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

# Visualize the individual color channels
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('R channel')
ax1.imshow(r, cmap='gray')
ax2.set_title('G channel')
ax2.imshow(g, cmap='gray')
ax3.set_title('B channel')
ax3.imshow(b, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x7f1b381fc1d0>



<img src="{{ "Images%20as%20Grids%20of%20Pixels_6_1.png" | prepend: imgUrl }}" class="md_image"/>