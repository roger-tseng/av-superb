import matplotlib.pyplot as plt
from cv2 import imread
from skimage import data, exposure
from skimage.feature import hog
from skimage.transform import resize

image = resize(imread("/home/rogert/Pictures/pic.png")[..., ::-1], (128, 128))
print(image.shape)

fd, hog_image = hog(
    image,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=-1,
)
print(fd.shape)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis("off")
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title("Input image")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
print(hog_image.shape)
print(hog_image_rescaled.shape)

ax2.axis("off")
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title("Histogram of Oriented Gradients")
plt.show()
