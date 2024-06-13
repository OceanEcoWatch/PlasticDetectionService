import matplotlib.pyplot as plt
import numpy as np
import rasterio

with rasterio.open("pred_raster.tif") as src:
    classification_image = src.read(1)
    # unique pixel values
    print(np.unique(classification_image))


# Define a colormap
cmap = plt.get_cmap(
    "tab20", 15
)  # 'tab20' has 20 distinct colors, we'll use the first 15

# Plot the image
plt.imshow(classification_image, cmap=cmap, vmin=1, vmax=15)

# Add a color bar
cbar = plt.colorbar(ticks=np.arange(1, 16))
cbar.set_label("Class Value")

# Display the plot
plt.title("Classification Image")
plt.show()
