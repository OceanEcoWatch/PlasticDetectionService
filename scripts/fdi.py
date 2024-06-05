import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Define the band indices (1-based index, so subtract 1 for 0-based indexing)
BAND_GREEN = 2  # Band 3 in the 13-band TIFF file
BAND_NIR = 7  # Band 8A in the 13-band TIFF file
BAND_SWIR1 = 10  # Band 11 in the 13-band TIFF file

# Define central wavelengths (in nm)
lambda_green = 560
lambda_nir = 865
lambda_swir1 = 1610
file = "/Users/marc.leerink/Downloads/S2B_MSIL2A_20231103T021839_N0509_R003_T51PTS_20231103T052521.tif"
# Open the multi-band TIFF file
with rasterio.open(file) as src:
    green = src.read(BAND_GREEN + 1).astype("float32")  # Band 3
    nir = src.read(BAND_NIR + 1).astype("float32")  # Band 8A
    swir1 = src.read(BAND_SWIR1 + 1).astype("float32")  # Band 11
    profile = src.profile

# Calculate FDI
fdi = green - (
    nir + (swir1 - nir) * (lambda_green - lambda_nir) / (lambda_swir1 - lambda_nir)
)

# Read all bands into an array
with rasterio.open(file) as src:
    all_bands = src.read()

# Append the FDI as the new band
all_bands = np.concatenate((all_bands, fdi[np.newaxis, :, :]), axis=0)

# Update the profile to reflect the new number of bands
profile.update(count=all_bands.shape[0])

# Write the new multi-band file with the FDI included as the last band
with rasterio.open("path_to_new_multiband_tif_with_FDI.tif", "w", **profile) as dst:
    dst.write(all_bands)

# Plot the FDI using matplotlib
plt.imshow(
    fdi, cmap="Spectral", vmin=np.percentile(fdi, 2), vmax=np.percentile(fdi, 98)
)
plt.colorbar(label="Floating Debris Index")
plt.title("Floating Debris Index (FDI)")
plt.show()
