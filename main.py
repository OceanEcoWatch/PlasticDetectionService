import matplotlib.pyplot as plt
from sentinelhub import CRS, BBox, DataCollection

from plastic_detection_service import config, evalscripts, stream


def main():
    # manilla bay
    bbox = BBox(
        bbox=(
            120.53058253709094,
            14.384463071206468,
            120.99038315968619,
            14.812423505754381,
        ),
        crs=CRS.WGS84,
    )

    images = stream.stream_in_images(
        config=config.config,
        bbox=bbox,
        time_interval=("2023-08-01", "2023-08-15"),
        maxcc=0.8,
        evalscript=evalscripts.EVALSCRIPT_L2A_ALL,
        data_collection=DataCollection.SENTINEL2_L2A,
    )
    print(len(images))
    for band in range(0, 12):
        print(band)
        plt.imshow(images[0][:, :, band])
        plt.show()


if __name__ == "__main__":
    main()
