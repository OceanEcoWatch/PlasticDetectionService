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

    images = stream.stream_in_image(
        config=config.config,
        bbox=bbox,
        time_interval=("2023-08-01", "2023-08-05"),
        evalscript=evalscripts.EVALSCRIPT_L2A_ALL,
        data_collection=DataCollection.SENTINEL2_L2A,
    )
    print(len(images))
    plt.imshow(images[0][:, :, [3, 2, 1]] / 10000)
    plt.show()


if __name__ == "__main__":
    main()
