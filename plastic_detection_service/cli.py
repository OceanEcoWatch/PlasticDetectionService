import click

from plastic_detection_service.constants import (
    LAST_WEEK_TIME_INTERVAL,
    MANILLA_BAY_BBOX,
)
from plastic_detection_service.patches import create_patches


@click.group("plastic_detection_service")
def cli():
    """main entry point."""
    pass


@click.command("create_patches")
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    default=MANILLA_BAY_BBOX,
    help=(
        "Bounding box of the area to download. Format: min_lon min_lat max_lon max_lat"
    ),
)
@click.option(
    "--time_interval",
    nargs=2,
    type=str,
    default=LAST_WEEK_TIME_INTERVAL,
    help="Time interval to download. Format: YYYY-MM-DD YYYY-MM-DD",
)
@click.option(
    "--maxcc",
    type=float,
    default=0.8,
    help="Maximum cloud coverage allowed. Float number from 0.0 to 1.0",
)
@click.option(
    "--output_folder",
    type=str,
    default="eopatches",
    help="Folder where to save downloaded EOPatches",
)
@click.option(
    "--resolution",
    type=int,
    default=10,
    help="Resolution of the data in meters",
)
@click.option(
    "--bbox_size",
    type=int,
    default=5000,
    help="The size of generated bounding boxes in meters",
)
def cli_patches(
    bbox: tuple[float, float, float, float],
    time_interval: tuple[float],
    maxcc: float,
    output_folder: str,
    resolution: int,
    bbox_size: int,
):
    create_patches(bbox, time_interval, maxcc, output_folder, resolution, bbox_size)
