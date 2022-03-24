from pathlib import Path

STORAGE_PATH = Path(__file__).parents[1] / "storage"

LOWER_RED_COLOR_BORDERS = (
    (0, 80, 15),
    (150, 80, 15)
)

UPPER_RED_COLOR_BORDERS = (
    (12, 255, 255),
    (180, 255, 255)
)

BLUE_COLOR_BORDERS = (

)

AVAILABLE_COLORS_DEFINITION = {
    "red": (LOWER_RED_COLOR_BORDERS, UPPER_RED_COLOR_BORDERS),
    "blue": ()
}

AVAILABLE_PICTURE_TYPES = (
    ".png",
    ".jpeg",
    ".jpg"
)


