from flask import render_template, send_file

from configs.config import STORAGE_PATH


def index():
    return render_template("squaremain.html")


def define_color_on_image(image_id):
    config = {
        "originimage": f"get_image/origin_{image_id}.jpg",
        "redimage": f"get_image/red_picture_{image_id}.jpg",
        "blueimage": f"get_image/blue_picture_{image_id}.jpg",
        "id_": image_id,
    }
    return render_template("image_colors_page.html", config=config)


def get_image_by_id(filename):
    img = STORAGE_PATH / filename
    return send_file(img)
