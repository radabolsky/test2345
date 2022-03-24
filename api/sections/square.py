from pathlib import Path
import logging
import io
from uuid import uuid4

import numpy as np
from flask import Blueprint, render_template
from flask import request, jsonify, Response, redirect, url_for, send_file

from utils.color_definition import define_color_in_image
from configs.config import AVAILABLE_PICTURE_TYPES, STORAGE_PATH

logger = logging.getLogger("BP picture")

color_definition_bp = Blueprint(
    "define_color",
    __name__,
    url_prefix="/api/v1/color_definition"
)


# @color_definition_bp.route("/")
# def index():
#     return render_template("squaremain.html")


# @color_definition_bp.route("/image/<image_id>")
# def define_color_on_image(image_id):
#     print("ID:", image_id, flush=True)
#     return render_template("image_colors_page.html")


@color_definition_bp.route("/redirect", methods=['POST'])
def picture_to_color_defenition():
    current_process = uuid4()
    picture_info = request.form
    picture = request.files

    picture = picture["picture"]
    size = picture_info.get("size")
    color = picture_info.get("color")

    if not any(
            [
                picture.filename.endswith(type_) for type_ in AVAILABLE_PICTURE_TYPES
            ]
    ):
        return Response("Unsupported extension for file", status=400)
    numpy_img = np.fromstring(picture.read(), np.uint8)
    red_picture, blue_picture, ratio = define_color_in_image(numpy_img, current_process, color)

    return jsonify({"redirect": f"/image/b8d2eec0-ad33-4f5a-8311-d24b9990c54b"})


@color_definition_bp.route("/colors_results")
def image_with_colors_page():
    image_id = request.args.get("image_id")
    if not image_id:
        return Response("No images ID was provided", status=400)
    filename = f"{image_id}.txt"
    return send_file(STORAGE_PATH / filename, as_attachment=True)
