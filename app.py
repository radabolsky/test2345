import os
import logging

from flask import Flask
from gevent.pywsgi import WSGIServer

from api.sections.square import color_definition_bp
from api.sections.main_sections import index, define_color_on_image, get_image_by_id

log = logging.getLogger("syngen")


def main():
    app = Flask("TDM-SYNGEN")
    app.register_blueprint(color_definition_bp)
    app.add_url_rule("/", view_func=index)
    app.add_url_rule("/image/<image_id>", view_func=define_color_on_image)
    app.add_url_rule("/image/get_image/<filename>", view_func=get_image_by_id)

    try:
        app.run(
            host="0.0.0.0",
            port=9090,
            debug=True
        )

        # TODO: Uncomment when it will be a production code
        # http_server = WSGIServer(("", int(os.environ.get("LISTEN_PORT", "5000"))), app)
        # http_server.serve_forever()

    finally:
        log.info("WSGI server stopped")


if __name__ == '__main__':
    main()


