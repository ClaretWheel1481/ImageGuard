import flask as flask

ImageGuard = flask.Flask(__name__)

# TODO: 图像检测接口

if __name__ == '__main__':
    # Flask
    ImageGuard.run(
        port=37882,
        debug=False,
        threaded=False,
        host='0.0.0.0'
    )