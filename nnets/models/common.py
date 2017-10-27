import os
import configparser

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_FILE = os.path.join(ROOT_DIR, 'config.conf')
CONFIG = configparser.ConfigParser()
CONFIG.read(CONFIG_FILE)

class ImageRecord:

    def __init__(self, height, width=None, depth=3, float32image=None, label=None, key=None):
        self.height = height
        self.width = width if width else height # use height as width if image is square
        self.depth = depth
        self.float32image = float32image
        self.label = label
        self.key = key
