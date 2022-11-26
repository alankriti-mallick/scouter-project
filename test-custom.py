from PIL import Image
import numpy as np


image_raw = Image.open('test-images/text.png').resize((260, 260)).convert('L')