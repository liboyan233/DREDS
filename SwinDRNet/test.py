import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt

import OpenEXR
import numpy as np
import cv2

# Path to the EXR file
exr_file = "F:/Lab_liboyan/new/DREDS/Dataset/sim_data/00000/0000_mask.exr"

# Open the EXR file
exr = OpenEXR.InputFile(exr_file)

# Get image metadata
header = exr.header()
dw = header['dataWindow']
width = dw.max.x - dw.min.x + 1
height = dw.max.y - dw.min.y + 1

# Read the RGB channels from the EXR file
channels = ["R", "G", "B"]
data = {}
for channel in channels:
    data[channel] = np.frombuffer(exr.channel(channel), dtype=np.float32)

# Combine the channels into an RGB image
rgb_image = np.dstack((data["R"].reshape(height, width), 
                       data["G"].reshape(height, width), 
                       data["B"].reshape(height, width)))

# Convert the image to uint8 format
rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

# Visualize the image using OpenCV
cv2.imshow("EXR Image", rgb_image_uint8)
cv2.waitKey(0)
cv2.destroyAllWindows()
