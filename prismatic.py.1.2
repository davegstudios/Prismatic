#####
# Copyright David Guezuraga 2022
# Prismatic 1.2 by studio daveg
# Code for the Kemper Museum of Contemporary Art 2022 Gala theme "Prismatic"
# 1.0 Original release
# 1.1 Added FPS

import time

# OpenCV
import  cv2
import numpy as np

# Choose your outputs!
kmeans_clustering = 1    # enable if you want to do kmeans clustering or just display captured image
local_output = 1
local_original_output = 1  #show original non-modified image
neopixel_output = 0
local_original_output_fps = 1  # show fps on original frame only
local_putput_fps_to_console = 0 # show fps on console
#640x480, 1280x720,960x544,800x448,640x360,424x240,352x288,320x240,800x600,176x144,160x120,1280x800
frame_width = 160
frame_height = 120

# Multiples configuration
#local_color_list = [15,4,5,7]
local_color_list = [15,4,3]
#local_color_list = [15]

# Neopixel
# sudo pip3 install Pillow
# sudo pip3 install rpi_ws281x adafruit-circuitpython-neopixel
# sudo pip3 install adafruit-circuitpython-pixel-framebuf
if (neopixel_output == 1):
    import board
    import neopixel
    from adafruit_pixel_framebuf import PixelFramebuffer
    from PIL import Image
    from PIL import ImageOps
    import board


# If you're using the local display, scale size to value below
output_scale_percentage = 4


# Percentage to decimate input if you're not able to get a low enough res from the webcam
#  If you set it to 0, then it'll use webcam native resolution
decimate_input_percentage = 0
# Change color map every 5 seconds
rotate_color = False   # do we even want to rotate the colors
color_freq = 3   #Number of seconds
color_start = 15   #See OpenCV Documentation
color_end = 15


# NeoPixel Configuration
if (neopixel_output == 1):
    pixel_pin = board.D18
    pixel_width  = 22
    pixel_height = 22
    neopixel_brightness = .01


#############
#  End Configuration
#############

# credit https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
def resize_image(image, width, height,COLOUR=[0,0,0]):
    h, w, layers = image.shape
    if h > height:
        ratio = height/h
        image = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if w > width:
        ratio = width/w
        image = cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)))
    h, w, layers = image.shape
    if h < height and w < width:
        hless = height/h
        wless = width/w
        if(hless < wless):
            image = cv2.resize(image, (int(image.shape[1] * hless), int(image.shape[0] * hless)))
        else:
            image = cv2.resize(image, (int(image.shape[1] * wless), int(image.shape[0] * wless)))
    h, w, layers = image.shape
    if h < height:
        df = height - h
        df /= 2
        image = cv2.copyMakeBorder(image, int(df), int(df), 0, 0, cv2.BORDER_CONSTANT, value=COLOUR)
    if w < width:
        df = width - w
        df /= 2
        image = cv2.copyMakeBorder(image, 0, 0, int(df), int(df), cv2.BORDER_CONSTANT, value=COLOUR)
    image = cv2.resize(image,(1280,720),interpolation=cv2.INTER_AREA)
    return image

# https://stackoverflow.com/questions/5906693/how-to-reduce-the-number-of-colors-in-an-image-with-opencv
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

###########
# Program Start
###########

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")


# Set the frame buffer size so that we are getting only fresh frames
#   Otherwise due to the lag in processing images you can things feel pretty laggy
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

# Set the frame size.  use uvcdynctrl -f to figure out what's available
#  This is a good spot to do the scaling without having to do an additional resize each frame
## Example: W: 160 H: 120 is what was used for prismatic gala / LED display with a decimate_input_percentage of .8
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize ColorMapping variables
current_colorMap = 0
color_current = color_start
start = time.time()


# Initialize neopixels
if (neopixel_output == 1):
    pixels = neopixel.NeoPixel(
        pixel_pin,
        pixel_width * pixel_height,
        brightness=neopixel_brightness,
        auto_write=False,
    )

    pixel_framebuf = PixelFramebuffer(
        pixels,
        pixel_width,
        pixel_height,
        reverse_x=True,
    )

# FPS Configuration
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX
# end FPS Configuration


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    result_count = 0   #Reset result count because there are no current results
    # Calculate Framerate
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = str(round(fps,2))
    # End Framerate Calc

    height, width, layers = frame.shape

    scale_percentage = decimate_input_percentage
    # Resize frame to process faster or decimate unneeded detail
    new_h = height * scale_percentage
    new_w = width * scale_percentage

    if (decimate_input_percentage != 0):
        frame = cv2.resize(frame, (int(new_w),int(new_h)))

    # Flip image so it mirrors the user
    frame = cv2.flip(frame, 1)


    # Reduce Colors
    if (kmeans_clustering == 1): 
        reduced = kmeans_color_quantization(frame, clusters=2, rounds = 1)
        # Denoise
        #reduced = cv2.fastNlMeansDenoisingColored(reduced,None,10,10,7,21)
    else:
        reduced = frame

    #if it's been color_freq change the color to the next pallet
    if (time.time() - start > color_freq):
        if (color_current + 1 > color_end):
            color_current = color_start
        else:
            color_current += 1
        start = time.time()



    if (rotate_color == True):
        result = cv2.applyColorMap(reduced, color_current)
        result_count += 1
    #print ("Current color = ",color_current)
    else:
        # Apply color map(s)
        # first case
        result = cv2.applyColorMap(reduced, local_color_list[0])
        result_count += 1

        if(len(local_color_list) > 1):
                for color in local_color_list[1:]:
                    # Duplicate current 
                    result = cv2.hconcat([result,cv2.applyColorMap(reduced, color)])
                    result_count += 1

    if(local_putput_fps_to_console == 1):
        print ("FPS = ",fps)

    # Display on the local screen
    if(local_output == 1):

        # Add original Frame to the output window
        if(local_original_output == 1):
            result = cv2.hconcat([result,frame])

        new_h = int(result.shape[0]) * output_scale_percentage
        new_w = int(result.shape[1]) * output_scale_percentage
        result = cv2.resize(result, (int(new_w),int(new_h)),interpolation = cv2.INTER_AREA)

        # Output frame rate up top if selected
        if(local_original_output_fps == 1):
            cv2.putText(result, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        # Display the final result
        cv2.imshow('Prismatic by daveg', result)

    # Display via Neopixels
    if(neopixel_output == 1):
        result = cv2.resize(result, (pixel_height, pixel_width))

        # Flip so it is the correct orientation for neopixel framebuffer
        result = cv2.flip(result,1)

        # Convert the image to RGB and display it
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        pixel_framebuf.image(Image.fromarray(result))
        pixel_framebuf.display()

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        if(neopixel_output == 1):  # blank neopixels
            pixels.fill((0, 0, 0))
            pixels.show()
        break

  # Break the loop if capture failed
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
