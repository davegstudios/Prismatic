#####
# Copyright David Guezuraga 2022
# Prismatic 1.2 by studio daveg
# Code for the Kemper Museum of Contemporary Art 2022 Gala theme "Prismatic"
# 1.0 Original release
# 1.1 Added FPS

import time, os, sys
from PIL import Image, ImageFont, ImageDraw


# OpenCV
import  cv2
import numpy as np

# Choose your outputs!
local_output = 1
local_original_output = 0  #show original non-modified image
neopixel_output = 0
local_original_output_fps = 1  # show fps on original frame only
local_putput_fps_to_console = 1 # show fps on console
#640x480, 1280x720,960x544,800x448,640x360,424x240,352x288,320x240,800x600,176x144,160x120,1280x800
frame_width = 800
frame_height = 448

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
output_scale_percentage = 2


# Percentage to decimate input if you're not able to get a low enough res from the webcam
#  If you set it to 0, then it'll use webcam native resolution
decimate_input_percentage = 1
# Change color map every 5 seconds
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

def imgtoAscii(image, fontSize):
    # In case, user wants to change the font-size of the ASCII image, this variable should be changed.
    # WARNING: Do not set font-size < 5, else the program may get stuck in infinite loop.

    if fontSize < 5:
        print("WARNING: Font-Size is too small. This may result in infinite-loop. Enter any key to continue.")
        _ = input()

    #CHAR_MAP = [temp for temp in reversed("""$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|(1{[?-_+~<i!lI;:,"^`'. """)]
    #CHAR_MAP = [temp for temp in reversed("""ahkbdpqwmZO0QLCJUYXzcvunxr.............""")]
    CHAR_MAP = [temp for temp in reversed("""qwertyuioplkjhgfdsazxcv.................""")]
    # The characters in the above string are sorted according to the amount of white they cover in mono-space fonts.
    # Monospace fonts are those fonts which use the same width no matter the character printed. They are convenient to use for ASCII-art.
    # There are some versions that have minor differences, I am using the one from : http://paulbourke.net/dataformats/asciiart/

    # We are using the mono-space fonts called "secret-code" from Matthew Welch (https://squaregear.net/fonts/)
    # I have added a copy of the font and liscence files in this repository, these fonts are published under MIT liscence.
    FONT_PATH = os.path.join("secret_code", "secrcode.ttf")


    # Let's read the input image and get its dimensions
    image_height, image_width, _ = image.shape

    # Now we load the font file and get the height and width for each font
    font = ImageFont.truetype("secrcode.ttf", fontSize)
    font_width, font_height = font.getsize(".")

    # We convert the BGR image to HSV and normalize Saturation and Value channels before converting image back to RGB.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # uncomment the below lines in case you want to apply Hist-equalization before the ASCIIfication as well.
    hsv_image[..., 1] = cv2.equalizeHist(hsv_image[..., 1])
    hsv_image[..., 2] = cv2.equalizeHist(hsv_image[..., 2])

    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Output blank canvas
    output = np.zeros_like(image)

    pillow_output = Image.fromarray(output)
    pillow_drawer = ImageDraw.Draw(pillow_output)
    for i in range(int(image_height / font_height)):
        for j in range(int(image_width / font_width)):
            # finding the bounding box for the next character to be inserted
            y_start = i * font_height
            x_start = j * font_width

            x_end = x_start + font_width
            y_end = y_start + font_height

            # deciding the next character to be inserted
            i1 = np.mean(hsv_image[y_start:y_end, x_start:x_end, 1])
            i2 = np.mean(hsv_image[y_start:y_end, x_start:x_end, 2])
            intensity = (i1 + i2) / 2

            position = int(intensity * len(CHAR_MAP) / 360)


            # deciding the color of the next character
            #color = np.mean(rgb_image[y_start:y_end, x_start:x_end], axis=(0, 1)).astype(np.uint8)
            # uncomment the below line if you want same color for each character
            color = (255, 255, 255)

            # inserting the next character
            pillow_drawer.text((x_start, y_start), str(CHAR_MAP[position]), font=font, fill=tuple(color))
            # uncomment the below line if you want same character and only vary its color
            #pillow_drawer.text((x_start, y_start), "*", font = font, fill=tuple(color))

    #output = np.array(pillow_output)

    # Performing Histogram Equalization on S and V channels of the output to improve sharpness.
    #output_hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    #output_hsv[..., 1] = cv2.equalizeHist(output_hsv[..., 1])
    #output_hsv[..., 2] = cv2.equalizeHist(output_hsv[..., 2])

    #output_bgr = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2BGR)

    # Converting back to OpenCV
    # Return the frame instead of outputting
    return (np.array(pillow_output))
    # Show Output on screen and write it to a file
    #cv2.imshow(OUTPUT_WINDOW_NAME, output)
    #cv2.imwrite(OUTPUT_PATH, output_bgr)

    # Wait till Key-press for exit
    #cv2.waitKey(0)


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

    if (decimate_input_percentage != 1):
        frame = cv2.resize(frame, (int(new_w),int(new_h)))

    # Flip image so it mirrors the user
    frame = cv2.flip(frame, 1)


    # Reduce Colors
    #result = kmeans_color_quantization(frame, clusters=2, rounds = 1)

    #if it's been color_freq change the color to the next pallet
    if (time.time() - start > color_freq):
        if (color_current + 1 > color_end):
            color_current = color_start
        else:
            color_current += 1
        start = time.time()

    #result = cv2.applyColorMap(result, color_current)
    #print ("Current color = ",color_current)
    result = imgtoAscii(frame,8)



    if(local_putput_fps_to_console == 1):
        print ("FPS = ",fps)
    # Display on the local screen
    if(local_output == 1):
    # Add original Frame to the output window
        if(local_original_output == 1):
            winout = np.concatenate((result,frame),axis = 1)
            scale_percentage = output_scale_percentage
            new_h = height * scale_percentage
            new_w = width * scale_percentage * 2
            winout = cv2.resize(winout, (int(new_w),int(new_h)))
            # Output frame rate up top if selected
            if(local_original_output_fps == 1):
                cv2.putText(winout, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            # Display modified and original image
            cv2.imshow('localPrismatic by daveg', winout)
        else:
            scale_percentage = output_scale_percentage
            new_h = height * scale_percentage
            new_w = width * scale_percentage
            result = cv2.resize(result, (int(new_w),int(new_h)))
            # Display modified image only
            cv2.imshow('outputPrismatic by daveg', result)




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
        if(neopixel_output == 1):
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
