import numpy as np
import jetson.inference
import jetson.utils

import argparse
import ctypes
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="point", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=175.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# the mask image is half the size
half_width = int(opt.width/2)
half_height = int(opt.height/2)

# allocate the output images for the overlay & mask
img_overlay = jetson.utils.cudaAllocMapped(opt.width * opt.height * 4 * ctypes.sizeof(ctypes.c_float))
img_mask = jetson.utils.cudaAllocMapped(half_width * half_height * 4 * ctypes.sizeof(ctypes.c_float))

# create the camera and display
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()

# work with the raw classification grid dimensions
grid_width, grid_height = net.GetGridSize()	
num_classes = net.GetNumClasses()

# allocate a single-channel uint8 image for the class mask
class_mask = jetson.utils.cudaAllocMapped(width=grid_width, height=grid_height, format="gray8")

# process frames until user exits
while display.IsOpen():
	# capture the image
	img, width, height = camera.CaptureRGBA()

	# process the segmentation network
	net.Process(img, width, height, opt.ignore_class)

	# generate the overlay and mask
	net.Overlay(img_overlay, width, height, opt.filter_mode)
	net.Mask(img_mask, half_width, half_height, opt.filter_mode)
    
    #jetson.utils.cudaDeviceSynchronize()
    #jetson.utils.saveImageRGBA(opt.file_out, img_output, width, height)

    # compute mask statistics
	print('computing class statistics...')

	# get the class mask (each pixel contains the classID for that grid cell)
	net.Mask(class_mask, grid_width, grid_height)

	# view as numpy array (doesn't copy data)
	mask_array = jetson.utils.cudaToNumpy(class_mask)	

	# compute the number of times each class occurs in the mask
	class_histogram, _ = np.histogram(mask_array, num_classes)

	print('grid size:   {:d}x{:d}'.format(grid_width, grid_height))
	print('num classes: {:d}'.format(num_classes))

	print('-----------------------------------------')
	print(' ID  class name        count     %')
	print('-----------------------------------------')

	for n in range(num_classes):
		percentage = float(class_histogram[n]) / float(grid_width * grid_height)
		print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, net.GetClassDesc(n), class_histogram[n], percentage)) 

	# render the images
	display.BeginRender()
	display.Render(img_overlay, width, height)
	display.Render(img_mask, half_width, half_height, width)
	display.EndRender()

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
