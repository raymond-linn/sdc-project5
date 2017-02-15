"""
main file of the project
"""
import numpy as np
import cv2
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from p5Utilities import *


# main pipe line
# params: img - input image frame
def process_img(img):
	# finding the cars on the image
    heat_map = find_cars(img, 1.5)
    # applying the treshold
    heat_map = apply_threshold(heat_map, 1)
    # using label () from scipy.ndimag.measurements
    labels = label(heat_map)
    # draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    return draw_img


# 1) read and list the training cars, notcars images
carsdir = 'vehicles/'
notcarsdir = 'non-vehicles/'
cars, notcars = read_n_list_data(carsdir, notcarsdir)
print('Number of vehicle images found: ', len(cars))
print('Number of non vehicle images found: ', len(notcars))

# 2) train and save the classifier for later use
# Define featutre parameters
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
test_size = 0.1

# should separate the training and predicting in two different processes
clf, scaler = train_vehicle_detection_classifier(cars, notcars, 
                                                color_space=color_space, orient=orient, 
                                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                                hog_channel=hog_channel, spatial_size=spatial_size,
                                                hist_bins=hist_bins, spatial_feat=spatial_feat, 
                                                hist_feat=hist_feat, hog_feat=hog_feat, 
                                                test_size=test_size)

# 3) store the classifier in pickle file which will be used later
joblib.dump(clf, 'classifer.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('saved classifier and scaler')

# 4) generate videos
# video read in and process frame by frame
project_output = 'project_video_out1.mp4'
clip = VideoFileClip("project_video.mp4")
test_clip = clip.fl_image(process_img)
test_clip.write_videofile(project_output, audio=False)

print('done generating video output')
