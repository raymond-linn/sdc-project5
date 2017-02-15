"""
Define utility functions to detect vehciles by using support vector machines
and drawing boxes around the detected vechiles on to streams of video frames
"""
"""
Color features in color space 
1) bin_spatial
2) color_historam
Histogram of gradient (HOG)
3) hog_features
Combination of features and normalization
4) extract_features
Classifying, Training
5) classify
6) train_class
Data Preparation
7) prep_data
False positive detection and filtering
8) heat_map
9) threshold
10) labeling
11) draw_labeled_bboxes
Defining region of interest, size of windows, sliding 
12) get_region_of_interest (y_start = 220, y_end = 720) ie. (0,220), (1280,720)
13) sliding_window
14) start and stop
15) total windows in different scales = 25? with 4 scales 
small_scaled with 10 windows at (440, 400), (990, 455) ie. box_size = 55x55
medium_scaled with 8 windows at (195,370), (1235, 500) ie. box_size = 130x130
large_scaled with 5 windows at (10,310),(1264,560) ie. box_size = 250x250
largest_scaled with 3 windows at (10,218), (1270, 638) ie. box_size = 420x420
16) search_window
"""

import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from sklearn.externals import joblib


# reading in and lisiting the cars and notcars images for training
# params: carsdir - uzipped vehicles directory that from training data provided by Udacity
# notcarsdir - uzipped non-vehicles directory that from training data provided by Udacity
# return - cars and notcars images
def read_n_list_data(carsdir, notcarsdir):    

    # look into vehicles and non-vehicles folder to read in the images 
    # from GTI, Kitti and extra, other sub-folders     
    image_types = os.listdir(carsdir)
    cars = []
    for imtype in image_types:
        cars.extend(glob.glob(carsdir+imtype+'/*'))

    # print('Number of vehicle images found: ', len(cars))
    # write file names of images to cars.txt file 
    with open("cars.txt", 'w') as f:
        for fn in cars:
            f.write(fn+'\n')

    # for non-vehicle images    
    image_types = os.listdir(notcarsdir)
    notcars = []
    for imtype in image_types:
        notcars.extend(glob.glob(notcarsdir+imtype+'/*'))

    # print('Number of non vehicle images found: ', len(notcars))
    with open("notcars.txt", 'w') as f:
        for fn in notcars:
            f.write(fn+'\n')

    return cars, notcars


# get histogram of gradient (HOG) features and visualization
# params: img - input image
# orient: orientation in int - number of oreintation in bins
# pix_per_cell: size (in pixel) of a cell
# cell_per_block - 2 tuple (int, int) - number of celss in each block
# vis - bool - whether return of the HOG image
# feature_vec - bool - return data as feature vector by calling .ravel()
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features


# Color features
# Computing binned color features - spatial binning
# params: img - image input 
# size - resolution to be scaled down
def bin_spatial(img, size=(32,32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    # features = cv2.resize(img, size).ravel()
    # return features
    return np.hstack((color1, color2, color3))


# Computing color histogram features
# params: img - image input
# nbins - number of bins
# bins_range - the range of the bins. depends on how image is read in
# the range needs to be adjusted.
# if png image and read in with mpimg, the range needs to be changed
def color_hist(img, nbins=32): # bins_range=(0, 256) for jpeg
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# extracting the features from single image window
# params: img - image input
# color_space - what color space do you want to extract features from
# spatial_size - the size to be reduced to extract spatial binning
# hist_bins - number bins for color histogram features
# pix_per_cell: size (in pixel) of a cell
# cell_per_block - 2 tuple (int, int) - number of celss in each block 
# hog_channel - 0, 1, 2 or "ALL"
# spatial_feat - True to extract or False not extract spatial binning feature
# hist_feat - True to extract or Fals not to extract color histogram feature
# hog_feat - True to extract or False not to extract hog feature
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=False):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True)) 
            
            hog_features = np.concatenate(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel],
                                                          orient, pix_per_cell, cell_per_block,
                                                          vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], 
                                                orient, pix_per_cell, cell_per_block, 
                                                vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)


# train LinearSVC classifier
# params: test_cars - cars data images
# test_notcars - notcars data images
# color_space - RGB, HSV, HLS, LUV, YCrCb
# orient - orientation of HOG
# pix_per_cell - how many pixels per cell
# cell_per_block - how many cells per a block
# hog_channel - how many channels for the hog feature - 0, 1, 2, "ALL"
# spatial_size - spaitail binning size
# hist_bins - the number of bins for the color historgram
# spatial_feat - boolean - True for including
# hist_feat - boolean - True for including
# hog_feat - boolean - True for including
# test_size - the size to split as the test data
# return - classifier and scaler
def train_vehicle_detection_classifier(test_cars, test_notcars, 
                                       color_space='RGB', orient=9, pix_per_cell=8,
                                       cell_per_block=2, hog_channel=0, spatial_size=(32,32),
                                       hist_bins=32, spatial_feat=True, hist_feat=True,
                                       hog_feat=True, test_size=0.1):
    
    # for the time stamp
    t = time.time()

    car_features = extract_features(test_cars, color_space='YCrCb',
                                    spatial_size= spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel='ALL', spatial_feat= spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(test_notcars, color_space='YCrCb',
                                    spatial_size= spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel='ALL', spatial_feat= spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

    print(time.time()-t, 'Seconds to compute features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector to be 1's for cars and 0's for not cars
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=test_size, random_state=rand_state)

    print('Using: ', orient, 'orientations', pix_per_cell, 'pixels per cell',
          cell_per_block, 'cells per block', hist_bins, 'histogram bins', spatial_size, 'spatial sampling')
    print('Feature vector length: ', len(X_train[0]))

    # Use LinearSVC
    svc = LinearSVC()

    # time stamps
    t = time.time()
    svc.fit(X_train, y_train)
    print(round(time.time()-t, 2), 'Seconds to train SVC...')
    # check the test score
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    
    return svc, X_scaler


# Slide the window of xy_window size with overlapped size 
# of xy_overlap from the img x_start_stop, y_start_stop to 
# the whole img area
# params: img - image that needs to be cut into small windows
# x_start_stop - start and end x coordinates of the image to cut
# y_start_stop - start and end y coordinates of the image to cut
# xy_window - size of small cut window
# xy_overlap - the overlap factor for each small window
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
# searching whether cars are in the small windows that cut 
# with slide_window() function
# params: img - image to be searched through
# windows - list of windows that cut into by slide_window()
# clf - trained classifier
# scaler - scaler that used while in training
# color_space - RGB, HSV, HLS, LUV, YCrCb
# spatial_size - spaitail binning size
# hist_bins - the number of bins for the color historgram
# hist_range - the range for histogram
# orient - orientation of HOG
# pix_per_cell - how many pixels per cell
# cell_per_block - how many cells per a block
# hog_channel - how many channels for the hog feature - 0, 1, 2, "ALL"
# spatial_feat - boolean - True for including
# hist_feat - boolean - True for including
# hog_feat - boolean - True for including 

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Drawing the bounding boxes on the copy of the input image
# params: img - original image
# bboxes - windows that found cars
# color - color to draw boxes
# thick - thickness of the boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# finding the cars from the inout images using hog features on whole image
# params: img - input immage
# scale - scale of the image size
# ystart - croping y coordinate
# ystop - cropping y coordinate
# window - window patch size
# orient - oriantation of hog features
# pix_per_cell - how many pixels per cell
# cell_per_block - how many cells per a block
# return: heatmap of the input image
def find_cars(img, scale=1, ystart=400, ystop=656,  
              window=64, orient=9, pix_per_cell=8, cell_per_block=2):
    
    count = 0
    # make a copy of original image
    draw_img = np.copy(img)
    # initialize the heatmap with zeros
    heatmap = np.zeros_like(img[:,:,0])
    # for jpeg - scale to png scale 0 to 1
    img = img.astype(np.float32)/255 
    
    # region of interest image 
    img_tosearch = img[ystart:ystop,:,:]
    # convert the color to be YCrCb of the region of interest
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)

    # loading the classifier and scaler to predict
    svc = joblib.load('classifer.pkl')
    X_scaler = joblib.load('scaler.pkl')
    
    # if the scale is not 1 then resize
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), 
        								np.int(imshape[0]/scale)))
    
    # color channel
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
    # number cells in the region - x and y blocks
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    # number of features per blocks
    nfeat_per_block = orient * cell_per_block**2
    
    # number of blocks per window
    nblocks_per_window = (window // pix_per_cell) - 1
    
    # cells to step through instead of overlapping
    cells_per_step = 2 
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # compute hog features per channel
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # step trhough x and y steps to get the features, predict using loaded classifier
    # then save the coordinates to heatmap
    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # extract hog feature 
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            # Get Color features
            spatial_features = bin_spatial(subimg, size=(32,32))
            hist_features = color_hist(subimg, nbins=32)
            
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, 
            												hist_features, 
            												hog_features)).reshape(1,-1))
            test_prediction = svc.predict(test_features)
            
            # check if prediction is good then draw boxe on the draw_img and add in heatmap
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), 
                              (xbox_left+window, ytop_draw+win_draw+ystart),
                              (0, 0, 255), 6)
                # add boxes to the deque
                #img_boxes.append(((xbox_left, ytop_draw+ystart),
                #                  (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1
    
    return heatmap

# threshold for heatmap
def apply_threshold(heatmap, threshold):
    # zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # return the thresholded map
    return heatmap

# drawing the bounding boxes on the input image
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


