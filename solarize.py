import cv2
from scipy import ndimage, misc, stats
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.externals import joblib
from skimage import data, io, filters, segmentation, color, measure
from skimage.feature import canny, corner_harris
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.future import graph
from skimage.exposure import histogram, equalize_hist
from StringIO import StringIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from os import listdir

'''
Decomposes a single zoom-level 19 image from Google Earth with the bottom watermark cut out
into a set of 20 images with shape (190,256).

Input: single shape (800, 1280, 3) image downloadad from Google Earth (which will be cropped to
have shape (760, 1280, 3))

Output: list of 20 equally sized grayscale arrays with shape (190, 256). Right now just channel 0
'''
def split_images(big_img):
    img_list = []
    img_copy = np.copy(big_img[0:760,:,:])
    print img_copy.shape, img_copy.shape[1]/5.0, img_copy.shape[0]/4.0

    for h_array in np.split(img_copy,4,axis=0):
        for hv_array in np.split(h_array,5,axis=1):
            img_list.append(hv_array[:,:,0])

    return img_list


'''
Finds a few simple features from looking for clusters of rectangles 

Input:
    img: 2D image array (any size)
    thresholds: array of minimum values of pixels to keep (set to 255) before finding contours
    
    
Output:
    Array of mean area, area variance, and cluster membership count for top 4 clusters by size
    
'''
def dbscan_features(img, threshold):
    features = []
    dbscan_classifier = joblib.load('DBSCAN_classifier.pkl')

    ret,thresh = cv2.threshold(img,threshold,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rect_areas = []
    rect_xs = []
    rect_ys = []
    rect_angles = []
    cluster_inputs = []
    for cnt in contours:
        #Plot a rectangle rotated to have min. area
        rect = cv2.minAreaRect(cnt)
        if((rect[1][0] > 0) and (rect[1][1] > 0) and (rect[2] != -45) and (rect[2] != -90) and (rect[2] != 0)):
            cluster_inputs.append([rect[1][0]*rect[1][1], rect[2], rect[0][0], rect[0][1]])
    
    #print "len(cluster_inputs): ", len(cluster_inputs)
    if len(cluster_inputs) == 0:
        return []

    #print cluster_inputs[0]
    cluster_inputs = np.array(cluster_inputs)
    dbscan_classifier.fit(cluster_inputs)
    n_clusters_dbscan = len(set(dbscan_classifier.labels_)) - (1 if -1 in dbscan_classifier.labels_ else 0)
    
    #print "clusters: ", n_clusters_dbscan
    if n_clusters_dbscan < 1:
        return []

    #Size of each cluster:
    cluster_size = np.zeros(n_clusters_dbscan)
    #Centroid of each cluster:
    centroids = np.zeros((n_clusters_dbscan,len(cluster_inputs[0])))
    #Overall inertia of each cluster
    inertias = np.zeros(n_clusters_dbscan) #[0]*n_clusters
    #Inertia of each cluster along each original data dimension
    inertias_by_type = np.zeros((n_clusters_dbscan,len(cluster_inputs[0])))

    #First, compute centroids of each cluster:
    for index,item in enumerate(cluster_inputs):
        if dbscan_classifier.labels_[index] > -1:
            centroids[dbscan_classifier.labels_[index]] += item
            cluster_size[dbscan_classifier.labels_[index]] += 1
        #print centroids
    centroids = [centroids[i]/cluster_size[i] for i in range(n_clusters_dbscan)]
    #print centroids

    #Now compute variances of each cluster:
    for index,item in enumerate(cluster_inputs):
        if dbscan_classifier.labels_[index] > -1:
            inertias_by_type[dbscan_classifier.labels_[index]] += \
            (item - centroids[dbscan_classifier.labels_[index]])**2
    inertias_by_type = [inertias_by_type[i]/cluster_size[i] for i in range(n_clusters_dbscan)] 
    stdev_by_type = np.sqrt(inertias_by_type)
    
    for i in range(n_clusters_dbscan):
        #features = [centroids[0], stdev_by_type[0], stdev_by_type[1], stdev_by_type[2], stdev_by_type[3]]
        features.append([centroids[i][0], stdev_by_type[i][0], stdev_by_type[i][1], stdev_by_type[i][2], stdev_by_type[i][3]])
    
    return features


'''
Finds a few simple features from looking for clusters of rectangles 

Input:
    img: 2D image array (any size)
    thresholds: array of minimum values of pixels to keep (set to 255) before finding contours
    
    
Output:
    Array of mean area, area variance, and cluster membership count for top 4 clusters by size
    
'''
def dbscan_features_new(color_img, threshold):
    features = []
    color_scale_factor = 50
    dbscan_classifier = joblib.load('DBSCAN_classifier.pkl')

    single_channel_img = color_img[:,:,0]
    ret,thresh = cv2.threshold(single_channel_img,threshold,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rect_areas = []
    rect_xs = []
    rect_ys = []
    rect_angles = []
    cluster_inputs = []
    for cnt in contours:
        #Plot a rectangle rotated to have min. area
        rect = cv2.minAreaRect(cnt)
        if((rect[1][0] > 0) and (rect[1][1] > 0) and (rect[2] != -45) and (rect[2] != -90) and (rect[2] != 0)):
            colors = color_img[int(rect[0][1]),int(rect[0][0])]
            #avg_colors = color_img[int(rect[0][1]),int(rect[0][0]) + 1]
            avg_colors = np.array(color_img[int(rect[0][1]),int(rect[0][0])]).astype(np.uint16)
            avg_colors += np.array(color_img[int(rect[0][1]) + 1,int(rect[0][0])]).astype(np.uint16)
            avg_colors += np.array(color_img[int(rect[0][1]) - 1,int(rect[0][0])]).astype(np.uint16)
            avg_colors += np.array(color_img[int(rect[0][1]),int(rect[0][0]) + 1]).astype(np.uint16)
            avg_colors += np.array(color_img[int(rect[0][1]),int(rect[0][0]) - 1]).astype(np.uint16)
            avg_colors = avg_colors/5.0

            #print "Color info at central pixel: ",colors[0],colors[1],colors[2]
            #print "Color average around central pixel: ", avg_colors
            cluster_inputs.append([rect[1][0]*rect[1][1], rect[2], rect[0][0], rect[0][1], avg_colors[0], avg_colors[1], avg_colors[2]]) 
    
    print "len(cluster_inputs): ", len(cluster_inputs)
    if len(cluster_inputs) == 0:
        return []

    cluster_inputs = np.array(cluster_inputs)
    dbscan_classifier.fit(cluster_inputs)
    n_clusters_dbscan = len(set(dbscan_classifier.labels_)) - (1 if -1 in dbscan_classifier.labels_ else 0)

    print "clusters: ", n_clusters_dbscan
    if n_clusters_dbscan < 1:
        return []

    #Size of each cluster:
    cluster_size = np.zeros(n_clusters_dbscan)
    #Centroid of each cluster:
    centroids = np.zeros((n_clusters_dbscan,len(cluster_inputs[0])))
    #Overall inertia of each cluster
    inertias = np.zeros(n_clusters_dbscan) #[0]*n_clusters
    #Inertia of each cluster along each original data dimension
    inertias_by_type = np.zeros((n_clusters_dbscan,len(cluster_inputs[0])))

    #First, compute centroids of each cluster:
    for index,item in enumerate(cluster_inputs):
        if dbscan_classifier.labels_[index] > -1:
            centroids[dbscan_classifier.labels_[index]] += item
            cluster_size[dbscan_classifier.labels_[index]] += 1
        #print centroids
    centroids = [centroids[i]/cluster_size[i] for i in range(n_clusters_dbscan)]
    #print centroids

    #Now compute variances of each cluster:
    for index,item in enumerate(cluster_inputs):
        if dbscan_classifier.labels_[index] > -1:
            inertias_by_type[dbscan_classifier.labels_[index]] += \
            (item - centroids[dbscan_classifier.labels_[index]])**2
    inertias_by_type = [inertias_by_type[i]/cluster_size[i] for i in range(n_clusters_dbscan)] 
    stdev_by_type = np.sqrt(inertias_by_type)
    
    for i in range(n_clusters_dbscan):
        features.append([centroids[i][0], np.mean(centroids[i][4:7]), color_scale_factor*centroids[i][4]/np.mean(centroids[i][4:7]), 
                         color_scale_factor*centroids[i][5]/np.mean(centroids[i][4:7]),
                         color_scale_factor*centroids[i][6]/np.mean(centroids[i][4:7]), stdev_by_type[i][0], stdev_by_type[i][1], stdev_by_type[i][2], 
                         stdev_by_type[i][3], centroids[i][2], centroids[i][3], cluster_size[i]])
     
    return features

#dbscan_classifier = DBSCAN(eps=1.50, min_samples=3) #For use with feature scaling
#dbscan_classifier = DBSCAN(eps=0.15, min_samples=3) #For use without feature scaling (but with normalization)
#dbscan_classifier = DBSCAN(eps=50, min_samples=3) #No feature scaling or normalization
#clf = joblib.load('logistic_regression_classifier.pkl')
#print clf

#file_list = listdir('.')
#print file_list

#img_file_list = [fname for fname in file_list if '.png' in fname]
#print img_file_list

#Go through all the files, split them and find their features, checking for any solar panels:
#features_array = []
#threshold = 70
#for img_file in img_file_list:
#	big_img = misc.imread(img_file)
	#print type(img), img.dtype
#	img_list = split_images(big_img)
#	for img_index,img in enumerate(img_list):
#		current_features = dbscan_features(img, threshold)
		#print current_features
#		if current_features != []:
#			prediction = clf.predict(current_features)
			#print prediction
#			if 1 in prediction:
#				print "Solar panels found in tile "+str(img_index)+" of "+img_file
#		features_array.append(current_features)

