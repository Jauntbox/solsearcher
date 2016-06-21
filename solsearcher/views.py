from flask import render_template, request, jsonify
from solsearcher import app
import json
import time
import requests
import cv2
from scipy import ndimage, misc, stats
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.externals import joblib
from skimage import data, io
from StringIO import StringIO
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
from os import listdir
import subprocess
import uuid

from solarize import split_images, dbscan_features, dbscan_features_new
from api_keys import google_maps_api_key
#TODO: Hook in code from solarize.py to analyze downloaded image and add a rectangle to the map

#user = database_settings.username #add your username here (same as previous postgreSQL) 
#pswd = database_settings.pswd                     
#host = 'localhost'
#dbname = 'birth_db'
#db = create_engine('postgresql://%s:%s@%s/%s'%(user,pswd,host,dbname)) #create_engine('postgres://%s:%s@%s/%s'%(user,pswd,host,dbname))
#con = None
#con = psycopg2.connect(database = dbname, user = user, password = pswd)

'''
Returns a 1280x800 pixel color image (3 channels) centered at lat, long
'''
def acquire_image(lat, long):
	my_params = {
	    'maptype': 'satellite',
	    'center': str(lat)+','+str(long),
	    'zoom': '20',
	    'size': '640x400',
	    'scale': '2',
	    'key': google_maps_api_key
	}
	req = requests.get('https://maps.googleapis.com/maps/api/staticmap?',params=my_params)
	print req
	image_array = io.imread(StringIO(req.content))
	return image_array

center_coords = [0,0]
file_name = ""

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Kevin' }, api_key = google_maps_api_key,
       )

@app.route('/output_maps')
def output_maps():
	#pull 'birth_month' from input field and store it
	coords = request.args.get('coords')
	print coords
	return render_template("output_maps.html", api_key = google_maps_api_key)

@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/contact')
def contact():
	return render_template("contact.html")

@app.route('/postmethod_clickFindSolarPanels', methods = ['POST'])
def get_post_clickFindSolarPanels():
	global file_name
	
	print "Inside clickFindSolarPanels (Python)"
	print "request.data:", request.data
	print "request.args:", request.args
	print "request.form:", request.form

	NE_coords = [float(request.form["NE_corner_lat"]), float(request.form["NE_corner_lng"])]
	SW_coords = [float(request.form["SW_corner_lat"]), float(request.form["SW_corner_lng"])]
	#if center_coords == []:
	#	center_coords.append([float(request.form["center_lat"]), float(request.form["center_lng"])])
	#else:
	center_coords[0] = float(request.form["center_lat"])
	center_coords[1] = float(request.form["center_lng"])
	print "NE coords:",NE_coords
	print "SW coords:",SW_coords
	print "Center coords:",center_coords

	#print request.form["hello"]
	#print request.form["center_lat"]
	#print request.form["center_lng"]
	#print request.form["NE_corner_lat"]
	#print request.form["SW_corner_lat"]

	#Latitude difference across entire image
	#delta_lat = NE_coords[0] - SW_coords[0]
	#delta_lng = NE_coords[1] - SW_coords[1] 
	#image_centers = [[center_coords[0] + delta_lat/4.0, center_coords[1] + delta_lng/4.0],
	#				[center_coords[0] + delta_lat/4.0, center_coords[1] - delta_lng/4.0],
	#				[center_coords[0] - delta_lat/4.0, center_coords[1] + delta_lng/4.0],
	#				[center_coords[0] - delta_lat/4.0, center_coords[1] - delta_lng/4.0]]
	#print image_centers

	#for i in range(len(image_centers)):
	#	img = acquire_image(*image_centers[i])
	#	misc.imsave('test_img_'+str(i)+'.png', img)

	print "Acquiring image"
	img = acquire_image(center_coords[0], center_coords[1])
	time.sleep(1)
	file_name = str(uuid.uuid1())+".png"
	print file_name
	#misc.imsave('test_img.png', img)
	print "Saving image"
	misc.imsave(file_name, img)
	return ""


@app.route('/postmethod_clickShowSolarPanels', methods = ['POST'])
def get_post_clickShowSolarPanels():
	global file_name

	print "Inside clickShowSolarPanels (Python)"
	print "request.data:", request.data
	print "request.args:", request.args
	print "request.form:", request.form

	solar_locs = []
	delta_lat = 0.000428567008235
	delta_lng = 0.000858306884766
	center_lat = float(request.form["center_lat"])
	center_lng = float(request.form["center_lng"])

	dbscan_classifier = joblib.load('DBSCAN_classifier.pkl')
	clf = joblib.load('RandomForest_classifier.pkl')
	print clf

	#file_list = listdir('.')
	#print file_list

	#img_file_list = [fname for fname in file_list if '.png' in fname]
	#img_file_list = ['test_img.png']
	print file_name
	img_file_list = [file_name]
	print img_file_list

	#Go through all the files, split them and find their features, checking for any solar panels:
	threshold = 70
	thresholds = [50,70,90,120] #[60,70,80]
	#thresholds = [70]
	for img_index,img_file in enumerate(img_file_list):
		print "Reading image, ",img_file
		big_img = misc.imread(img_file)
		#print type(img), img.dtype
		#img_list = split_images(big_img)
		#for tile_index,img in enumerate(img_list):
		for thresh in thresholds:
			dbscan_results = dbscan_features_new(big_img, thresh)

			if len(dbscan_results) <= 0:
				print "Skipping because len(dbscan_results)=", len(dbscan_results)
				continue

			features = [res[0:9] for res in dbscan_results]
			if len(features) == 1:
				features = np.array(features).reshape(1, -1)
			else:
				features = np.array(features)

			print "After continue"
			predictions = clf.predict(features)
			solar_indices = np.where(predictions == 1)[0]
			print "Classifier predictions: ",predictions
			print "Success indices: ",solar_indices
			if len(solar_indices) > 0:
				for s_index in solar_indices:
					#Define circles
					x_cent = dbscan_results[s_index][9]
					y_cent = dbscan_results[s_index][10]
					#x_cluster = dbscan_results[j][9]
					#y_cluster = dbscan_results[j][10]
					#n_cluster = dbscan_results[j][11]
					#print "Solar panels found in tile "+str(tile_index)+" of "+img_file+" thresh:"+str(thresh)
					print "Solar panels found!"
					print x_cent, y_cent
					print center_lat, center_lng
					solar_lat = (center_lat + delta_lat/2.0) - (y_cent/800.0)*delta_lat
					solar_lng = (center_lng - delta_lng/2.0) + (x_cent/1280.0)*delta_lng
					print solar_lat
					print solar_lng

					#solar_locs.append({"center_loc":[solar_lat,solar_lng]})
					solar_locs.append([solar_lat,solar_lng])
	
	#return "Hi from Python!"
	#return jsonify([0.0001, 0.0001])
	#return jsonify({"center_loc":[36.9741385,-122.0317605]})

	#Also make sure to delete file that we downloaded so that it doesn't pollute the server
	subprocess.call(['rm', file_name])

	return jsonify({"data":solar_locs})


