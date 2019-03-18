from keras import *
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import sys
sys.path.insert(0, './drive/My Drive/DL/Face Recognition')
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from matplotlib.pyplot import *
import scipy
from inception_blocks_v2 import *
from PIL.Image import ANTIALIAS
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


np.set_printoptions(threshold=np.nan)

def take_photo(filename, quality):
		js = Javascript('''
			async function takePhoto(quality) {
				const div = document.createElement('div');
				const capture = document.createElement('button');
				capture.textContent = 'Capture';
				div.appendChild(capture);

				const video = document.createElement('video');
				video.style.display = 'block';
				const stream = await navigator.mediaDevices.getUserMedia({video: true});

				document.body.appendChild(div);
				div.appendChild(video);
				video.srcObject = stream;
				await video.play();

				// Resize the output to fit the video element.
				google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

				// Wait for Capture to be clicked.
				await new Promise((resolve) => capture.onclick = resolve);

				const canvas = document.createElement('canvas');
				canvas.width = video.videoWidth;
				canvas.height = video.videoHeight;
				canvas.getContext('2d').drawImage(video, 0, 0);
				stream.getVideoTracks()[0].stop();
				div.remove();
				return canvas.toDataURL('image/jpeg', quality);
			}
			''')
		display(js)
		data = eval_js('takePhoto({})'.format(quality))
		binary = b64decode(data.split(',')[1])
		with open(filename, 'wb') as f:
			f.write(binary)
		return filename

class FRModel:
	def build(weights_path='drive/My Drive/DL/Face Recognition/weights'):
		model = faceRecoModel(input_shape=(3, 96, 96))
		model.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
		load_weights_from_FaceNet(model,weights_path)
		return model
	
	# Preprocess image for fit in algorithm
	def preprocess_image(image_path):
		"""
		@param image_path: The path to the image to edit
		"""
		captured_image = cv2.imread(image_path)
		if not ((captured_image.shape[0],captured_image.shape[1]) == (96,96)):
			detected_face,_,coord = detect_faces(captured_image)
			if (len(detected_face),len(coord)) == (0,0):
				return False
			area = [coord[0], coord[1], coord[0]+coord[2], coord[1]+coord[3]]
			cropped = (crop(image_path, area)).resize((96,96), ANTIALIAS)
			scipy.misc.imsave(image_path,cropped)
		return cv2.imread(image_path)
	
	def print_database_items(database):
		for key, db_enc in database.items():
			print(key,len(db_enc))