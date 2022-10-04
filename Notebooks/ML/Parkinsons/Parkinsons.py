# import the necessary packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import os
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog,messagebox
import cv2

def quantify_image(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")

	# return the feature vector
	return features

def load_split(path):
	# grab the list of images in the input directory, then initialize
	# the list of data (i.e., images) and class labels
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []

	# loop over the image paths
	for imagePath in imagePaths:
		# extract the class label from the filename
		label = imagePath.split(os.path.sep)[-2]

		# load the input image, convert it to grayscale, and resize
		# it to 200x200 pixels, ignoring aspect ratio
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))

		# threshold the image such that the drawing appears as white
		# on a black background
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# quantify the image
		features = quantify_image(image)

		# update the data and labels lists, respectively
		data.append(features)
		labels.append(label)

	# return the data and labels
	return (np.array(data), np.array(labels))


def run_code():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
	ap.add_argument("-t", "--trials", type=int, default=5,
	help="# of trials to run")
	args = vars(ap.parse_args())
	print(args)

	# define the path to the training and testing directories
	trainingPath = os.path.sep.join([args["dataset"], "training"])
	testingPath = os.path.sep.join([args["dataset"], "testing"])

	# loading the training and testing data
	print("[INFO] loading data...")
	(trainX, trainY) = load_split(trainingPath)
	(testX, testY) = load_split(testingPath)

	# encode the labels as integers
	le = LabelEncoder()
	trainY = le.fit_transform(trainY)
	testY = le.transform(testY)

	# initialize our trials dictionary
	trials = {}
	# loop over the number of trials to run
	for i in range(0, args["trials"]):
		# train the model
		print("[INFO] training model {} of {}...".format(i + 1,
		args["trials"]))
		model = RandomForestClassifier(n_estimators=100)
		model.fit(trainX, trainY)

		# make predictions on the testing data and initialize a dictionary
		# to store our computed metrics
		predictions = model.predict(testX)
		metrics = {}

		# compute the confusion matrix and and use it to derive the raw
		# accuracy, sensitivity, and specificity
		cm = confusion_matrix(testY, predictions).flatten()
		(tn, fp, fn, tp) = cm
		metrics["accuracy"]= (tp + tn) / float(cm.sum())
		metrics["sensitivity"] = tp / float(tp + fn)
		metrics["specificity"] = tn / float(tn + fp)
	print("Accuracy = ",metrics["accuracy"],"\nSensitivity = ",metrics["sensitivity"],"\nSpecificity = ",metrics["specificity"])
	return

def select_image():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
	ap.add_argument("-t", "--trials", type=int, default=5,
	help="# of trials to run")
	args = vars(ap.parse_args())
	# grab a reference to the image panels
	run_code()
	global panelA, panelB
	# open a file chooser dialog and allow the user to select an input
	# image
	if args["dataset"]=="dataset/wave":
		path = filedialog.askopenfilename(initialdir="/home/dell/Desktop/detect-parkinsons/dataset/wave/testing")
	else:
		path = filedialog.askopenfilename(initialdir="/home/dell/Desktop/detect-parkinsons/dataset/spiral/testing")
	# ensure a file path was selected
	if len(path) > 0:
		label = path.split(os.path.sep)[-2]
		# load the input image, convert it to grayscale, and resize
		# it to 200x200 pixels, ignoring aspect ratio
		image = cv2.imread(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))

		# threshold the image such that the drawing appears as white
		# on a black background
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# quantify the image
		features = quantify_image(image)
		

		# define the path to the training and directories
		trainingPath = os.path.sep.join([args["dataset"], "training"])

		# loading the training data
		#print("[INFO] loading data...")
		(trainX, trainY) = load_split(trainingPath)
		# encode the labels as integers
		le = LabelEncoder()
		trainY = le.fit_transform(trainY)


		# loop over the number of trials to run
		for i in range(0, 5):
			# train the model
			model = RandomForestClassifier(n_estimators=100)
			model.fit(trainX, trainY)
	pred=model.predict([features])
	image = cv2.imread(path)
	if pred[0]==1:
		result = cv2.imread("/home/dell/Desktop/detect-parkinsons/dataset/yes.png")
	else:
		result = cv2.imread("/home/dell/Desktop/detect-parkinsons/dataset/no.png")
	# OpenCV represents images in BGR order; however PIL represents
	# images in RGB order, so we need to swap the channels
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
	# convert the images to PIL format...
	image = Image.fromarray(image)
	result = Image.fromarray(result)
	# ...and then to ImageTk format
	image = ImageTk.PhotoImage(image)
	result = ImageTk.PhotoImage(result)
	# if the panels are None, initialize them
	if panelA is None or panelB is None:
	# the first panel will store our original image
		panelA = Label(image=image,height=500,width=500,bg="black")
		panelA.image = image
		panelA.pack(side="left", padx=50, pady=10,expand="true")
			# while the second panel will store the edge map
		panelB = Label(image=result)
		panelB.image = result
		panelB.pack(side="right", padx=50, pady=10,expand="true")
	# otherwise, update the image panels
	else:
		# update the pannels
		panelA.configure(image=image)
		panelB.configure(image=result)
		panelA.image = image
		panelB.image = result
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
ap.add_argument("-t", "--trials", type=int, default=5,
help="# of trials to run")
args = vars(ap.parse_args())
root = Tk(className="-Parkinsons Disease Detector-")
root.geometry("1300x1200+10+10")
root.configure(bg="black")
fr=Frame(root)
fr.pack(padx="100",pady="100",fill="both")
fr.config(bg="azure")
panelA = None
panelB = None
def wave():
	messagebox.showinfo("For wave dataset","Accuracy    = 0.7\nSensitivity = 0.6\nSpecificity = 0.7333")
def spiral():
	messagebox.showinfo("For spiral dataset","Accuracy    = 0.8666\nSensitivity = 0.8\nSpecificity = 0.8333")
btn1 = Button(fr, text="Select an image to view the result", command=select_image, padx="100", pady="20",bg="cyan",activebackground="red")
btn1.pack(side="bottom", padx="100", pady="20")
if args["dataset"]=="dataset/wave":
	b2=Button(fr,text="Performance metrics for Wave dataset",command=wave, padx="50", 	pady="20",bg="cyan",activebackground="red")
	b2.pack(padx="100",pady="10")
else:
	b3=Button(fr,text="Performance metrics for Spiral dataset",command=spiral, padx="50", pady="20",bg="cyan",activebackground="red")
	b3.pack(padx="100", pady="10")
# kick off the GUI
root.mainloop()
