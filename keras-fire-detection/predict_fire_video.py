

# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch import config
from collections import deque
from imutils import paths
from imutils.video import FPS
import numpy as np
import imutils
import random
import pickle
import argparse
import cv2
import os



ap = argparse.ArgumentParser()
ap.add_argument('-s', '--size', type=int, default=128,
                help="size of queue for averaging")
ap.add_argument("-i", "--input", required=True,
                help="path to our input video")
ap.add_argument("-o", "--output", required=True,
                help="path to our output video")

args = vars(ap.parse_args())

# load the trained model from disk
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# grab the paths to the fire and non-fire images, respectively
print("[INFO] predicting...")
#firePaths = list(paths.list_images(config.FIRE_PATH))
#nonFirePaths = list(paths.list_images(config.NON_FIRE_PATH))
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])


# combine the two image path lists, randomly shuffle them, and sample
# them
#imagePaths = firePaths + nonFirePaths

#random.shuffle(imagePaths)
#imagePaths = imagePaths[:config.SAMPLE_SIZE]

# loop over the sampled image paths
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
fps = FPS().start()
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (W, H) = frame.shape[:2]
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (128, 128)).astype("float32")
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = config.CLASSES[i]
    text = label if label == "Non-Fire" else "WARNING! Fire!"
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.25, (0, 255, 0), 5)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
    writer.write(output)
    cv2.imshow("Output", output)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#release the file pointer
print("[INFO] cleaning up...")
writer.release()
vs.release()

