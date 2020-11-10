# USAGE
# python predict_fire.py

# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

# load the trained model from disk
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# grab the paths to the fire and non-fire images, respectively
print("[INFO] predicting...")
firePaths = list(paths.list_images(config.FIRE_PATH))
nonFirePaths = list(paths.list_images(config.NON_FIRE_PATH))

# combine the two image path lists, randomly shuffle them, and sample
# them

firePaths = firePaths[:250]
nonFirePaths = nonFirePaths[:250]
# print(firePaths[np.random.shuffle(sort)])
# imagePaths = firePaths + nonFirePaths
# random.shuffle(imagePaths)
# imagePaths = imagePaths[:config.SAMPLE_SIZE]
#
# # loop over the sampled image paths
# for (i, imagePath) in enumerate(imagePaths):
#     # load the image and clone it
#     image = cv2.imread(imagePath)
#     output = image.copy()
#
#     # resize the input image to be a fixed 128x128 pixels, ignoring
#     # aspect ratio
#     image = cv2.resize(image, (128, 128))
#     image = image.astype("float32") / 255.0
#
#     # make predictions on the image
#     preds = model.predict(np.expand_dims(image, axis=0))[0]
#     # print('*********************')
#     print(preds)
#     j = np.argmax(preds)
#     label = config.CLASSES[j]
#
#     # draw the activity on the output frame
#     text = label if label == "Non-Fire" else "WARNING! Fire!"
#     output = imutils.resize(output, width=500)
#     cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.25, (0, 255, 0), 5)
#
#     # write the output image to disk
#     filename = "{}.png".format(i)
#     p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
#     cv2.imwrite(p, output)

#  loop over the sampled image paths
tp = 0
fn = 0
fp = 0
tn = 0
for (i, (firePaths,nonFirePaths)) in enumerate(zip(firePaths,nonFirePaths)):
    # load the image and clone it
    image_fire = cv2.imread(firePaths)
    image_nonfire = cv2.imread(nonFirePaths)
    output_fire = image_fire.copy()
    output_nonfire = image_nonfire.copy()

    # resize the input image to be a fixed 128x128 pixels, ignoring
    # aspect ratio
    image = cv2.resize(output_fire, (128, 128))
    image1 = cv2.resize(output_nonfire, (128, 128))
    image = image.astype("float32") / 255.0
    image1 = image1.astype("float32") / 255.0

    # make predictions on the image
    preds = model.predict(np.expand_dims(image, axis=0))[0]
    preds_non = model.predict(np.expand_dims(image1, axis=0))[0]
    # print('*********************')
    # print(preds)
    j = np.argmax(preds)
    j1 = np.argmax(preds_non)
    label = config.CLASSES[j]
    label1 = config.CLASSES[j1]

    if label == "Fire":
        tp +=1
    else:
        fn+=1
    if label1=="Non-Fire":
        tn+=1
    else:
        fp+=1


    # # draw the activity on the output frame
    # text = label if label == "Non-Fire" else "WARNING! Fire!"
    # output = imutils.resize(output, width=500)
    # cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
    #             1.25, (0, 255, 0), 5)
    #
    # # write the output image to disk
    # filename = "{}.png".format(i)
    # p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
    # cv2.imwrite(p, output)
print(tp)
print(fp)
print(tn)
print(fn)
precision = tp/(tp+fp)*100
recall = tp/(tp+fn)*100

print("准确率为{}%".format(precision))
print("召回率为{}%".format(recall))
