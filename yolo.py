# importing the necessary packages
import numpy as np
import time
import argparse
import cv2 as cv
import os

# constructing the argument parse and parsing the arguments
argp = argparse.ArgumentParser()
argp.add_argument("-i", "--image", required=True, help="Path to input image")
argp.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
argp.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
argp.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold while applying non-maxima suppression")
args = vars(argp.parse_args())

# loading the COCO class labels for the YOLO model
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent the class labels
np.random.seed(34)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# getting the paths to the YOLO wrights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configurationPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# loading the YOLO object detector trained on COCO dataset
print("(==>) Loading YOLO from the disk")
model = cv.dnn.readNetFromDarknet(configurationPath, weightsPath)

# loading the input and taking the spacial dimensions
image = cv.imread(args["image"])
(H, W) = image.shape[:2]

# determining the output layer names from YOLO
lr = model.getLayerNames()
lr = [lr[i - 1] for i in model.getUnconnectedOutLayers()]

# constructing a blob from the input image and performing
# the forward pass of the YOLO detector, for giving the bounding boxes
# and the probabilities of the objects
blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
model.setInput(blob)
start = time.time()
layerOutputs = model.forward(lr)
end = time.time()

# showing the processing time of YOLO
print("(==>) YOLO took {:.6f} seconds".format(end - start))

# initializing the lists of detected bounding boxes, confidences
# and class ID's
boxes = []
confidences = []
classIDs = []

# looping over each of the layer outputs
for output in layerOutputs:
    # looping over each of the detections
    for detection in output:
        # extracting the class IDs and confidence
        # of the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filtering out weak predictions by checking
        # the detected probability is greater
        # than the minimum probability (i.e., CONFIDENCE values)
        if confidence > args["confidence"]:
            # scaling the bounding box coordinates back relative to the
            # size of the image, YOLO returns the center (x,y)-coordinates
            # of the bounding boxes followed by the width and height of the boxes
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # using the center coordinates to derive the top
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # updating the list of bounding box coordinates,
            # confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

            # applying non-maxima suppression to suppress weak,
            # overlapping bounding boxes
            # Note: YOLO does not apply non-maxima suppression
            # hence explicitly applying
            NMS = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

            # ensuring at least one detection exists
            if len(NMS) > 0:
                # looping over the indexes we are keeping
                for i in NMS.flatten():
                    # extracting the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    # calculating the text width and height to draw boxes for text
                    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# displaying the output image
cv.imshow("Image", image)
cv.waitKey(0)
