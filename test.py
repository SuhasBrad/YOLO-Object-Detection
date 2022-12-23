# importing the necessary packages
import numpy as np
import time
import argparse
import cv2
import os


def extract_boxes_confidences_classids(outputs, confidence, width, height):
    # initializing the lists of detected bounding boxes, confidences
    # and class ID's
    boxes = []
    confidences = []
    classIDs = []
    # looping over each of the layer outputs
    for output in outputs:
        # looping over each of the detections
        for detection in output:
            # extracting the class IDs and confidence
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]

            # filtering out weak predictions by checking
            # the detected probability is greater
            # than the minimum probability (i.e., CONFIDENCE values)
            if conf > confidence:
                # scaling the bounding box coordinates back relative to the
                # size of the image, YOLO returns the center (x,y)-coordinates
                # of the bounding boxes followed by the width and height of the boxes
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype("int")

                # using the center coordinates to derive the top
                # and left corner of the bounding box
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                # updating the list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs


def draw_bounding_boxes(image, boxes, confidences, classIDs, nms, colors):
    if len(nms) > 0:
        # looping over the indexes we are keeping
        for i in nms.flatten():
            # extracting the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            # calculating the text width and height to draw boxes for text
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def make_predictions(model, layer_names, labels, image, confidence, threshold):
    # extracting dimensions of the image
    height, width = image.shape[:2]

    # constructing a blob from the input image and performing
    # the forward pass of the YOLO detector, for giving the bounding boxes
    # and the probabilities of the objects
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    outputs = model.forward(layer_names)
    end = time.time()

    # showing the processing time of YOLO
    print("(==>) YOLO took {:.6f} seconds".format(end - start))

    # extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # applying non-maxima suppression to suppress weak,
    # overlapping bounding boxes
    # Note: YOLO does not apply non-maxima suppression
    # hence explicitly applying
    nms = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, nms


# constructing the argument parse and parsing the arguments
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument("-w", "--weights", type=str, default='', help="Path to YOLO weights")
    argp.add_argument("-cfg", "--config", type=str, default='', help="Path to YOLO configuration file")
    argp.add_argument("-l", "--labels", type=str, default='', help="Path to Labels file")
    argp.add_argument("-c", "--confidence", type=float, default=0.5,
                      help="Minimum confidence to filter out weaker detections")
    argp.add_argument("-t", "--threshold", type=float, default=0.3, help="Threshold for Non-Max Suppression")
    argp.add_argument("-u", "--use_gpu", default=False, action='store_true',
                      help="Use GPU(Note: OpenCV must be compiled correctly")
    argp.add_argument("-s", "--save", default=False, action='store_true',
                      help='Whether or not the output should be saved')
    argp.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')
    argp.add_argument('-i', '--image_path', type=str, default='', help='Path to the image file.')

    input_group = argp.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image_path', type=str, default='', help="Path to image file")
    input_group.add_argument('-v', '--video_path', type=str, default='', help="Path to video file")

    args = argp.parse_args()

    # Get the labels
    labels = open(args.labels).read().strip().split('\n')

    # Initialize a list of colors to represent the class labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Loading config file and weights using OpenCV
    print("(==>) Loading YOLO from disk")
    model = cv2.dnn.readNetFromDarknet(args.config, args.weights)

    if args.use_gpu:
        print("Using GPU")
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if args.save:
        print("Creating an output directory if it doesn't already exist.")
        os.makedirs("output", exist_ok=True)

    # Getting Output Layer names
    layer_names = model.getLayerNames()
    layer_names = [layer_names[i - 1] for i in model.getUnconnectedOutLayers()]

    # Getting the input image
    image = cv2.imread(args.image_path)

    boxes, confidences, classIDs, nms = make_predictions(model, layer_names, labels, image, args.confidence,
                                                         args.threshold)

    image = draw_bounding_boxes(image, boxes, confidences, classIDs, nms, colors)

    # show the output image
    if args.show:
        cv2.imshow("YOLO Object Detection", image)
        cv2.waitKey(0)

    if args.save:
        cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image)
    cv2.destroyAllWindows()
