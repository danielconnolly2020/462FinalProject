#Dan Connolly 6/13/2020
#MSDS 462 - Final Project

#Leveraging Yolov3 model retrained on hands, accessed from
#https://github.com/cansik/yolo-hand-detection

#Also leveraged some code elements for basics of running yolov3 in OpenCV from:
#https://github.com/nandinib1999/object-detection-yolo-opencv

#Code for taking bounding box info and figuring out how to move mouse, scroll up/down, and click
#was all developed by Dan Connolly

import cv2
import numpy as np
import time
import pyautogui

verbose=True

#get user's current resolution
screenx, screeny = pyautogui.size()

# We will be controlling the mouse with the center of the bounding box.
# it is impossible to reach the very corners, (because we are pulling
# x and y from the middle of the bounding box).  So we will create a rectangle within the image
# that represents the user's screen, and anything outside that rectangle will be treated as being at the very edge.
# We will set these edges and 20% and 80% of both the height and width, but it can be customized to the user's liking
topborder = 0.2
bottomborder = 0.8

#scroll threshold - don't want it to scroll up and down with every tiny movement, so ignore movements smaller than this
#image is 480 tall, if I set threshold to 10 then any changes under 10 are ignored, anything bigger than 10
#triggers a scroll
scrollthreshold = 10

#set the size of image that we will feed the CNN
#higher is more accurate but slower, lower is less accurate but faster
imgsize = 256

# Load yolo model
def load_yolo():
    net = cv2.dnn.readNet("/tmp/yolov3Hands/if verbose: ", "/tmp/yolov3Hands/cross-hands.cfg")
    classes = []
    with open("/tmp/yolov3Hands/cross-hands.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes)+10, 3))
    return net, classes, colors, output_layers

def detect_objects(img, net, outputLayers):
    #take in an image, convert it to a blog, and run it through the model to generate inferences
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(imgsize, imgsize), mean=(0, 0, 0), swapRB=True, crop=False) #was 320, 320
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    #process inferences and translate them into boxes with x, y, width, and height
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)

    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    #take in boxes, and draw the bounding boxes on the imaage and label them
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)

def do_boxes_overlap(box1x, box1y, box1w, box1h, box2x, box2y, box2w, box2h):
    boxes_overlap = False
    #if box2x >= box1x and box2x <= box1x + box1w and box2y >= box1y and box2y <= box1y +box1h
    if abs(box1x - box2x) < 40 and abs(box1y - box2y) < 40: boxes_overlap = True

    return boxes_overlap

def control_os(boxes, height, width, state, lastscrolly, clickcount, lastpointx, lastpointy):
    #based on the box(es) that were found, move the mouse, scroll, click, etc.
    pyautogui.FAILSAFE = False

    if verbose: print("found ", len(boxes), ' hands')

    numboxes = len(boxes)

    if numboxes == 2 and do_boxes_overlap(boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3], boxes[1][0], boxes[1][1], boxes[1][2], boxes[1][3]):
        #if it found two boxes but they are right on top of each other, then it really found the same hand twice
        #so we should treat it as one hand
        if verbose: print("found same hand twice, treat it as one hand")
        numboxes=1

    if numboxes == 1:
        # get x, y of center of bounding box
        handx = boxes[0][0] #+ .5*boxes[0][2]
        handy = boxes[0][1] #+ .5*boxes[0][3]
        handytop = boxes[0][1]

        boxwidth = boxes[0][2]
        boxheight = boxes[0][3]

        #translate x and y into a percentage = e.g. 0% for top/left, 100% for bottom/right
        handxperc = handx / width
        handyperc = handy / height

        clickcount = 0

        #print("width and height are", boxwidth, "and ", boxheight)

        if boxwidth > boxheight: #boxheight > 125:
            #we are in "scroll mode"
            if state == "scrolling":
                #if we were scrolling last time through, calculate the difference between current position and last position
                #and scroll up or down based on that difference
                if handytop > lastscrolly and abs(handytop-lastscrolly) > scrollthreshold:
                    #scroll down
                    #pyautogui.scroll(-500)
                    pyautogui.press('pagedown')
                    if verbose: print("Scroll down")
                elif handytop < lastscrolly and abs(handytop-lastscrolly) > scrollthreshold:
                    #scroll up
                    #pyautogui.scroll(500)
                    if verbose: print("scroll up")
                    pyautogui.press('pageup')
                lastscrolly = handytop
#            else: print("first time through scrolling")
            state = "scrolling"

        else:
            #we are in "move the mouse" mode
            state = "moving"
            lastscrolly=-1

            #it is impossible to reach the very corners, (because we are pulling
            #x and y from the middle of the bounding box).  So percents below .2 will be set to .2,
            #percents above .8 will be set to .8, and then we will re-scale the resulting number on a 0-1 scale
            handxpercAdj = (min(max(handxperc, topborder), bottomborder) - topborder) / (bottomborder - topborder)
            handypercAdj = (min(max(handyperc, topborder), bottomborder) - topborder) / (bottomborder - topborder)

            mousex = handxpercAdj * screenx
            mousey = handypercAdj * screeny

            if mousex == screenx:  mousex = mousex -1
            if mousey == screeny:  mousey = mousey - 1

            #print("x and y are", handx, ', ', handy, ' and width and height are ', boxes[0][2], ' and ', boxes[0][3])
            if verbose:
                print("x and y  percents are ", handxperc, ' and ', handyperc)
                print("x and y adjusted percents are ", handxpercAdj, ' and ', handypercAdj)
                print("about to move mouse x, y to ", mousex, ' and ', mousey)

            if abs(mousex-lastpointx) > 20 or abs(mousey-lastpointy) > 20:
                pyautogui.moveTo(mousex, mousey)
            lastpointx = mousex
            lastpointy = mousey

    elif numboxes >= 2: #==2:
        #we are in "click the mouse" mode
        #if we just clicked the mouse last time through, don't click it again
        if clickcount == 3: #state!="clicking":
            #click the mouse
            pyautogui.click()
        state = "clicking"
        clickcount = clickcount + 1
        if verbose: print("clickcount is ", clickcount)
        lastscrolly = -1

        #put this in for debugging - try to figure out why it is sometimes seeing same hand twice
        #print("firstbox x", boxes[0][0])
        #print("firstbox y", boxes[0][1])
        #print("firstbox width", boxes[0][2])
        #print("firstbox height", boxes[0][3])
        #print("secondbox x", boxes[1][0])
        #print("secondbox y", boxes[1][1])
        #print("secondbox width", boxes[1][2])
        #print("secondbox height", boxes[1][3])

    #elif numboxes > 2:
    #    print("****************OMG more than two hands found, are you a mutant? Or two people in frame?: ", len(boxes))

    else:
        state = "none"
        lastscrolly = -1
        clickcount = 0

    return state, lastscrolly, clickcount, lastpointx, lastpointy

def webcam_detect():
    totalframes=0
    state = "none"
    lastscrolly = -1
    clickcount = 0
    lastpointx = 100000
    lastpointy = 10000

    while True:
        t = time.time()
        totalframes += 1

        _, frame = cap.read()
        height, width, channels = frame.shape
        #print('height and width are', height, ' and', width)
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)

        #call function to process the boxes and control the OS (move mouse, click, scroll)
        state, lastscrolly, clickcount, lastpointx, lastpointy = control_os(boxes, height, width, state, lastscrolly, clickcount, lastpointx, lastpointy)

        if verbose: print("time taken = {:.2f} sec".format(time.time() - t))
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    return totalframes

#load yolo hand model
model, classes, colors, output_layers = load_yolo()

#start the webcam
cap = cv2.VideoCapture(0)

starttime = time.time()

#run loop to capture frames from video, do object detection, and control OS mouse movement/clicks
totalframes = webcam_detect()

totaltime = time.time() - starttime
print(str(totalframes), " frames captured in {:.2f} sec".format(totaltime), "seconds for ", totalframes/totaltime, "frames per second")

cv2.destroyAllWindows()
