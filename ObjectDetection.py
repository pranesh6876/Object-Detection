import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(len(classNames))

modelConfiguration = 'yolov3.cfg' #yolov3_320 is the best trade off between speed and accuracy
modelWeights = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []      # bounding box corner points
    classIds = []  # class id with the highest confidence
    confs = []     # confidence value of the highest class
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = cap.read()

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(len(layerNames))
    #print(net.getUnconnectedOutLayers())
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])

    findObjects(outputs,img)  
    cv.imshow("image", img)

    if cv.waitKey(1) == 27 :
        break


cap.release()
cv.destroyAllWindows()







# import cv2
# import numpy as np
# import time
# # A required callback method that goes into the trackbar function.
# def nothing(x):
#     pass

# # Initializing the webcam feed.
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# # Create a window named trackbars.
# cv2.namedWindow("Trackbars")

# # Now create 6 trackbars that will control the lower and upper range of 
# # H,S and V channels. The Arguments are like this: Name of trackbar, 
# # window name, range,callback function. For Hue the range is 0-179 and
# # for S,V its 0-255.
# cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
 
# while True:
    
#     # Start reading the webcam feed frame by frame.
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Flip the frame horizontally (Not required)
#     frame = cv2.flip( frame, 1 ) 
    
#     # Convert the BGR image to HSV image.
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Get the new values of the trackbar in real time as the user changes 
#     # them
#     l_h = cv2.getTrackbarPos("L - H", "Trackbars")
#     l_s = cv2.getTrackbarPos("L - S", "Trackbars")
#     l_v = cv2.getTrackbarPos("L - V", "Trackbars")
#     u_h = cv2.getTrackbarPos("U - H", "Trackbars")
#     u_s = cv2.getTrackbarPos("U - S", "Trackbars")
#     u_v = cv2.getTrackbarPos("U - V", "Trackbars")
 
#     # Set the lower and upper HSV range according to the value selected
#     # by the trackbar
#     lower_range = np.array([l_h, l_s, l_v])
#     upper_range = np.array([u_h, u_s, u_v])
    
#     # Filter the image and get the binary mask, where white represents 
#     # your target color
#     mask = cv2.inRange(hsv, lower_range, upper_range)
 
#     # You can also visualize the real part of the target color (Optional)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
    
#     # Converting the binary mask to 3 channel image, this is just so 
#     # we can stack it with the others
#     mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
#     # stack the mask, orginal frame and the filtered result
#     stacked = np.hstack((mask_3,frame,res))
    
#     # Show this stacked frame at 40% of the size.
#     cv2.imshow('Trackbars',cv2.resize(stacked,None,fx=0.4,fy=0.4))
    
#     # If the user presses ESC then exit the program
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
    
#     # If the user presses `s` then print this array.
#     if key == ord('s'):
        
#         thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
#         print(thearray)
        
#         # Also save this array as penval.npy
#         np.save('hsv_value',thearray)
#         break
    
# # Release the camera & destroy the windows.    
# cap.release()
# cv2.destroyAllWindows()