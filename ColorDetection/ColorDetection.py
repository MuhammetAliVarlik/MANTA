import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import imutils
from collections import deque
import argparse
from matplotlib import cm
from matplotlib.ticker import LinearLocator
"""
10 100 100
0 100 100

"""


def empty(a):
    pass

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
cap = cv2.VideoCapture(0)
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 480)
cv2.createTrackbar("HUE Min", 'HSV', 0, 179, empty)
cv2.createTrackbar("HUE Max", 'HSV', 179, 179, empty)
cv2.createTrackbar("SAT Min", 'HSV', 0, 255, empty)
cv2.createTrackbar("SAT Max", 'HSV', 255, 255, empty)
cv2.createTrackbar("VALUE Min", 'HSV', 0, 255, empty)
cv2.createTrackbar("VALUE Max", 'HSV', 255, 255, empty)
cv2.createTrackbar("HUE2 Min", 'HSV', 0, 179, empty)
cv2.createTrackbar("HUE2 Max", 'HSV', 179, 179, empty)
cv2.createTrackbar("SAT2 Min", 'HSV', 0, 255, empty)
cv2.createTrackbar("SAT2 Max", 'HSV', 255, 255, empty)
cv2.createTrackbar("VALUE2 Min", 'HSV', 0, 255, empty)
cv2.createTrackbar("VALUE2 Max", 'HSV', 255, 255, empty)
height = 516
width = 516
blank_image = np.zeros((height, width, 3), np.uint8)
x=np.arange(0,width,1)
y=np.arange(0,height,1)
s = (height,width)
X, Y = np.meshgrid(x, y)
z = np.zeros(s)
Z = z.reshape(X.shape)
now = datetime.now()
# heat_map_list={0:[0,0,0],1:[1,0,0],2:[2,0,0],3:[3,0,0],4:[4,0,0],5:[5,0,0],6:[6,0,0],7:[7,0,0],8:[7,0,0],9:[9,0,0]}

n = 0
while True:
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    #frame = cv2.medianBlur(frame, 5)
    # frame=cv2.blur(frame,(10,10))
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
    h_min2 = cv2.getTrackbarPos("HUE2 Min", "HSV")
    h_max2 = cv2.getTrackbarPos("HUE2 Max", "HSV")
    s_min2 = cv2.getTrackbarPos("SAT2 Min", "HSV")
    s_max2 = cv2.getTrackbarPos("SAT2 Max", "HSV")
    v_min2 = cv2.getTrackbarPos("VALUE2 Min", "HSV")
    v_max2 = cv2.getTrackbarPos("VALUE2 Max", "HSV")

    # 170, 70, 50  180, 255, 255
    # lower=np.array([h_min,s_min,v_min])
    # upper = np.array([h_max, s_max, v_max])
    # mask=cv2.inRange(hsv_frame,lower,upper)
    # mask1 = cv2.inRange(hsv_frame, (h_min,s_min,v_min), (h_max, s_max, v_max))
    # mask2 = cv2.inRange(hsv_frame, (h_min2,s_min2,v_min2), (h_max2, s_max2, v_max2))
    # mask1 = cv2.inRange(hsv_frame, (0, 89, 103), (4, 255, 255))
    # mask2 = cv2.inRange(hsv_frame, (140, 89, 103), (179, 255, 255))
    # mask1 = cv2.inRange(hsv_frame, (0, 145, 20), (5, 255, 255))
    # mask2 = cv2.inRange(hsv_frame, (175, 145, 20), (180, 255, 255))
    mask1 = cv2.inRange(hsv_frame, (0, 123, 20), (7, 255, 255))
    mask2 = cv2.inRange(hsv_frame, (163, 123, 20), (180, 255, 255))

    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    red = cv2.bitwise_and(frame, frame, mask=mask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(red, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(red, center, 5, (0, 0, 255), -1)
    # update the points queue

    # cv2.imshow("hsv", hsv_frame)
    #cv2.imshow("blank", blank_image)
    #cv2.imshow('mask', mask)
    #cv2.imshow('img', frame)
    cv2.imshow("Red Mask", red)
    k = cv2.waitKey(1) & 0xff
    # Exit code if ESC is pressed
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
def createList(n):
    lst = []
    for i in range(n+1):
        lst.append(i)
    return(lst)

# Plot the surface.
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
 #                      linewidth=0, antialiased=False)



# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)
plt.plot(x, z)
plt.show()
"""img = cv2.imread('circles.jpg')
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

## Gen lower mask (0-5) and upper mask (175-180) of RED
#mask1 = cv2.inRange(img_hsv, (0, 155, 84), (5, 255, 255))
#mask2 = cv2.inRange(img_hsv, (161, 155, 84), (180, 255, 255))
mask1 = cv2.inRange(img_hsv, (0, 145, 20), (5, 255, 255))
mask2 = cv2.inRange(img_hsv, (175, 145, 20), (180, 255, 255))
## Merge the mask and crop the red regions
mask = cv2.bitwise_or(mask1, mask2 )
red = cv2.bitwise_and(img, img, mask=mask)
mask = cv2.GaussianBlur(mask, (9, 9), 0)
detected_circles = cv2.HoughCircles(mask,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=1, maxRadius=100)

# Draw circles that are detected.
if detected_circles is not None:

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()"""
