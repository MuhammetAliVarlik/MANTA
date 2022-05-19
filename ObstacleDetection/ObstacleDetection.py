import numpy as np
import cv2
import pyautogui
from matplotlib import pyplot as plt
from matplotlib import image as image
def empty(a):
    pass
def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha


    # Calculate new histogram with desired range and show histogram
    """new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()"""


    auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 480)
cv2.createTrackbar("Thresh", 'HSV', 0, 255, empty)
cv2.createTrackbar("Thresh2", 'HSV', 0, 255, empty)
"""cv2.createTrackbar("HUE Min", 'HSV', 0, 179, empty)
cv2.createTrackbar("HUE Max", 'HSV', 179, 179, empty)
cv2.createTrackbar("SAT Min", 'HSV', 0, 255, empty)
cv2.createTrackbar("SAT Max", 'HSV', 255, 255, empty)
cv2.createTrackbar("VALUE Min", 'HSV', 0, 255, empty)
cv2.createTrackbar("VALUE Max", 'HSV', 255, 255, empty)"""

while True:
    screenshot=pyautogui.screenshot(region=(100,100, 1024, 768))
    screenshot=np.array(screenshot)
    screenshot_orj=cv2.cvtColor(screenshot,cv2.COLOR_RGB2BGR)
    hsv=cv2.cvtColor(screenshot_orj, cv2.COLOR_BGR2HSV)
    screenshot = cv2.cvtColor(screenshot_orj, cv2.COLOR_BGR2GRAY)
    thresh=cv2.getTrackbarPos("Thresh", "HSV")
    thresh2 = cv2.getTrackbarPos("Thresh2", "HSV")
    """
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")"""
    #mask = cv2.inRange(hsv, (h_min,s_min,v_min), (h_max, s_max, v_max))
    """mask = cv2.inRange(hsv, (87, 106, 207), (99, 255, 255))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    bluefilter = cv2.bitwise_and(screenshot, screenshot, mask=mask)
    filter_net=cv2.subtract(screenshot,bluefilter)
    div = 20
    I = filter_net // div * div + div // 2
    height, width, channels = filter_net.shape
    half = filter_net[0:int(height / 2), 0:width]
    G = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    G = cv2.GaussianBlur(G, (11, 11), 0)
    G = cv2.medianBlur(G, 9)"""
    #G=cv2.blur(G,(30,30))
    height, width = screenshot.shape
    half_orj = screenshot_orj[0:int(height / 2), 0:width]
    half = cv2.cvtColor(half_orj, cv2.COLOR_BGR2GRAY)
    half = cv2.medianBlur(half, 9)
    # Canny Edge Detection:
    Threshold1 = 100
    Threshold2 = 200
    FilterSize = 5
    """
    137
    255
    """
    ret, thresh1 = cv2.threshold(half, 137, 255, cv2.THRESH_BINARY)
    #auto_result, alpha, beta = automatic_brightness_and_contrast(thresh1)
    E = cv2.Canny(thresh1, Threshold1, Threshold2, FilterSize)
    Rres = 1
    Thetares = 1 * np.pi / 180
    Threshold = 1
    minLineLength = 1
    maxLineGap = 100

    contours, hierarchy = cv2.findContours(E, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    half = cv2.drawContours(half, contours, -1, (0, 255, 75), 2)
    ret2, thresh3 = cv2.threshold(half, 135, 255, cv2.THRESH_BINARY)
    height, width = thresh3.shape
    thresh3 = thresh3[0:int(height / 2)-40, 0:width]
    top_half=thresh3
    bottom_half = thresh3
    E1 = cv2.Canny(thresh3, Threshold1, Threshold2, FilterSize)
    contours2, hierarchy2 = cv2.findContours(E1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """cnt=contours2[]
    print(len(contours2))
    left=tuple(cnt[cnt[:,:,0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmax()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmin()][0])"""
    #print(left,right,top,bottom)
    #cv2.circle(half_orj, top, 5, (0, 0, 255), -1)
    half_orj = cv2.drawContours(half_orj, contours2, -1, (0, 255, 75), 1)
    cv2.imshow("Screen",thresh3)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
"""
I_Org = cv2.imread('blue1.jpg')
# colorReduce()
div = 100
I = I_Org // div * div + div // 2
scale_percent = 30
width = int(I.shape[1] * scale_percent / 100)
height = int(I.shape[0] * scale_percent / 100)
dim = (width, height)
I = cv2.resize(I, dim, interpolation=cv2.INTER_AREA)
height, width, channels = I.shape
half = I[int(height/2)+35:int(height/2)+150,0:width]
G = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
#G = cv2.GaussianBlur(G, (11, 11), 0)
G = cv2.medianBlur(G, 9)
#G=cv2.blur(G,(30,30))
cv2.imshow("I",G)
cv2.waitKey()
# Canny Edge Detection:
Threshold1 = 100
Threshold2 = 200
FilterSize = 5
ret,thresh1 = cv2.threshold(G,100,255,cv2.THRESH_BINARY)
cv2.imshow("E",thresh1)
cv2.waitKey()
E = cv2.Canny(thresh1, Threshold1, Threshold2, FilterSize)

cv2.imshow("E",E)
cv2.waitKey()
Rres = 1
Thetares = 1*np.pi/180
Threshold = 1
minLineLength = 1
maxLineGap = 100
lines = cv2.HoughLinesP(E,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 250,maxLineGap = 200)
N = lines.shape[0]
for i in range(N):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]
    cv2.line(half,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow("I2",half)
#cv2.imshow("I1",I_Org)
cv2.waitKey()"""