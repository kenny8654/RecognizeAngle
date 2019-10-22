import numpy as np
import cv2
import math
def azimuthAngle( x1,  y1,  x2,  y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return 360-(angle * 180 / math.pi)

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, img1 = cap.read()
    img = cv2.GaussianBlur(img1,(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    mean = np.mean(img)
    ret,thresh = cv2.threshold(img,mean,255,0)
    _ ,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    areas = {}
    for i,contour in enumerate(contours):
        area = cv2.contourArea(contour)
        areas[i] = area

    try:
        areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
        drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
        color = (255, 0, 0)
        cv2.drawContours(img1, contours, areas[1][0], color, 1, 8, hierarchy)
        cv2.drawContours(img1, contours, areas[2][0], color, 1, 8, hierarchy)

        p1 = np.around(np.mean(contours[areas[1][0]], axis=0).flatten())
        p2 = np.around(np.mean(contours[areas[2][0]], axis=0).flatten())
        p1 = ((int(p1[0]),int(p1[1])))
        p2 = ((int(p2[0]),int(p2[1])))

        # print('thresh.shape[0]/2 :',thresh.shape[0]/2,thresh.shape[1]/2)
        # print('p1 :' , p1 , ', p2 : ',p2)
        dist_p1 = abs(thresh.shape[1]/2 - p1[0]) + abs(thresh.shape[0]/2 - p1[1])
        dist_p2 = abs(thresh.shape[1]/2 - p2[0]) + abs(thresh.shape[0]/2 - p2[1])
        # print('dist_p1 : ',dist_p1,', dist_p2 : ',dist_p2)
        if dist_p2 > dist_p1:
            center = p1
            target = p2
        else:
            center = p2
            target = p1

        cv2.circle(img1, (int(thresh.shape[1]/2),int(thresh.shape[0]/2)), 1, (255,255,255), -1)
        cv2.circle(img1, center, 1, (0,255,0), -1) #green
        cv2.circle(img1, target, 1, (0,0,255), -1) #red

        deg = azimuthAngle(center[1], center[0], target[1], target[0])
        print('deg : ',deg)

        text = 'deg : '+str(int(deg))
        cv2.putText(img1, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
    except:
        print('',end='')
    # cv2.imshow('frame',drawing)
    cv2.imshow('show',img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
