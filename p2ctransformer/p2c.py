import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math


def gencirc(x_center, y_center, D):
    if D%2 == 1:
        r = (D - 1) / 2
    else:
        r = D / 2
    xc1 = x_center - r
    yc1 = y_center - r
    x_center = yc1
    y_center = xc1
    xout = np.zeros(D)
    yout = np.zeros(D)
    for i in range(D):
        thet = math.tan((2 * math.pi * (i)) / D)
        a = pow(thet, 2) + 1
        b = -2 * y_center + 2 * x_center * thet
        c = pow(y_center, 2) - 2 * y_center * x_center * thet + pow(x_center, 2) * pow(thet, 2) - pow(r, 2) * pow(thet, 2)
        print(a,b,c)
        rad = math.sqrt(pow(b, 2) - 4 * a * c) / (2 * a)
        if (i < r):
            yout[i] = -b/(2 * a) + rad
        else:
            yout[i] = -b/(2 * a) - rad
        thet = math.tan((2 * math.pi * (i)) / D)
        a = pow(thet, 2) + 1
        b = -2 * x_center * pow(thet, 2) + 2 * y_center * thet
        c = pow(x_center, 2) * pow(thet, 2) - 2 * x_center * y_center * thet + pow(y_center, 2) - pow(r, 2)
        rad = math.sqrt(pow(b, 2) - 4 * a * c) / (2 * a)
        if( i < r/2 or i > (3 * r / 2)):
            xout[i] = -b / ( 2 * a ) + rad
        else:
            xout[i] = -b / ( 2 * a) - rad
    #print(xout, yout)
    return [xout, yout]

def genthet(x_center0, y_center0, t, d):
    x_center = x_center0 - d / 2
    y_center = y_center0 - d / 2
    r = d / 2
    thet = math.tan(t)
    temp_t = (t + math.pi / 2) % (2 * math.pi)
    a = pow(thet, 2) + 1
    b = -2 * y_center * pow(thet, 2) + 2 * x_center * thet
    c = pow(y_center, 2) * pow(thet, 2) - 2 * x_center * y_center * thet + pow(x_center, 2) - pow(r, 2)
    rad = math.sqrt(pow(b, 2) - 4 * a * c) / (2 * a)
    if temp_t <= math.pi:
        y2 = -b / (2 * a) + rad
    else:
        y2 = -b / (2 * a) - rad
    b = -2 * x_center + 2 * y_center * thet
    c = pow(x_center, 2) - 2 * x_center * y_center * thet + pow(y_center, 2) * pow(thet, 2) - pow(r, 2) * pow(thet, 2)
    rad = math.sqrt(pow(b, 2) - 4 * a * c) / (2 * a)
    if temp_t <= (math.pi / 2) or temp_t >= (3 * math.pi / 2):
        x2 = -b / (2 * a) - rad
    else:
        x2 = -b / (2 * a) + rad
    return math.sqrt(pow((y2 - y_center), 2) + pow((x2 - x_center), 2))


def p2c_dic_gen(x_center, y_center, m, n):
    trans_dic = {}
    d = min(m,n)
    [Xd, Yd] =gencirc(x_center, y_center, d)
    X = np.round(Xd + d/2)
    Y = np.round(Yd + d/2)
    for i in range(d):
        for j in range(d):
            if (pow((i - d / 2), 2 ) + pow((j - d / 2), 2)) > pow((d / 2), 2):
                trans_dic[(j,i)] = 'blank'
            else:
                x = i - x_center
                y = j - y_center
                thet = math.atan2(x, y)
                yin = round(thet / (2 * math.pi / d)) % d
                thet = yin * (2 * math.pi / d)
                r = math.sqrt(pow(x, 2) + pow(y, 2))
                Len = genthet(x_center, y_center, thet, d)
                xin = max(min(round(d * (r / Len)),255), 1)
                trans_dic[(j,i)] = [yin,xin]
    
    return trans_dic


def p2c(image, trans_dic):
    [m, n] = image.shape
    d = min(m,n)
    P = np.zeros((m,n), dtype = np.uint8)
    for i in range(d):
        for j in range(d):
            if trans_dic[(j,i)] == 'blank':
                P[j][i] = np.uint8(0)
            else:
                yin = trans_dic[(j,i)][0]
                xin = trans_dic[(j,i)][1]
                P[j][i] = image[yin][xin]

    P = np.rot90(P, 3)
    P = np.fliplr(P)
    return P
    
    
'''img = cv2.imread('/home/scram-2004/Project/Our-UNet-Code/p2ctransformer/5.tif')
[m, n, o] = img.shape
trans_dic = p2c_dic_gen(127, 127, m, n)
P = p2c(img, trans_dic)
cv2.imshow('old image', img)
cv2.waitKey(0)
cv2.imshow('new image', P)
cv2.waitKey(0)
cv2.destroyAllWindows()'''