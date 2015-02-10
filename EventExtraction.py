__author__ = 'Anochjhn Iruthayam'

import os
import cv2
import numpy as np
from numpy.polynomial import polynomial as P
def getFileList(path, extension):
    sampleList = []
    for file in os.listdir(path):
        if file.endswith(extension):
            sampleList.append(file)
    return sampleList


def createSpectrogram(path):
    sampleList = getFileList(path,".s16")
    os.chdir(path)
    for soundFile in sampleList:
        print "Processing " + soundFile + " at channel 1"
        soxCommand = "sox -c 4 -r 500e3 " + soundFile + " -n remix 1 trim 0s 500000s spectrogram -r -m -x 5000 -y 1025 -z 88 -o Spectrogram/" + os.path.splitext((soundFile))[0] + ".png"
        os.system(soxCommand)
    print "Conversion process done!"

def verticalScan(img):
    topX = []
    topY = []
    endX = []
    threshold = 10
    ColumnCount = 0
    reset = 1
    getHeight, getWidth  = img.shape
    for x in range(0,getWidth):
        #Make sure that the shape has the right size width, when it does save the TOP x,y (left top) and the end x, rightmost
        if ColumnCount > 60 and ColumnCount < 1000 and reset == 1:
            topX.append(startX)
            topY.append(startY)
            endX.append(x)
            ColumnCount = 0
        #subtract -4 to filter out the noise at the very bottom
        for y in range(0, getHeight-4):
            #When we see a  white pixel register it and break this loop
            if img.item(y,x) > threshold:
                #for eaach white pixel, count one up
                ColumnCount += 1
                #if we find a white pixel for the first time, then save the point
                if reset == 1:
                    startX = x
                    startY = y
                    reset = 0
                ########################### Procentage pixel decision maker ##########################
                #This occours when the there has been several black vertical lines
                if y < startY:
                    countWhitePixels = 0
                    length = startY-y
                    for i in range(y, startY):
                        if img.item(i,x) > threshold:
                            countWhitePixels += 1
                    if ((float(countWhitePixels)/length)*100) > 80.0:
                        startY = y
                ######################################################################################
                break
            elif y == getHeight-5:# if we reach end of the vertical line, then there is no white pixel
                reset = 1
    return topX, topY, endX

#This algorithm is different from the vertical scanner. Diffrent way to filter out noise.
def horizontelScan(img, StartX, StartY, EndX):
    getHeight, getWidth  = img.shape
    topX = StartX
    topY = StartY
    end = EndX
    bottomY = getHeight-4
    #This is defined as the minimum size pixel that a bat event can be
    count_threshold = 10
    COUNT_THRESHOLD_MIN = 10
    COUNT_THRESHOLD_MAX = 500
    EventFlag = 0
    threshold = 10
    rowCount = 0
    tempY = 0
    allowed_black_count = 0
    ALLOWED_BLACK = 3
    for eventY in range(StartY, getHeight-4):
        #this IS recognized as a bat event, set flag to TRUE and save the values
        if rowCount > COUNT_THRESHOLD_MIN and COUNT_THRESHOLD_MAX > rowCount and allowed_black_count < ALLOWED_BLACK:
            EventFlag = 1
            bottomY = tempY
            if allowed_black_count == ALLOWED_BLACK-1:
                return topX, topY, end, bottomY, EventFlag
        elif rowCount < count_threshold or allowed_black_count > ALLOWED_BLACK or rowCount > COUNT_THRESHOLD_MAX:
            #this IS NOT recognized as a bat event, set flag to FALSE
            EventFlag = 0
            #if thisistheend == 1:
            #    return topX, topY, end, bottomY, EventFlag
        for eventX in range(StartX, EndX):
            #whenever it finds a white pixel, then count one up
            if img.item(eventY,eventX) > threshold:
                rowCount += 1
                tempY = eventY
                allowed_black_count = 0
                break
            elif EndX - 1 == eventX:
                allowed_black_count += 1


    return topX, topY, end, bottomY, EventFlag

def bestFit(imgEventPath):
    imgEvent = cv2.imread(imgEventPath,0)
    imgColor = cv2.cvtColor(imgEvent,cv2.COLOR_GRAY2RGB)
    X = []
    Y = []
    tempX = 0
    tempY = 0
    firstTime = 1
    threshold = 5
    firstX = 0
    firstY = 0
    isFirstFound = 0
    getHeight, getWidth  = imgEvent.shape

    # Find the first occurence of the frontline, by looking at the y axis

    for preX in range(0,getWidth):
        if isFirstFound == 1:
            break
        for preY in range (0, getHeight):
            if imgEvent.item(preY,preX) > threshold:
                firstX = preX
                firstY = preY
                isFirstFound = 1
                break



    for mEventY in range(firstY,getHeight):
        for mEventX in range (firstX, getWidth):
            if imgEvent.item(mEventY,mEventX) > threshold:
                #mEvent condition check to remove noise by only looking at the first part of the divided image
                if len(X) == 0 and mEventX < (getWidth/2):
                    tempX = mEventX
                    tempY = mEventY
                if abs(mEventX-tempX) < 5 and abs(mEventY-tempY) < 5:
                    #if firstTime == 1:
                        X.append(mEventX)
                        Y.append(mEventY)
                        tempX = mEventX
                        tempY = mEventY
                    #    firstTime = 0
                    #elif mEventX-tempX != 0:
                    #    X.append(mEventX)
                     #   Y.append(mEventY)
                    #    tempX = mEventX
                     #   tempY = mEventY
                break
    reduceX = []
    reduceY = []
    index = 0
    tempValY = 0
    for n in range(0, len(X)):
        X_val = X[n]
        tempValY = 0

        for m in range(0, len(X)):
            if X_val == X[m]:
                if tempValY < Y[m]:
                    tempValY = Y[m]
                    index = m
        reduceX.append(X[index])
        reduceY.append(Y[index])


    redX = []
    redY = []
    for i in range(0, len(reduceX)-1):
        if reduceX[i] != reduceX[i+1]:
            redX.append(reduceX[i])
            redY.append(reduceY[i])
       #if i == len(reduceX)-1:
        #    if reduceX[i-1] != reduceX[i-2]:
         #       redX.append(reduceX[i-1])
          #      redY.append(reduceY[i-1])




    #####
    if len(redX) > 5:
        for i in range(0,len(redX)):
            cv2.circle(imgColor,(redX[i],redY[i]), 1, (0,0,255), -1)



        REDX = []
        REDY =[]
        # ignores the last entry. To get more stable data
        for i in range(0,len(redX)-1):
            REDX.append(redX[i])
            REDY.append(redY[i])
            #i = i+1
        #print "X: " + str(REDX)
        #print "Y: " + str(REDY)
        x_new = np.linspace(min(REDX), max(REDX))
        residual = 999999
        residual_index = 0
        #Find the degree with the lowest residuals. The lowest amount of LS error
        for i in range(0,8):
            coef, stats = P.polyfit(REDX,REDY,i, full=True)
            #print "residuals: " + str(stats[0])
            if residual > stats[0]:
                residual = stats[0]
                residual_index = i
        coef = P.polyfit(REDX,REDY,residual_index)
        #ffit = P.polyval(x_new, coef)

        #plt.plot(x_new, ffit)
       # plt.show()
        #print "plotting.." + str(residual_index)
        ffit = P.Polynomial(coef)

        #plt.plot(redX, redY, '.', x_new, ffit(x_new), '-')
        #for i in range(0,len(x_new)):
        #    cv2.circle(imgColor,(int(x_new[i]),int(ffit(x_new)[i])), 1, (0,255,0), -1)
        #plt.show()



        #print "\n"
        #plt.imshow(imgColor)
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        return ffit, min(REDX), max(REDX), min(REDY), max(REDY)
    else:
        #print "X: " + str(redX)
        #print "Y: " + str(redY)
        #print np.polyfit(redX,redY,7)
        return 0, 0, 0, 0, 0

def getFrontLineFeature(imgPath):
    X = []
    ffit, miniX, maxiX, miniY, maxiY = bestFit(imgPath)
    if ffit == 0 and miniX == 0 and maxiX == 0 and maxiY == 0 and miniY == 0:
        X.append(0)
        return X

    rangey = maxiX - miniX
    #print "Range X: " + str(rangey), str(maxiX),  str(miniX)
    rangey = maxiY - miniY
    #print "Range Y: " + str(rangey), str(maxiY), str(miniY)
    step = rangey/10.0
    #print miniY, maxiY, step
    we = miniX
    for s in range(0,11):
        #print "step: " + str(s)
        newY = (step*s) + miniY
        #print "for y: " + str(newY)
        #We want to ensure we make a good search.
        we -= 2
        for i in range(0, 10000):
            tempy = ffit(we)

            if ((tempy + 1) > newY) and (newY > (tempy - 1)):
                #print "GOTIT for: " + str(newY)
                #print "Result x: "+ str(we) + "  y: " + str(tempy) +  "\n"
                X.append(we)
                break

            we += 0.01

    return X