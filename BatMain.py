__author__ = 'Anochjhn Iruthayam'



import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Set up global frequency band. Set to the range of Bat Calls aka. 13 Khz to 75 KHz into Pixel values
getHeightMin = 500
getHeightMax = 980

def getSampleList(path):
    sampleList = []
    for file in os.listdir(path):
        if file.endswith(".s16"):
            sampleList.append(file)
    return sampleList

#Scan the spectrogram with sliding window.
#def scanHorizontal(img)
#    getHeight, getWidth  = img.shape


def findEvent(soundFilePath):
    threshold = 5
    bottomY = []
    img = cv2.imread(soundFilePath,0)
    imgColor = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #cv2.circle(imgColor, (50,10), 10, (0,0,255), -1)
    #cv2.imshow('image', img)
    print img.shape
    ColumnCount = 0
    RowCount = 0
    reset = 1
    resetRow = 1
    startY = 0
    startX = 0
    bX = 0
    bY = 0
    end = 0
    getHeight, getWidth  = img.shape

    topX, topY, endX = verticalScan(img)
    for i in range (0,len(topX)):
        bottomY.append(horizontelScan(img, topX[i], topY[i], endX[i]))


    #bottomY = horizontelScan(img,topX,topY,endX)
    print len(topX)
    print len(topY)
    print len(endX)
    print len(bottomY)

    for i in range(0,len(bottomY)):
        if topY[i] > getHeightMin and bottomY[i] < getHeightMax:
            cv2.rectangle(imgColor, (topX[i],topY[i]), (endX[i],bottomY[i]), (0,0,255),3)
            #/home/anoch/Documents/BatSamples/SpectrogramMarked
            imgEvent = img[topY[i]:bottomY[i], topX[i]:endX[i]]
            cv2.imwrite("/home/anoch/Documents/BatSamples/SpectrogramMarked/Event" + str(i) + ".png", imgEvent)
    plt.imshow(imgColor)
    plt.xticks([]), plt.yticks([])
    plt.show()
    #cv2.imshow('image',imgColor)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def verticalScan(img):
    topX = []
    topY = []
    endX = []
    threshold = 10
    ColumnCount = 0
    reset = 1
    getHeight, getWidth  = img.shape
    for x in range(0,getWidth):
        if ColumnCount > 60 and reset == 1:
            #cv2.line(imgColor,(startX,startY), (x,startY), (0,0,255),3)
            topX.append(startX)
            topY.append(startY)
            endX.append(x)
            ColumnCount = 0
        for y in range(getHeightMin, getHeightMax):
            if img.item(y,x) > threshold:
                ColumnCount += 1
                if reset == 1:
                    startX = x
                    startY = y
                    reset = 0
                break
            elif y == getHeightMax-1:# if we reach end of the vertical line, then there is no white pixel
                reset = 1
    return topX, topY, endX

def horizontelScan(img, StartX, StartY, EndX):
    indexDeletion = []
    threshold = 5
    getHeight, getWidth  = img.shape
    rowCount = 0
    bottomY = getHeightMax
    rowReset = 1
    for eventY in range(StartY, getHeightMax):
        #if rowCount > 10:
        #    bottomY = (bY)
        #    rowCount = 0
        for eventX in range(StartX, EndX):
            if img.item(eventY,eventX) > threshold:
                rowCount += 1
                if rowCount > 10:
                    bottomY = (bY)
                    rowCount = 0
                bY = eventY
                break
    return bottomY

def createSpectrogram(path):
    sampleList = getSampleList(path)
    os.chdir(path)
    for soundFile in sampleList:
        print "Processing " + soundFile + " at channel 1"
        soxCommand = "sox -c 4 -r 500e3 " + soundFile + " -n remix 1 trim 0s 500000s spectrogram -r -m -x 5000 -y 1025 -z 88 -o Spectrogram/" + os.path.splitext((soundFile))[0] + "Ch1.png"
        os.system(soxCommand)
    print "Done!"

def main():
    #createSpectrogram("/home/anoch/Documents/BatSamples/")
    findEvent("/home/anoch/Documents/BatSamples/Spectrogram/srTest.png")



#run main
main()