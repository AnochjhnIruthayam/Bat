__author__ = 'Anochjhn Iruthayam'



import numpy as np
import cv2
import os


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
    img = cv2.imread(soundFilePath,0)
    imgColor = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #cv2.circle(imgColor, (50,10), 10, (0,0,255), -1)
    #cv2.imshow('image', img)
    print img.shape
    ColumnCount = 0
    reset = 1
    start = 0
    end = 0
    getHeight, getWidth  = img.shape
    getHeight -= 3 #to remove noise
    for x in range(0,getWidth):
        whiteFlag = 0
        if ColumnCount > 20:
            cv2.line(imgColor,(start,500), (end,500), (0,0,255),5)
            #cv2.circle(imgColor, (start,500), 10, (0,0,255), -1)
            #cv2.circle(imgColor, (end,500), 10, (0,0,255), -1)
            #print str(img.item(y,x)) + " at position: " + str(y) + " " + str(x)

        for y in range(0, getHeight):
            if img.item(y,x) > threshold and whiteFlag == 0:
                whiteFlag = 1
                ColumnCount += 1
                if reset == 1:
                    start = x
                    reset = 0

                break
            elif y == getHeight-1 and whiteFlag == 0:
                reset = 1
                ColumnCount = 0
                end = x-1

    cv2.imshow('image',imgColor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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
    findEvent("/home/anoch/Documents/BatSamples/Spectrogram/srCh1.png")



#run main
main()