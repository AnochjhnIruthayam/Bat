__author__ = 'Anochjhn Iruthayam'



import cv2
import os
import EventExtraction as ee
import getFunctions
import LabelFunctions

# import MultiNEAT as NEAT #not as neat I thought it would be! BASTARD!! Still remember ET to phone home
# from pybrain.tools.shortcuts import buildNetwork

# Set up global frequency band. Set to the range of BatClassification Calls aka. 13 Khz to 75 KHz into Pixel values
#getHeightMin = 500
#getHeightMax = 980



#Scan the spectrogram with sliding window.
#def scanHorizontal(img)
#    getHeight, getWidth  = img.shape


def findEvent(SearchPath, eventFile, SavePath):
    threshold = 5
    soundFilePath = SearchPath + eventFile #"/home/anoch/Documents/BatSamples/Spectrogram/sr_500000_ch_4_offset_00000000008460585000.png"
    #Read image
    img = cv2.imread(soundFilePath,0)
    imgColor = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #get information about size
    imgHeight,imgLength = img.shape
    #Create Lists
    topX = []
    topY = []
    endX = []
    bottomY = []
    #Run Vertical Scanner
    StartX, StartY, end = ee.verticalScan(img)
    for i in range (0,len(StartX)):
        #flag is added to filter out the non bat events by looking at the horizontal axis
        tempSX, tempSY, tempEnd, tempBottom, flag = ee.horizontelScan(img, StartX[i], StartY[i], end[i])
        if flag == 1:
            topX.append(tempSX)
            topY.append(tempSY)
            endX.append(tempEnd)
            bottomY.append(tempBottom)
        #bottomY.append()
    crop_offset = 5
    eventNum = []
    for i in range(0,len(bottomY)):
        #if topY[i] > getHeightMin and bottomY[i] < getHeightMax: # ensure that the call is in range

        cv2.rectangle(imgColor, (topX[i],topY[i]), (endX[i],bottomY[i]), (0,0,255),3)
        if topX[i]>crop_offset and topY[i]>crop_offset and endX[i]< imgLength-crop_offset and bottomY[i] < imgHeight- crop_offset:
            imgEvent = img[topY[i]-crop_offset:bottomY[i]+crop_offset, topX[i]-crop_offset:endX[i]+crop_offset]
            eventNum.append(i)
            checkFolder = SavePath + os.path.splitext((eventFile))[0]
            if not os.path.exists(checkFolder):
                os.makedirs(checkFolder)
            cv2.imwrite(SavePath + os.path.splitext((eventFile))[0] + "/Event" + str(i) + ".png", imgEvent)
    cv2.imwrite(SavePath + os.path.splitext((eventFile))[0] + "SpectrogramAllMarked.png", imgColor)
    #If there are event, then label them
    if len(eventNum)> 0:
        LabelFunctions.eventLabel(os.path.splitext((eventFile))[0], topX, topY, endX, bottomY, eventFile, SavePath,imgHeight,imgLength, eventNum)
    #plt.imshow(imgColor)
    #plt.xticks([]), plt.yticks([])
   # plt.show()
    #cv2.imshow('image',imgColor)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()










#####################################################MAIN###############################################################

def main():
    rootpath = "/home/anoch/Documents/BatSamples/"

    #createSpectrogram(rootpath)
    getFunctions.getAllEvents(rootpath)
    #img = cv2.imread("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000029921020500SpectrogramAllMarked.png",0)
    #verticalScan2(img)
    #GUI(rootpath)
    #ANN_SupervisedBackPro()
    #ANN_Classifier()
    #saveData()
    ######################################FOR THE NEW FEATURE EXTRACTION -- WORKS#########################################
    #poly =  getFrontLineFeature("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000029923979000/Event0.png")
    #print poly, len(poly)
    #poly =  getFrontLineFeature("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000007842559000/Event1.png")
    #print poly, len(poly)
    #bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000041706671000/Event1.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000018916088000/Event5.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event0.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event3.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event4.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event8.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event9.png")

#run main
main()