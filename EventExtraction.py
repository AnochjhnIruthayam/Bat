__author__ = 'Anochjhn Iruthayam'

import os
import cv2
import numpy as np
from numpy.polynomial import polynomial as P
import LabelFunctions

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

def getFileList(path, extension):
    sampleList = []
    for file in os.listdir(path):
        if file.endswith(extension):
            sampleList.append(file)
    return sampleList

def month_to_int(month):
    if month == "Jan":
        return "1"
    if month == "Feb":
        return "2"
    if month == "Mar":
        return "3"
    if month == "Apr":
        return "4"
    if month == "May":
        return "5"
    if month == "Jun":
        return "6"
    if month == "Jul":
        return "7"
    if month == "Aug":
        return "8"
    if month == "Sep":
        return "9"
    if month == "Oct":
        return "10"
    if month == "Nov":
        return "11"
    if month == "Dec":
        return "12"
    return "0"


def get_time_for_modified_files(file):
    import os, time
    (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(file)
    mstr = time.ctime(mtime)
    splittedstr = mstr.split(' ')
    mytime = splittedstr[3].split(':')
    day = splittedstr[2]
    month = month_to_int(splittedstr[1])
    year = splittedstr[4]
    hour = mytime[0]
    min = mytime[1]
    sec = mytime[2]

    return day,month,year,hour,min,sec

def hdfgroup(minFreq_pixel,maxFreq_pixel,minMiliSec_pixel,maxMiliSec_pixel,points,eventName, eventNum):
    import h5py
    soundfile = "/home/anoch/Documents/BatSamples/" +eventName + ".s16"
    f = h5py.File("Groups.hdf5")
    #print "Creating Database"

    #Meta data for weather
    temperature = 0
    humidity = 0
    windspeed = 0
    weathercondition = 0
    bat_id = 0


    #testdata = np.random.random(14)

    #Process which zero pads feature data, if there is not enough data
    test_zero = 0
    zero_int = 0
    for point in points:
        test_zero += point
        zero_int += 1
    if test_zero <= 0 or zero_int < 11:
        points = np.zeros(11)

    #Feature data
    data = np.array([minFreq_pixel, maxFreq_pixel,minMiliSec_pixel, maxMiliSec_pixel,points[0],points[1],points[2],points[3],points[4],points[5],points[6],points[7],points[8],points[9],points[10]])
    #print data.shape

    currentfile = soundfile
    day,month,year,hour,min,sec = get_time_for_modified_files(currentfile)
    DirectoryString = year + "/" + month + "/" + day  + "/" + eventName
    e = DirectoryString in f

    if not e:
        out = f.create_group(DirectoryString)
    else:
        out = f[DirectoryString]
    dbName = "FeatureDataEvent_" + str(eventNum)
    out[dbName] = data
    out[dbName].attrs["Temperature"] = temperature
    out[dbName].attrs["Humidity"] = humidity
    out[dbName].attrs["Wind Speed"] = windspeed
    out[dbName].attrs["Weather Condition"] = weathercondition
    out[dbName].attrs["BatID"] = bat_id
    #f.close()

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
    StartX, StartY, end = verticalScan(img)
    for i in range (0,len(StartX)):
        #flag is added to filter out the non bat events by looking at the horizontal axis
        tempSX, tempSY, tempEnd, tempBottom, flag = horizontelScan(img, StartX[i], StartY[i], end[i])
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
