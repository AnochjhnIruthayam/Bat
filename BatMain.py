__author__ = 'Anochjhn Iruthayam'




import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET #phone  home
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import Tkinter as tk
import Image, ImageTk
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from pybrain.structure import SoftmaxLayer
from pybrain.utilities import percentError


import re
#import MultiNEAT as NEAT #not as neat I thought it would be! BASTARD!! Still remember ET to phone home
#from pybrain.tools.shortcuts import buildNetwork

# Set up global frequency band. Set to the range of Bat Calls aka. 13 Khz to 75 KHz into Pixel values
#getHeightMin = 500
#getHeightMax = 980

def getFileList(path, extension):
    sampleList = []
    for file in os.listdir(path):
        if file.endswith(extension):
            sampleList.append(file)
    return sampleList

#Scan the spectrogram with sliding window.
#def scanHorizontal(img)
#    getHeight, getWidth  = img.shape


def findEvent(SearchPath, eventFile, SavePath):
    threshold = 5
    soundFilePath = SearchPath + eventFile
    img = cv2.imread(soundFilePath,0)
    imgColor = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #cv2.circle(imgColor, (50,10), 10, (0,0,255), -1)
    #cv2.imshow('image', img)
    imgHeight,imgLength = img.shape
    topX = []
    topY = []
    endX = []
    bottomY = []
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
        eventLabel(os.path.splitext((eventFile))[0], topX, topY, endX, bottomY, eventFile, SavePath,imgHeight,imgLength, eventNum)
    #plt.imshow(imgColor)
    #plt.xticks([]), plt.yticks([])
   # plt.show()
    #cv2.imshow('image',imgColor)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def verticalScan2(img):
    getHeight, getWidth  = img.shape
    imgColor = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    topX = []
    topY = []
    endX = []
    ColumnCount = 0
    reset = 1
    threshold = 10
    for x in range(0,getWidth):
        if ColumnCount > 60 and reset == 1:
            topX.append(startX)
            topY.append(startY)
            endX.append(x)
            ColumnCount = 0
        for y in range(0,getHeight-4):
            if img.item(y,x) > threshold:
                cv2.circle(imgColor,(x,y), 1, (0,0,255), -1)

                #We encounter a white pixel
                ColumnCount += 1
                #First time encounter then save and set reset to 0
                if reset == 1:
                    startX = x
                    startY = y
                    reset = 0


                break
            elif y == getHeight-5:
                reset = 1
    for i in range (0,len(topX)):
        cv2.line(imgColor,(topX[i],500),(endX[i],500),(255,0,0),5)

    plt.imshow(imgColor)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def verticalScan(img):
    topX = []
    topY = []
    endX = []
    threshold = 10
    ColumnCount = 0
    reset = 1
    getHeight, getWidth  = img.shape
    for x in range(0,getWidth):
        #Make sure that the shape has the right size width
        if ColumnCount > 60 and reset == 1:
            topX.append(startX)
            topY.append(startY)
            endX.append(x)
            ColumnCount = 0
        for y in range(0, getHeight-4):
            if img.item(y,x) > threshold:

                ColumnCount += 1
                #if we find a white pixel for the first time, then save the point
                if reset == 1:
                    startX = x
                    startY = y
                    reset = 0
                ########################### Procentage pixel decision maker
                if y < startY:
                    count = 0
                    length = startY-y
                    for i in range(y, startY):
                        if img.item(i,x) > threshold:
                            count += 1
                    if ((float(count)/length)*100) > 80.0:
                        startY = y
                ############################
                break
            elif y == getHeight-5:# if we reach end of the vertical line, then there is no white pixel
                reset = 1
    return topX, topY, endX

def horizontelScan(img, StartX, StartY, EndX):
    getHeight, getWidth  = img.shape
    topX = StartX
    topY = StartY
    end = EndX
    bottomY = getHeight-4
    count_threshold = 10
    procent_threshold = 80.0
    EventFlag = 0
    threshold = 10
    rowCount = 0
    tempY = 0
    allowed_black_count = 0
    ALLOWED_BLACK = 3
    thisistheend = 0
    first_new_encounter = 1
    for eventY in range(StartY, getHeight-4):
        #this IS recognized as a bat event, set flag to TRUE and save the values
        if rowCount > count_threshold and allowed_black_count < ALLOWED_BLACK:
            EventFlag = 1
            bottomY = tempY
            if allowed_black_count == ALLOWED_BLACK-1:
                return topX, topY, end, bottomY, EventFlag
        elif rowCount < count_threshold or allowed_black_count > ALLOWED_BLACK:
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
    threshold = 5
    getHeight, getWidth  = imgEvent.shape
    for mEventY in range(0,getHeight):
        for mEventX in range (0, getWidth):
            if imgEvent.item(mEventY,mEventX) > threshold:
                if len(X) == 0:
                    tempX = mEventX
                    tempY = mEventY
                if abs(mEventX-tempX) < 6 and abs(mEventY-tempY) < 6:
                    X.append(mEventX)
                    Y.append(mEventY)
                    tempX = mEventX
                    tempY = mEventY
                break
    if len(X) > 5:
        for i in range(0,len(X)):
            cv2.circle(imgColor,(X[i],Y[i]), 1, (0,0,255), -1)
        print np.polyfit(X,Y,7)
        print "\n"
        plt.imshow(imgColor)
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        return X, Y
    else:
        print np.polyfit(X,Y,7)
        return 0, 0


def pixelCount(eventimg_path):
    whiteCount = 0
    imgEvent = cv2.imread(eventimg_path,0)
    threshold = 5
    getHeight, getWidth  = imgEvent.shape
    for mEventY in range(0,getHeight):
        for mEventX in range (0, getWidth):
            if imgEvent.item(mEventY,mEventX) > threshold:
                whiteCount += 1

    return whiteCount

def createSpectrogram(path):
    sampleList = getFileList(path,".s16")
    os.chdir(path)
    for soundFile in sampleList:
        print "Processing " + soundFile + " at channel 1"
        soxCommand = "sox -c 4 -r 500e3 " + soundFile + " -n remix 1 trim 0s 500000s spectrogram -r -m -x 5000 -y 1025 -z 88 -o Spectrogram/" + os.path.splitext((soundFile))[0] + ".png"
        os.system(soxCommand)
    print "Conversion process done!"

def getAllEvents(rootpath):
    SearchPath = rootpath + "Spectrogram/"
    SavePath = rootpath + "SpectrogramMarked/"
    sampleList = getFileList(SearchPath,".png")
    for eventFile in sampleList:
        print "Analyzing " + os.path.splitext((eventFile))[0]
        #topX to endX is the time range, while topX and bottomY is the frequency range
        findEvent(SearchPath, eventFile, SavePath)
    print "Event extraction done!"

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def eventLabel(eventName, topX, topY, endX, bottomY, event_img_path, savePath,imgHeight,imgLength, eventNum):
    top = Element('top')
    comment = Comment("Event information for: " + eventName)
    top.append(comment)

    event = SubElement(top, 'Event')
    for i in eventNum:
        subEvent = SubElement(event, "Event" + str(i))


        bottomFreq = (250.0/imgHeight)*(imgHeight-bottomY[i])
        minFreq = SubElement(subEvent,"minFreq")
        minFreq.text = str(bottomFreq)


        topFreq = (250.0/imgHeight)*(imgHeight-topY[i])
        maxFreq = SubElement(subEvent,"maxFreq")
        maxFreq.text = str(topFreq)

        startMiliSec = (1000.0/imgLength)*topX[i]
        minMiliSec = SubElement(subEvent,"minMiliSec")
        minMiliSec.text = str(startMiliSec)

        endMiliSec = (1000.0/imgLength)*endX[i]
        maxMiliSec = SubElement(subEvent,"maxMiliSec")
        maxMiliSec.text = str(endMiliSec)

        polyfitCof = SubElement(subEvent,"polyfitCof")
        polyfitCof.text = str(pixelCount(savePath + eventName + "/Event" + str(i) + ".png"))#"insert min polyfit Coefficient"

        batID = SubElement(subEvent, "batID")
        batID.text = "NOT CLASSIFIED"

        MainEventSoundFile = SubElement(subEvent,"MainEventSoundFile")
        MainEventSoundFile.text = eventName

        MainEventFile = SubElement(subEvent,"MainEventFile")
        MainEventFile.text = event_img_path

    #print prettify(top)
    tree = ET.ElementTree(top)
    tree.write(savePath + eventName + "/label.xml")

def eventLabelChange(event_label_path, eventNo, batLabel):
    tree = ET.parse(event_label_path)
    root = tree.getroot()

#############################################GUI SETTINGS###############################################################

def event_non_bat(image, event_dir, eventNo, top):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"

    print "NON BAT CALL Registered and written to XML file: " + labelPath
    print "Image: " + image

    tree = ET.parse(labelPath)
    root = tree.getroot()
    lookup = "Event"+str(eventNo)

    #Look for BatID tag
    for BatID in tree.iter(lookup+'/batID'):
        #Mark as NONBAT
        BatID.text = str(0)
    for elem in tree.find("Event/"+lookup):
        if elem.tag == "batID":
            elem.text = str(0)
        print "\t"+elem.tag, elem.text

    tree.write(labelPath)
    top.destroy()

def event_bat(image, event_dir, eventNo, top):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"
    print "BAT CALL Registered and written to XML file: " + labelPath
    print "Image: " + image

    #Read XML file
    tree = ET.parse(labelPath)
    root = tree.getroot()
    lookup = "Event"+str(eventNo)

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "batID":
            elem.text = str(1)
        print "\t"+elem.tag, elem.text

    tree.write(labelPath)
    top.destroy()
    #root.findall(".")
    #print tree.findall("./Event/Event"+str(eventNo)).find("minFreq").text

def keyLeft(event):
    print "ARROW BAT CALL Registered and written to XML file: " + event

def keyRight(event):
    print "ARROW NON BAT CALL Registered and written to XML file: " + event

def get_all_bat_event(rootpath):

    path = rootpath + "SpectrogramMarked/"
    listdirectoryTEMP =  os.listdir(path)
    #eventlist = getFileList(path, ".png")
    listdirectory = []
    list_event = []
    list_event_dir = []
    temppath = ""
    for dir in listdirectoryTEMP:
        if not ".png" in dir:
            if not "~" in dir:
                listdirectory.append(dir)
                #print dir

    for dir in listdirectory:
        eventlist = getFileList(path+dir, ".png")
        for current_event in eventlist:
            list_event.append(current_event)
            list_event_dir.append(dir)
    #print list_event
    #print list_event_dir
    return list_event, list_event_dir


def GUIClassifier(rootpath, image, event_dir, eventNo):

    top = tk.Tk()
    top.minsize(width=300, height=300)
    image = rootpath + "SpectrogramMarked/" + event_dir +"/"+  image
    print image
    #IMAGE
    im = Image.open(image)
    tkimage = ImageTk.PhotoImage(im)
    tk.Label(top, image=tkimage).pack()
    i = 0
    #BUTTONS
    btn_bat = tk.Button(top, text ="BAT", command = lambda: event_bat(image, event_dir, eventNo, top))
    btn_bat.pack(side=tk.TOP)

    nonbat_btn = tk.Button(top, text ="NONBAT", command = lambda: event_non_bat(image, event_dir, eventNo, top))
    nonbat_btn.pack(side=tk.TOP)
    #top.bind('<Left>',keyLeft(event_bat(image, event_dir, eventNo, top)))
    #top.bind('<Right>', keyRight(event_non_bat(image, event_dir, eventNo, top)))
    #top.focus_set()
    top.mainloop()


def GUI(rootpath):
    all_events, event_dir = get_all_bat_event(rootpath)

    for i in range(0, len(all_events)):
        print str(i) +" out of " + str(len(all_events))
        eventNo= ''.join(x for x in all_events[i] if x.isdigit())
        GUIClassifier(rootpath, all_events[i],event_dir[i], eventNo)


#############################################Classified Neural Network##################################################

def ANN_input(event_dir,eventNo):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"
    lookup = "Event"+str(eventNo)

    tree = ET.parse(labelPath)
    root = tree.getroot()

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "minFreq":
            minFreq = elem.text
        if elem.tag == "maxFreq":
            maxFreq = elem.text
        if elem.tag == "minMiliSec":
            minMiliSec = elem.text
        if elem.tag == "maxMiliSec":
            maxMiliSec = elem.text
        if elem.tag == "polyfitCof":
            polyfitCof = elem.text
    MiliSec = float(maxMiliSec)-float(minMiliSec)


    return float(minFreq), float(maxFreq), MiliSec, int(polyfitCof)

def ANN_outout(event_dir,eventNo):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"
    lookup = "Event"+str(eventNo)

    tree = ET.parse(labelPath)
    root = tree.getroot()
    for elem in tree.find("Event/"+lookup):
        if elem.tag == "batID":
            batID = elem.text
    return int(batID)

def ANN_SupervisedBackPro():
    realnonbat = 0
    realthisisBat = 0
    trndata = SupervisedDataSet(4,1) #4 inputs and one output target
    tstdata = SupervisedDataSet(4,1)

    rootpath = "/home/anoch/Documents/BatSamples/"
    event, list_event_dir = get_all_bat_event(rootpath)
    print "Add true event"
    list_event_dir,eventNo = getSample(160,1)
    for i in range(0, len(list_event_dir)):
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo[i])
        print minFreq, maxFreq, MiliSec, pixels
        target = ANN_outout(list_event_dir[i],eventNo[i])
        print target
        trndata.addSample([minFreq, maxFreq, MiliSec, pixels],[target])

    print "Add non true event"

    list_event_dir, eventNo  = getSample(160,0)
    #event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(0, len(list_event_dir)):
        #eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo[i])
        print "Input: " + str(minFreq), str(maxFreq), str(MiliSec), str(pixels)
        target = ANN_outout(list_event_dir[i],eventNo[i])
        print "Output Target: " + str(target)
        trndata.addSample([minFreq, maxFreq, MiliSec, pixels],[target])

    event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(1000, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        #Get input data
        print "Input: " + str(minFreq), str(maxFreq), str(MiliSec), str(pixels)
        print minFreq, maxFreq, MiliSec, pixels
        #get output data
        target = ANN_outout(list_event_dir[i],eventNo)
        print "Output Target: " + str(target)
        #Add the samples to dataset
        tstdata.addSample([minFreq, maxFreq, MiliSec, pixels], [target])
    print "target result"
    print "non bat: " + str(realnonbat)
    print "bat: " + str(realthisisBat)
    net = buildNetwork(4,6,1, bias=True, hiddenclass=SigmoidLayer)
    trainer = BackpropTrainer(net, dataset=trndata, momentum=0.02, learningrate=0.002 , verbose=True, weightdecay=0)
    print "Training data"
    #trainer.trainUntilConvergence()
    for epoch in range(0, 1000):
        error = trainer.train()
        if epoch % 10 == 0:
            print "Epoch: " + str(epoch)
            print "Error: " + str(error)

        #print "error: " + str(error)
        if error < 0.001:
            break
    print "Training Done!"

    nonbat = 0
    thisisBat = 0
    realnonbat = 0
    realthisisBat = 0
    for i in range(1001, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo)
        real_target = ANN_outout(list_event_dir[i],eventNo)
        print minFreq, maxFreq, MiliSec, pixels
        print "Sample: " + list_event_dir[i]
        print "EventNo: " + eventNo
        print net.activate([minFreq,maxFreq,MiliSec,pixels])
        if net.activate([minFreq,maxFreq,MiliSec,pixels]) < 0.5:
            print 0
            nonbat = nonbat + 1
        else:
            print 1
            thisisBat = thisisBat + 1
        if real_target == 1:
            realthisisBat = realthisisBat + 1
        else:
            realnonbat = realnonbat + 1
        print "\n\n"

    print "total result"
    print "non bat: " + str(nonbat)
    print "bat: " + str(thisisBat)
    print "Real result"
    print "non bat: " + str(realnonbat)
    print "bat: " + str(realthisisBat)

#Get a desired number of a desired output
def getSample(sampleAmount, desired_target):
    listEvent_dir = []
    listEvent_No = []
    sampleCount = 0
    rootpath = "/home/anoch/Documents/BatSamples/"
    event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(0, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        #get output data
        target = ANN_outout(list_event_dir[i], eventNo)
        if target == desired_target:
            listEvent_dir.append(list_event_dir[i])
            listEvent_No.append(eventNo)
            sampleCount = sampleCount + 1
        elif target == desired_target:
            listEvent_dir.append(list_event_dir[i])
            listEvent_No.append(eventNo)
            sampleCount = sampleCount + 1
        if sampleAmount < sampleCount:
            break
    return listEvent_dir, listEvent_No

def ANN_Classifier():
    realnonbat = 0
    realthisisBat = 0


    #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
    trndata = ClassificationDataSet(4,target=1, nb_classes=2)
    tstdata = ClassificationDataSet(4,target=1, nb_classes=2)

    rootpath = "/home/anoch/Documents/BatSamples/"
    #get all events
    #event, list_event_dir = get_all_bat_event(rootpath)
    print "Add true event"
    list_event_dir,eventNo = getSample(160,1)
    for i in range(0, len(list_event_dir)):
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo[i])
        print minFreq, maxFreq, MiliSec, pixels
        target = ANN_outout(list_event_dir[i],eventNo[i])
        print target
        trndata.addSample([minFreq, maxFreq, MiliSec, pixels],[target])

    print "Add nontrue event"

    list_event_dir, eventNo  = getSample(160,0)
    #event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(0, len(list_event_dir)):
        #eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo[i])
        print "Input: " + str(minFreq), str(maxFreq), str(MiliSec), str(pixels)
        target = ANN_outout(list_event_dir[i],eventNo[i])
        print "Output Target: " + str(target)
        trndata.addSample([minFreq, maxFreq, MiliSec, pixels],[target])

    event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(1000, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        #Get input data
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo)
        print "Input: " + str(minFreq), str(maxFreq), str(MiliSec), str(pixels)
        print minFreq, maxFreq, MiliSec, pixels
        #get output data
        target = ANN_outout(list_event_dir[i],eventNo)
        print "Output Target: " + str(target)
        #Add the samples to dataset
        tstdata.addSample([minFreq, maxFreq, MiliSec, pixels], [target])
    #print "Add training set"
    """
    for i in range(0, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        #Get input data
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo)
        print minFreq, maxFreq, MiliSec, pixels
        #get output data
        target = ANN_outout(list_event_dir[i],eventNo)
        print target
        if target == 1:
            realthisisBat = realthisisBat + 1
        else:
            realnonbat = realnonbat + 1
        #Add the samples to dataset
        DS.addSample([minFreq, maxFreq, MiliSec, pixels], [target])
    """
    # we want a 75% training data and 25% test data

    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( )
    #Print information out
    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0], trndata['class'][0]
    print "200th sample (input, target, class):"
    print trndata['input'][200], trndata['target'][200], trndata['class'][200]
    #print "Over all true results"
    #print "non bat: " + str(realnonbat)
    #print "bat: " + str(realthisisBat)

    #set up the Feed Forward Network
    net = buildNetwork(trndata.indim,3,trndata.outdim, bias=True, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, learningrate=0.01 , verbose=True, weightdecay=0)
    print "Training data"
    #trainer.trainUntilConvergence()
    """
    for epoch in range(0, 1000):
        error = trainer.train()
        #if epoch % 10 == 0:
        print "Epoch: " + str(epoch)
        print "Error: " + str(error)
        #print "error: " + str(error)
        if error < 0.001:
             break
    """
    for i in range(0,100):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(),
                                 trndata['class'])
        tstresult = percentError(trainer.testOnClassData(
                                 dataset=tstdata), tstdata['class'])
        print("epoch: %4d" % trainer.totalepochs,
              "  train error: %5.2f%%" % trnresult,
              "  test error: %5.2f%%" % tstresult)

    nonbat = 0
    thisisBat = 0
    realnonbat = 0
    realthisisBat = 0
    event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(500, 1000):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo)
        real_target = ANN_outout(list_event_dir[i],eventNo)
        print minFreq, maxFreq, MiliSec, pixels
        print "Sample: " + list_event_dir[i]
        print "EventNo: " + eventNo
        print "Real Target: " + str(real_target)
        pred = net.activate([minFreq,maxFreq,MiliSec,pixels])
        print pred
        if real_target == 1:
            realthisisBat = realthisisBat + 1
        else:
            realnonbat = realnonbat + 1
        print "\n\n"

    print "total result"
    print "non bat: " + str(nonbat)
    print "bat: " + str(thisisBat)
    print "Real result"
    print "non bat: " + str(realnonbat)
    print "bat: " + str(realthisisBat)


def saveData():
    print "Saving to file.."
    rootpath = "/home/anoch/Documents/BatSamples/"
    event, list_event_dir = get_all_bat_event(rootpath)
    text_file = open("/home/anoch/Documents/Output.txt", "w")
    for i in range(0, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, pixels = ANN_input(list_event_dir[i],eventNo)
        target = ANN_outout(list_event_dir[i],eventNo)
        data = str(minFreq) +"," + str(maxFreq)+"," + str(MiliSec)+"," + str(pixels)+"," +str(target) + "\n"
        text_file.write(data)
    text_file.close()
    print "Save Done!"






#####################################################MAIN###############################################################

def main():
    rootpath = "/home/anoch/Documents/BatSamples/"

    #createSpectrogram(rootpath)
    #getAllEvents(rootpath)
    #img = cv2.imread("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000029921020500SpectrogramAllMarked.png",0)
    #verticalScan2(img)
    #GUI(rootpath)
    #ANN_SupervisedBackPro()
    #ANN_Classifier()
    #saveData()
    #bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000016611196000/Event2.png")
    bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000016807303500/Event1.png")
    bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000016807303500/Event0.png")
    bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000016807303500/Event5.png")
#run main
main()