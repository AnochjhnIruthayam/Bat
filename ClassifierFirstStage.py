__author__ = 'Anochjhn Iruthayam'

## Used to classify whetever the a event is a bat call or not. This process will ease the process of
## classifing species of the bats, since lot of the noise will be removed

import h5py, pybrain, re
from pybrain.datasets import ClassificationDataSet
import BatSpecies as BS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import random, os
from pybrain.tools.customxml.networkwriter import NetworkWriter #To save a network
from pybrain.tools.customxml.networkreader import NetworkReader #To load a network
import numpy as np

# Classifier with the HDF5 interface

def toTime(timePixel):
    imageLength = 5000.0
    return (1000.0/imageLength)*timePixel

def tokFreq(freqPixel):
    imageWidth = 1025.0
    return (250.0/imageWidth)*(imageWidth-freqPixel)


class BinaryClassifier():
    def __init__(self):

        self.pathEventList = []
        self.TrainingSetEventList = []
        self.HDFFile = h5py
        self.Bat = BS.BatSpecies()

    def saveEventPath(self, name):
        self.pathEventList.append(name)

    def initClasissifer(self):
        print "Initilazing HDF5 database"
        self.HDFFile = h5py.File("/home/anoch/Documents/BatOutput/BatData.hdf5")
        self.HDFFile.visit(self.saveEventPath)

    def RemoveTrainingDataFromTestData(self, TrainingSetEventList, TestDataEventList):
        EventPath = []
        for TestSetPath in TestDataEventList:
            FlagAccepted = 0
            for TrainingSetPath in TrainingSetEventList:
                if TrainingSetPath != TestSetPath:
                    FlagAccepted = 1
                else:
                    FlagAccepted = 0
                    break
            if FlagAccepted == 1:
                EventPath.append(TestSetPath)
        return EventPath

    def getSpecificHDFInformation(self, paths, BatID):
            pathcorr = []
            pathcorrImg = []
            for path in paths:
                temp = re.split('/', path)
                # if there are 5 elements in the array, means that this one has an event
                index = 7
                length = 8
                if len(temp) == length and temp[index] == "FeatureDataEvent":
                    #get data from path
                    data = self.HDFFile[path]
                    if data.attrs["BatID"] == BatID:
                        pathcorr.append(path)
                        imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                        pathcorrImg.append(imgPath)

            return pathcorr, pathcorrImg

    ## Scans for all available events
    ## Returns: path where the event data is, target result, path where the event image data is
    def getBatEventHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # as long as it is not other spices, noise and something else;then add, include all events
                if data.attrs["BatID"] != 7 and data.attrs["BatID"] != 8 and data.attrs["BatID"] != 9:
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)

        return pathcorr, pathcorrImg

    ## Scans for all available events
    ## Returns: path where the event data is, target result, path where the event image data is
    def getNoiseEventHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # as long as it is not other spices, noise and something else;then add, include all events
                if data.attrs["BatID"] == 8:
                    #BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)

        return pathcorr, pathcorrImg

    ## Scans for all available events and noises
    ## Returns: path where the event data is, target result, path where the event image data is
    def getAllEventHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # as long as it is not other spice and something else;then add. Include all events and noise
                if data.attrs["BatID"] != 7 and data.attrs["BatID"] != 9: #and data.attrs["BatID"] != 10 and data.attrs["BatID"] != 11 and data.attrs["BatID"] != 12 and data.attrs["BatID"] != 13 and data.attrs["BatID"] != 14 and data.attrs["BatID"] != 15:
                    BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)

        return pathcorr, BatID, pathcorrImg

    def getHDFInfoFromIDList(self, paths, BatIDList):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # as long as it is not other spice and something else;then add. Include all events and noise
                for ID in BatIDList:
                    if ID == data.attrs["BatID"]:
                        BatID.append(data.attrs["BatID"])
                        pathcorr.append(path)
                        imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
                        pathcorrImg.append(imgPath)

        return pathcorr, BatID, pathcorrImg

    def getHDF5Size(self, list):
        pathcorr, BatID, pathcorrImg = self.getAllEventHDFInformation(list)
        return len(pathcorr)

    def getTestRandomDistributedData(self, amount):
        minFreq = []
        maxFreq = []
        Durantion = []
        fl1 = []
        fl2 = []
        fl3 = []
        fl4 = []
        fl5 = []
        fl6 = []
        fl7 = []
        fl8 = []
        fl9 = []
        fl10 = []

        target = []

        pathcorr, BatID, pathcorrImg = self.getAllEventHDFInformation(self.pathEventList)
        EventSize = len(BatID)
        currentEvent = 0
        randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        for i in randomPathIterator:
            data = self.HDFFile[pathcorr[i]]
            minFreq.append(tokFreq(data[0]))
            maxFreq.append(tokFreq(data[1]))
            Durantion.append(toTime(abs(data[2]-data[3])))
            pix0 = data[4]
            pix1 = data[5]
            pix2 = data[6]
            pix3 = data[7]
            pix4 = data[8]
            pix5 = data[9]
            pix6 = data[10]
            pix7 = data[11]
            pix8 = data[12]
            pix9 = data[13]
            pix10 = data[14]

            # Calculate the difference from previous point
            fl1.append(toTime(pix1)-toTime(pix0))
            fl2.append(toTime(pix2)-toTime(pix1))
            fl3.append(toTime(pix3)-toTime(pix2))
            fl4.append(toTime(pix4)-toTime(pix3))
            fl5.append(toTime(pix5)-toTime(pix4))
            fl6.append(toTime(pix6)-toTime(pix5))
            fl7.append(toTime(pix7)-toTime(pix6))
            fl8.append(toTime(pix8)-toTime(pix7))
            fl9.append(toTime(pix9)-toTime(pix8))
            fl10.append(toTime(pix10)-toTime(pix9))

            target.append(BatID[i])


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target


    def getTestData(self, amount, EventPath):
        minFreq = []
        maxFreq = []
        Durantion = []
        fl1 = []
        fl2 = []
        fl3 = []
        fl4 = []
        fl5 = []
        fl6 = []
        fl7 = []
        fl8 = []
        fl9 = []
        fl10 = []
        target = []


        pathcorr, BatID, pathcorrImg = self.getAllEventHDFInformation(EventPath)
        EventSize = len(BatID)
        currentEvent = 0
        for i in range(0, amount):
            data = self.HDFFile[pathcorr[i]]
            minFreq.append(tokFreq(data[0]))
            maxFreq.append(tokFreq(data[1]))
            Durantion.append(toTime(abs(data[2]-data[3])))
            pix0 = data[4]
            pix1 = data[5]
            pix2 = data[6]
            pix3 = data[7]
            pix4 = data[8]
            pix5 = data[9]
            pix6 = data[10]
            pix7 = data[11]
            pix8 = data[12]
            pix9 = data[13]
            pix10 = data[14]

            # Calculate the difference from previous point
            fl1.append(toTime(pix1)-toTime(pix0))
            fl2.append(toTime(pix2)-toTime(pix1))
            fl3.append(toTime(pix3)-toTime(pix2))
            fl4.append(toTime(pix4)-toTime(pix3))
            fl5.append(toTime(pix5)-toTime(pix4))
            fl6.append(toTime(pix6)-toTime(pix5))
            fl7.append(toTime(pix7)-toTime(pix6))
            fl8.append(toTime(pix8)-toTime(pix7))
            fl9.append(toTime(pix9)-toTime(pix8))
            fl10.append(toTime(pix10)-toTime(pix9))


            target.append(BatID[i])


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target


        #Output: returns list random picked test data (features)
    def getDistrubedTestDataRUNVERSION(self, BatIDToAdd):
        #BatIDToAdd.append(8)
        #BatIDToAdd.append(9)
        minFreq = []
        maxFreq = []
        Durantion = []
        fl1 = []
        fl2 = []
        fl3 = []
        fl4 = []
        fl5 = []
        fl6 = []
        fl7 = []
        fl8 = []
        fl9 = []
        fl10 = []
        target = []
        pixelAverage = []
        path = []
        #EventPath = self.RemoveTrainingDataFromTestData(self.TrainingSetEventList, self.pathEventList)
        pathcorr, BatID, pathcorrImg = self.getHDFInfoFromIDList(self.pathEventList, BatIDToAdd)
        EventSize = len(pathcorr)
        currentEvent = 0
        #if EventSize < amount:
        #    amount = EventSize-1
        for i in range(0,EventSize):
            data = self.HDFFile[pathcorr[i]]
            img = self.HDFFile[pathcorrImg[i]]
            pixelAverage.append(img.attrs["AveragePixelValue"])
            minFreq.append(tokFreq(data[0]))
            maxFreq.append(tokFreq(data[1]))
            Durantion.append(toTime(abs(data[2]-data[3])))
            pix0 = data[4]
            pix1 = data[5]
            pix2 = data[6]
            pix3 = data[7]
            pix4 = data[8]
            pix5 = data[9]
            pix6 = data[10]
            pix7 = data[11]
            pix8 = data[12]
            pix9 = data[13]
            pix10 = data[14]
            path.append(pathcorr[i])
            # Calculate the difference from previous point
            fl1.append(toTime(pix1)-toTime(pix0))
            fl2.append(toTime(pix2)-toTime(pix1))
            fl3.append(toTime(pix3)-toTime(pix2))
            fl4.append(toTime(pix4)-toTime(pix3))
            fl5.append(toTime(pix5)-toTime(pix4))
            fl6.append(toTime(pix6)-toTime(pix5))
            fl7.append(toTime(pix7)-toTime(pix6))
            fl8.append(toTime(pix8)-toTime(pix7))
            fl9.append(toTime(pix9)-toTime(pix8))
            fl10.append(toTime(pix10)-toTime(pix9))

            target.append(BatID[i])


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path

    def getTrainingSpeciesDistributedData(self, BatIDToAdd, AmountPerSpecies):
        minFreq = []
        maxFreq = []
        Durantion = []
        fl1 = []
        fl2 = []
        fl3 = []
        fl4 = []
        fl5 = []
        fl6 = []
        fl7 = []
        fl8 = []
        fl9 = []
        fl10 = []

        for BatSpecies in BatIDToAdd:
            print "BatID: " + str(BatSpecies)

            minFreqTemp, maxFreqTemp, DurantionTemp, fl1Temp, fl2Temp, fl3Temp, fl4Temp, fl5Temp, fl6Temp, fl7Temp, fl8Temp, fl9Temp, fl10Temp= self.getTrainingDistributedData(AmountPerSpecies, BatSpecies)
            for i in range(0,len(minFreqTemp)):
                minFreq.append(minFreqTemp[i])
                maxFreq.append(maxFreqTemp[i])
                Durantion.append(DurantionTemp[i])
                fl1.append(fl1Temp[i])
                fl2.append(fl2Temp[i])
                fl3.append(fl3Temp[i])
                fl4.append(fl4Temp[i])
                fl5.append(fl5Temp[i])
                fl6.append(fl6Temp[i])
                fl7.append(fl7Temp[i])
                fl8.append(fl8Temp[i])
                fl9.append(fl9Temp[i])
                fl10.append(fl10Temp[i])


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10

    def getTrainingDistributedData(self, amount, BatID):
        minFreq = []
        maxFreq = []
        Durantion = []
        fl1 = []
        fl2 = []
        fl3 = []
        fl4 = []
        fl5 = []
        fl6 = []
        fl7 = []
        fl8 = []
        fl9 = []
        fl10 = []

        pathcorr, pathcorrImg = self.getSpecificHDFInformation(self.pathEventList, BatID)
        EventSize = len(pathcorr)
        randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        currentEvent = 0
        for i in randomPathIterator:
            #Save in a trainingset, the list of trainingset path. Used to exclude these from testdata
            self.TrainingSetEventList.append(pathcorr[i])
            data = self.HDFFile[pathcorr[i]]
            minFreq.append(tokFreq(data[0]))
            maxFreq.append(tokFreq(data[1]))
            Durantion.append(toTime(abs(data[2]-data[3])))
            pix0 = data[4]
            pix1 = data[5]
            pix2 = data[6]
            pix3 = data[7]
            pix4 = data[8]
            pix5 = data[9]
            pix6 = data[10]
            pix7 = data[11]
            pix8 = data[12]
            pix9 = data[13]
            pix10 = data[14]

            # Calculate the difference from previous point
            fl1.append(toTime(pix1)-toTime(pix0))
            fl2.append(toTime(pix2)-toTime(pix1))
            fl3.append(toTime(pix3)-toTime(pix2))
            fl4.append(toTime(pix4)-toTime(pix3))
            fl5.append(toTime(pix5)-toTime(pix4))
            fl6.append(toTime(pix6)-toTime(pix5))
            fl7.append(toTime(pix7)-toTime(pix6))
            fl8.append(toTime(pix8)-toTime(pix7))
            fl9.append(toTime(pix9)-toTime(pix8))
            fl10.append(toTime(pix10)-toTime(pix9))


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10


    def getTrainingData(self, amount, dataID):
        minFreq = []
        maxFreq = []
        Durantion = []
        fl1 = []
        fl2 = []
        fl3 = []
        fl4 = []
        fl5 = []
        fl6 = []
        fl7 = []
        fl8 = []
        fl9 = []
        fl10 = []

        if dataID == 0:
            pathcorr, pathcorrImg = self.getNoiseEventHDFInformation(self.pathEventList)
        if dataID == 1:
            pathcorr, pathcorrImg = self.getBatEventHDFInformation(self.pathEventList)
        EventSize = len(pathcorr)
        currentEvent = 0

        randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        for i in randomPathIterator:

            #Save in a trainingset, the list of trainingset path. Used to exclude these from testdata
            self.TrainingSetEventList.append(pathcorr[i])
            data = self.HDFFile[pathcorr[i]]
            minFreq.append(tokFreq(data[0]))
            maxFreq.append(tokFreq(data[1]))
            Durantion.append(toTime(abs(data[2]-data[3])))
            pix0 = data[4]
            pix1 = data[5]
            pix2 = data[6]
            pix3 = data[7]
            pix4 = data[8]
            pix5 = data[9]
            pix6 = data[10]
            pix7 = data[11]
            pix8 = data[12]
            pix9 = data[13]
            pix10 = data[14]

            # Calculate the difference from previous point
            fl1.append(toTime(pix1)-toTime(pix0))
            fl2.append(toTime(pix2)-toTime(pix1))
            fl3.append(toTime(pix3)-toTime(pix2))
            fl4.append(toTime(pix4)-toTime(pix3))
            fl5.append(toTime(pix5)-toTime(pix4))
            fl6.append(toTime(pix6)-toTime(pix5))
            fl7.append(toTime(pix7)-toTime(pix6))
            fl8.append(toTime(pix8)-toTime(pix7))
            fl9.append(toTime(pix9)-toTime(pix8))
            fl10.append(toTime(pix10)-toTime(pix9))


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10

    ## Input: Target ID
    ## Converts the Target ID to a number which can be interpreted by the ANN
    ## Returns: returns a modifed target ID
    def convertID(self, ID):
        if ID == 8:
            newID = 0
        else:
            newID = 1
        return  newID


    def runClassifier(self):
        out = []
        true = []
        BatIDToAdd = [1, 2, 3, 5, 6, 10, 11, 12, 14, 8, 9] #1-14 are bats; 8 is noise; 9 is something else
        print "Loading Network.."
        net = NetworkReader.readFrom("FirstStageClassifier.xml")
        print "Loading feature data..."
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path = self.getDistrubedTestDataRUNVERSION(BatIDToAdd)
        SAMPLE_SIZE = len(minFreq)
        for i in range(0, SAMPLE_SIZE):
            ClassifierOutput = net.activate([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]])
            ClassifierOutputID = np.argmax(ClassifierOutput)
            currentTarget = self.convertID(target[i])
            out.append(ClassifierOutputID)
            true.append(currentTarget)
            #MAPPING FROM BATID TO TSC value:
            FSC_value = ClassifierOutputID
            # Metadata Setup, get path and write: TSC = value
            ds = self.HDFFile[path[i]]
            ds.attrs["FSC"] = FSC_value
            ds.attrs["SSC"] = -1
            ds.attrs["TSC"] = -1
        # Close HDF5 file to save to disk. This is also done to make sure the next stage classifier can open the file
        self.HDFFile.close()
        return self.CorrectRatio(out,true)



    def goClassifer(self, iteration, learningrate, momentum, toFile):
        #Clear list
        self.TrainingSetEventList[:] = []
        print "Iteration Count: " + str(iteration)
        #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
        trndata = ClassificationDataSet(13, target=1, nb_classes=2)
        tstdata = ClassificationDataSet(13, target=1, nb_classes=2)
        BatIDToAdd = [1, 2, 3, 5, 6, 10, 11, 12, 14]
        AmountPerSpecies = 30


        print "Adding Bat Events"
        myBat = 1

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingSpeciesDistributedData(BatIDToAdd, AmountPerSpecies)

        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [myBat])

        SAMPLE_SIZE = len(trndata)
        print "Adding Noise Events"
        myBat = 0 # NOISE
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(SAMPLE_SIZE, myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [myBat])

        EventPath = self.RemoveTrainingDataFromTestData(self.TrainingSetEventList, self.pathEventList)
        TraningDataAmount = 5000
        print "Adding test data"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target = self.getTestData(TraningDataAmount, EventPath)
        maxSize = len(minFreq)
        for i in range (0, maxSize):
            tempSave = i % 1000
            if tempSave == 0:
                print i
            myBat = self.convertID(target[i])
            tstdata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [myBat])


        trndata._convertToOneOfMany( )
        tstdata._convertToOneOfMany( )
        print "Number of training patterns: ", len(trndata)
        print "Input and output dimensions: ", trndata.indim, trndata.outdim
        print "Learning Rate: " + str(learningrate)
        print "Momentum: " + str(momentum)
        #print "First sample (input, target, class):"
        #print trndata['input'][0], trndata['target'][0], trndata['class'][0]
        #print "200th sample (input, target, class):"
        #print trndata['input'][100], trndata['target'][100], trndata['class'][100]


        #set up the Feed Forward Network
        HiddenNeurons = 10
        #learningrate = 0.01
        #momentum = 0.1
        weightdecay = 0
        #net = buildNetwork(trndata.indim, HiddenNeurons, trndata.outdim, bias=True, outclass=SoftmaxLayer)
        net = buildNetwork(trndata.indim, HiddenNeurons, trndata.outdim, bias=True, outclass=SigmoidLayer)
        trainer = BackpropTrainer(net, dataset=trndata, momentum=momentum, learningrate=learningrate, verbose=False, weightdecay=weightdecay)
        print "Training data"
        root = "/home/anoch/Dropbox/SDU/10 Semester/MSc Project/Data Results/Master/BinarySpeciesTestMSECE/"
        if toFile:
            #filename = "InputN" + str(trndata.indim) + "HiddenN" + str(HiddenNeurons) + "OutputN" + str(trndata.outdim) + "Momentum"+ str(momentum) + "LearningRate" + str(learningrate) + "Weightdecay" + str(weightdecay)
            baseFileName = "ClassifierBinaryTest"
            filename = baseFileName + "_" + str(iteration) +"_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            folderName = root + "ClassifierBinaryTest_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            f = open(folderName + "/"+ filename + ".txt", 'w')
            value = "Added Bat Species: " + str(BatIDToAdd) + "\n"
            f.write(value)

            value = "Number of bat patterns: " + str(SAMPLE_SIZE) + "\n"
            f.write(value)

            value = "Number of noise patterns: " + str(SAMPLE_SIZE) + "\n"
            f.write(value)

            value = "Number of patterns per species: " + str(AmountPerSpecies) + "\n"
            f.write(value)

            value = "Number of test data: " + str(TraningDataAmount) + "\n"
            f.write(value)

            value = "Input, Hidden and output dimensions: " + str(trndata.indim) + ", " + str(HiddenNeurons) + ", " + str(trndata.outdim) + "\n"
            f.write(value)

            value = "Momentum: " + str(momentum) + "\n"
            f.write(value)

            value = "Learning Rate: " + str(learningrate) + "\n"
            f.write(value)

            value = "Weight Decay: " + str(weightdecay) + "\n"
            f.write(value)

            f.write("Input Activation function: Linear function\n")
            f.write("Hidden Activation function: Sigmoid function\n")
            f.write("Output Activation function: Sigmoid function\n")
        myError = 10000
        maxEpoch = 100
        saveTime = 0
        Newlearningrate = 0
        from collections import deque
        fifo = deque()
        oneTime = True
        for i in range(0, maxEpoch):
            # Train one epoch

            trainer.trainEpochs(10)
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            averageErrorMSE = trainer.testOnData(dataset=tstdata, verbose=False)
            #averageErrorCE = self.CrossEntropyErrorAveraged(net, tstdata)

            #print "CE: " + str(averageErrorCE)
            #print "MSE: " + str(averageErrorMSE)

            #fifo.append(averageError)
            #if len(fifo) == 6:
            #    fifo.popleft()
            #    myError = sum(fifo)/5
            #    print "My Error: " + str(myError)
            #if myError < 0.07 and oneTime:
            #    oneTime = False
            #    saveTime = trainer.totalepochs
            #    print "Changing LearningRate"
            #    Newlearningrate = 0.0005
            #    trainer = BackpropTrainer(net, dataset=trndata, momentum=momentum, learningrate=Newlearningrate, verbose=True, weightdecay=weightdecay)

            #self.CorrectRatio(trainer.testOnClassData(dataset=tstdata), tstdata['class'])


            #"""procentError(out, true) return percentage of mismatch between out and target values (lists and arrays accepted) error= ((out - true)/true)*100"""
            #trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            #tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            #self.CorrectRatio(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

            print("epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,"  test error: %5.2f%%" % tstresult)

            if toFile:
                dataString = str(trainer.totalepochs) + ", " + str(averageErrorMSE) + ", " + str(trnresult) + ", " + str(tstresult) + "\n"
                f.write(dataString)
        NetworkWriter.writeToFile(net, "FirstStageClassifier.xml")
        if toFile:
            results = self.CorrectRatio(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            filename = filename + "_CR"
            folderName = root + "ClassifierBinaryTest_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            result_file = open(folderName + "/"+ filename + ".txt", 'w')
            result_file.write("[TruePostive, TrueNegative, FalsePostive, FalseNegative, CorrectRatio, TrueBats, TrueNonBats]\n")
            result_file.write(str(results) + "\n")
            result_file.write("saveTime: " + str(saveTime)+ "\n")
            result_file.write("New LearningRate: " + str(Newlearningrate))
            result_file.close()
        if toFile:
            f.close()
        print "Done training"


    #Input: A list of the classifier output and the true target
    #Method: calculates the correct ratio based on true and false negative and positive
    #Output: A list of result: [TruePostive, TrueNegative, FalsePostive, FalseNegative, CorrectRatio, TrueBats, TrueNonBats]
    def CorrectRatio(self, out, true):
        TotalTest = len(out)
        TruePostive = 0
        TrueNegative = 0
        FalsePostive = 0
        FalseNegative = 0
        TrueBats = 0
        TrueNonBats = 0

        for i in range(0,TotalTest):
            if true[i] == 1 and out[i] == 1:
                TruePostive += 1

            if true[i] == 0 and out[i] == 0:
                TrueNegative += 1

            if true[i] == 1 and out[i] == 0:
                FalseNegative += 1

            if true[i] == 0 and out[i] == 1:
                FalsePostive += 1

            if true[i] == 1:
                TrueBats += 1

            if true[i] == 0:
                TrueNonBats += 1

        print "True Positive: " + str(TruePostive)
        print "True Negative: " + str(TrueNegative)
        print "False Positive: " + str(FalsePostive)
        print "False Negative: " + str(FalseNegative)
        print "True Bats: " + str(TrueBats)
        print "True Non Bats: " + str(TrueNonBats)
        CorrectRatio = float(TruePostive + TrueNegative) / float(TotalTest) * 100
        print "Correct Ratio: " + str(CorrectRatio)
        results = [TruePostive, TrueNegative, FalsePostive, FalseNegative, CorrectRatio, TrueBats, TrueNonBats]
        return results



    def CrossEntropyErrorAveraged(self, net, dataset):
        import math
        import numpy as np
        outputList = []
        targetList = []
        #Load dataset
        for seq in dataset._provideSequences():
            for input, target in seq:
                #compute net output for given input
                outputList.append(net.activate(input))
                #Load target
                targetList.append(target)
        # Flattern the array to get one stream of data
        outputList = np.array(outputList).flatten()
        targetList = np.array(targetList).flatten()
        #initilize error to zero
        error = 0
        for i in range (0, len(outputList)):
            #avoid zero division
            if targetList[i] != 0 and outputList[i] != 0:
                #Cross entropy error for two classes
                error += float(targetList[i]) * math.log(float(outputList[i])/float(targetList[i])) + (1 - float(targetList[i])) * math.log((1-outputList[i])/(1-float(outputList[i])))
        #Return the averaged error
        return (error*-1)/len(outputList)



