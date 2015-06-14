__author__ = 'Anochjhn Iruthayam'
import h5py, pybrain, re
from pybrain.datasets import ClassificationDataSet
import BatSpecies as BS
import numpy as np
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import random, os
from pybrain.tools.customxml.networkwriter import NetworkWriter #To save a network
from pybrain.tools.customxml.networkreader import NetworkReader #To load a network

# Classifier with the HDF5 interface

def toTime(timePixel):
    imageLength = 5000.0
    return (1000.0/imageLength)*timePixel

def tokFreq(freqPixel):
    imageWidth = 1025.0
    return (250.0/imageWidth)*(imageWidth-freqPixel)


class Classifier():
    def __init__(self):
        self.pathEventList = []
        self.TrainingSetEventList = []
        self.HDFFile = h5py
        self.Bat = BS.BatSpecies()
        self.ConfusionMatrix = 0

    def saveEventPath(self, name):
        self.pathEventList.append(name)

    def initClasissifer(self, filename):
        print "Initilazing HDF5 database"
        self.HDFFile = h5py.File(filename)
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

    def pixelCount(self, img):
        height, length = img.shape
        pixelValue = 0
        count = 0
        for x in range(0, length):
            for y in range(0, height):
                if img[y][x] > 1:
                    pixelValue += img[y][x]
                    count += 1
        averagePixel = float(pixelValue)/float(count)
        return averagePixel


    def getHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                data = self.HDFFile[path]
                #Exclude certain classes/groups 4 because not enough data,
                #if data.attrs["BatID"] != 0 and data.attrs["BatID"] != 4 and data.attrs["BatID"] != 7 and data.attrs["BatID"] != 8 and data.attrs["BatID"] != 9 and data.attrs["BatID"] != 10 and data.attrs["BatID"] != 11 and data.attrs["BatID"] != 12 and data.attrs["BatID"] != 13 and data.attrs["BatID"] != 14 and data.attrs["BatID"] != 15:
                # We are NOT including 0: not classified, 4: too low data, 13: too low data, 15: too low data, 7: opther species
                if data.attrs["BatID"] != 0 and data.attrs["BatID"] != 4 and data.attrs["BatID"] != 13 and data.attrs["BatID"] != 15 and data.attrs["BatID"] != 7:
                    BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)


        return pathcorr, BatID, pathcorrImg

    ## Scans for all available events
    ## Returns: path where the event data is, target result, path where the event image data is
    def getBatEventHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # as long as it is not other spices, noise and something else;then add, include all events
                if data.attrs["BatID"] != 0 and data.attrs["BatID"] != 7 and data.attrs["BatID"] != 8 and data.attrs["BatID"] != 9:
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)

        return pathcorr, pathcorrImg

    def getSpecificHDFInformation(self, paths, BatID):
        pathcorr = []
        pathcorrImg = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                #get data from path
                data = self.HDFFile[path]
                if data.attrs["BatID"] == BatID:
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
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
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
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
                if data.attrs["BatID"] != 7 and data.attrs["BatID"] != 4 and data.attrs["BatID"] != 13 and data.attrs["BatID"] != 15:# and data.attrs["BatID"] != 9 and data.attrs["BatID"] != 8:
                    BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
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

    ## Scans for all available events
    ## Returns: path where the event data is, target result, path where the event image data is
    def getHDFInformationToAddPixelInfo(self, paths):
        pathcorrImg = []
        BatID = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                #data = self.HDFFile[path]
                # as long as it is not other spices, noise and something else;then add, include all events
                #BatID.append(data.attrs["BatID"])
                imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
                pathcorrImg.append(imgPath)

        return pathcorrImg

    def getHDFSSC(self, paths):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        SSC_VALUE = 1
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # If the first stage classisfier results is 1, then continue. Also exclude some bat calls
                if 4 != data.attrs["BatID"] and 7 != data.attrs["BatID"] and 13 != data.attrs["BatID"] and 15 != data.attrs["BatID"]:
                    if SSC_VALUE == data.attrs["SSC"]:
                        BatID.append(data.attrs["BatID"])
                        pathcorr.append(path)
                        imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
                        pathcorrImg.append(imgPath)

        return pathcorr, BatID, pathcorrImg


    #Output: returns list random picked test data (features)
    def getDistrubedTestDataRUNVERSION(self):
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
        pathcorr, BatID, pathcorrImg = self.getHDFSSC(self.pathEventList)
        EventSize = len(pathcorr)
        currentEvent = 0
        #if EventSize < amount:
        #    amount = EventSize-1
        for i in range(0, EventSize):
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


    def getTestData(self, amount):
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

        pathcorr, BatID, pathcorrImg = self.getHDFInformation(self.pathEventList)
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

    #Output: returns list random picked test data (features)
    def getDistrubedTestData(self, amount, BatIDToAdd):
        BatIDToAdd.append(8)
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
        EventPath = self.RemoveTrainingDataFromTestData(self.TrainingSetEventList, self.pathEventList)
        pathcorr, BatID, pathcorrImg = self.getHDFInfoFromIDList(EventPath, BatIDToAdd)
        EventSize = len(BatID)
        currentEvent = 0
        if EventSize < amount:
            amount = EventSize-1
        randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        for i in randomPathIterator:
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


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target




    #Adds all the needed species in one
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
        pixelAverage = []
        target = []

        for BatSpecies in BatIDToAdd:
            print "BatID: " + str(BatSpecies)

            minFreqTemp, maxFreqTemp, DurantionTemp, fl1Temp, fl2Temp, fl3Temp, fl4Temp, fl5Temp, fl6Temp, fl7Temp, fl8Temp, fl9Temp, fl10Temp, pixelAverageTemp = self.getTrainingDistributedData(AmountPerSpecies, BatSpecies)
            #minFreqTemp, maxFreqTemp, DurantionTemp, fl1Temp, fl2Temp, fl3Temp, fl4Temp, fl5Temp, fl6Temp, fl7Temp, fl8Temp, fl9Temp, fl10Temp= self.getTrainingSequenceData(AmountPerSpecies, BatSpecies)
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
                pixelAverage.append(pixelAverageTemp[i])
                target.append(BatSpecies)

        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target

    #Output: returns list of traning feautures in a random order
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
        pixelAverage = []
        pathcorr, pathcorrImg = self.getSpecificHDFInformation(self.pathEventList, BatID)
        EventSize = len(pathcorr)
        randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        currentEvent = 0
        for i in randomPathIterator:
            data = self.HDFFile[pathcorr[i]]
            img = self.HDFFile[pathcorrImg[i]]
            pixelAverage.append(img.attrs["AveragePixelValue"])
            self.TrainingSetEventList.append(pathcorr[i])
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


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage

    #Output: returns list of traning feautures in a sequence
    def getTrainingSequenceData(self, amount, BatID):
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
        #randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        #currentEvent = 0
        for i in range(0,amount):
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

    def getDistributedData(self, amount, ID):

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
        pixelAverage = []

        pathcorr, pathcorrImg = self.getSpecificHDFInformation(self.pathEventList, ID)

        EventSize = len(pathcorr)
        currentEvent = 0

        randomPathIterator = random.sample(xrange(0,EventSize-1), amount)
        for i in randomPathIterator:
            data = self.HDFFile[pathcorr[i]]
            img = self.HDFFile[pathcorrImg[i]]
            pixelAverage.append(img.attrs["AveragePixelValue"])
            self.TrainingSetEventList.append(pathcorr[i])
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


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage


    def convertIDAll(self, ID):

        if ID == 1:
            newID = 0
        elif ID == 2:
            newID = 1
        elif ID == 3:
            newID = 2
        elif ID == 5:
            newID = 3
        elif ID == 6:
            newID = 4
        elif ID == 10:
            newID = 5
        elif ID == 11:
            newID = 6
        elif ID == 12:
            newID = 7
        elif ID == 14:
            newID = 8
        #this is for noise
        elif ID == 8:
            newID = 9
        #this is for something else
        elif ID == 9:
            newID = 10
        else:
            print "Could not assign the ID " + str(ID) + " to newID"


        return  newID


    def convertIDSingle(self, ID):

        if ID == 1:
            newID = 0
        elif ID == 2:
            newID = 1
        elif ID == 3:
            newID = 2
        elif ID == 5:
            newID = 3
        elif ID == 6:
            newID = 4
        #this is for noise
        elif ID == 8:
            newID = 5

        #this is for something else. Classify multiple calls as something else
        elif ID == 10:
            newID = 6
        elif ID == 11:
            newID = 6
        elif ID == 12:
            newID = 6
        elif ID == 14:
            newID = 6
        elif ID == 9:
            newID = 6
        else:
            print "Could not assign the ID " + str(ID) + " to newID"


        return newID


    #assign 0 to noise, 1 to single call, 2 to multiple calls
    def convertIDlow(self, ID):

        if ID == 1:
            newID = 1
        elif ID == 2:
            newID = 1
        elif ID == 3:
            newID = 1
        elif ID == 5:
            newID = 1
        elif ID == 6:
            newID = 1
        elif ID == 10:
            newID = 2
        elif ID == 11:
            newID = 2
        elif ID == 12:
            newID = 2
        elif ID == 14:
            newID = 2
        #this is for noise
        elif ID == 8:
            newID = 0
        #this is for something else
        elif ID == 9:
            newID = 10
        else:
            print "Could not assign the ID " + str(ID) + " to newID"


        return newID

    def convertID2(self, ID):

        if ID == 1:
            newID = 0
        elif ID == 2:
            newID = 1
        elif ID == 3:
            newID = 2
        elif ID == 5:
            newID = 3
        elif ID == 6:
            newID = 4
        #this is for noise
        elif ID == 8:
            newID = 5
        elif ID == 9:
            newID = 6

        else:
            print "Could not assign the ID " + str(ID) + " to newID"


        return newID

    def pixelFix(self):
        pathcorrImg = self.getHDFInformationToAddPixelInfo(self.pathEventList)
        TotalEvents = len(pathcorrImg)
        for i in range (0, TotalEvents):
            print "Calculating Average pixel for event " + str(i) + " out of " + str(TotalEvents)
            img = self.HDFFile[pathcorrImg[i]]
            averagePixelValue = self.pixelCount(img)
            img.attrs["AveragePixelValue"] = averagePixelValue

    def printy(self, s):
        from scipy import mean
        if ((s._num_updates * s.batch_size < 100
             and s._num_updates % (20 / s.batch_size) == 0)
            or s._num_updates % (100 / s.batch_size) == 0):
            print s._num_updates * s.batch_size, #s.bestParameters,
            s.provider.nextSamples(4)
            print mean(s.provider.currentLosses(s.bestParameters))
            #s.provider.nextSamples(1)

    def runClassifier(self):
        out = []
        true = []
        #SingleBatIDToAdd = [1, 2, 3, 5, 6] # for single
        Correct = 0
        print "Loading Network.."
        net = NetworkReader.readFrom("ThirdStageClassifier.xml")
        print "Loading feature data with SSC = 1 (Single call type)"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path = self.getDistrubedTestDataRUNVERSION()
        SAMPLE_SIZE = len(minFreq)
        for i in range(0, SAMPLE_SIZE):
            ClassifierOutput= net.activate([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i]])

            ClassifierOutputID = np.argmax(ClassifierOutput)
            currentTarget = self.convertIDSingle(target[i])
            out.append(ClassifierOutputID)
            true.append(currentTarget)

            #MAPPING FROM BATID TO TSC value:
            TSC_value = ClassifierOutputID
            # Metadata Setup, get path and write: TSC = value
            ds = self.HDFFile[path[i]]
            ds.attrs["TSC"] = TSC_value
        self.HDFFile.flush()
        self.HDFFile.close()
        return self.CorrectRatio(out, true)
        #self.ConfusionMatrix = self.CorrectRatio(out, true)
        #return self.ConfusionMatrix



    def goClassifer(self, iteration, learningrate, momentum, toFile):
        self.TrainingSetEventList[:] = []
        print "Iteration Count: " + str(iteration)
        #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
        trndata = ClassificationDataSet(14, nb_classes=7)
        tstdata = ClassificationDataSet(14, nb_classes=7)
        SAMPLE_SIZE = 100
        AmountPerSpecies = 100
        SingleBatIDToAdd = [1, 2, 3, 5, 6] # for single
        MultiBatIDToAdd = [10, 11, 12, 14]# for multi
        AddBatIDToAdd = [1, 2, 3, 5, 6]
        AddSingleMulti = [1, 2, 3, 5, 6,10, 11, 12, 14]
        TraningDataAmount = 5000

        print "Adding Bat Single Species Events"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target = self.getTrainingSpeciesDistributedData(SingleBatIDToAdd, AmountPerSpecies)

        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            #trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i] ], [1]) #self.convertID(target[i])
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i] ], [self.convertIDSingle(target[i])]) #self.convertID(target[i])

        #print "Adding Bat Multi Species Events"
        #minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target = self.getTrainingSpeciesDistributedData(MultiBatIDToAdd, AmountPerSpecies)

        #SAMPLE_SIZE = len(minFreq)
        #for i in range (0, SAMPLE_SIZE):
        #    trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i] ], [2])


        print "Adding noise events"
        NoiseID = 8
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage = self.getDistributedData(AmountPerSpecies, NoiseID)
        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i] ], [self.convertIDSingle(NoiseID)]) #self.convertID(NoiseID)

        print "Adding something else events"
        SomethingElseID = 9
        SEAmount = 20
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage = self.getDistributedData(SEAmount, SomethingElseID)
        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i] ], [self.convertIDSingle(SomethingElseID)])

        # Try to put all multievent in the something else event
        print "Adding something else events"
        SomethingElseID = 9
        BatIDToAdd2 = [10, 11, 12, 14]
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target = self.getTrainingSpeciesDistributedData(BatIDToAdd2, SEAmount)
        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i] ], [self.convertIDSingle(SomethingElseID)])


        print "Adding test data"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target = self.getDistrubedTestData(TraningDataAmount, SingleBatIDToAdd)
        maxSize = len(minFreq)
        for i in range (0, maxSize):
            tstdata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i]], [ self.convertIDSingle (target[i]) ])

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
        #from datainterface import ModuleWrapper, ClassificationModuleWrapper
        #from sgd import SGD

        net = buildNetwork(trndata.indim, HiddenNeurons, trndata.outdim, bias=True, outclass=SoftmaxLayer)
        #p0 = net.params.copy()

        #provider = ClassificationModuleWrapper(trndata, net, shuffling=False)
        #algo = SGD(provider, net.params.copy(), callback=self.printy, learning_rate=learningrate, momentum=momentum)
        #print '\n' * 2
        #print 'SGD-CE'
        #algo.run(1000)
        trainer = BackpropTrainer(net, dataset=trndata, momentum=momentum, learningrate=learningrate, verbose=False, weightdecay=weightdecay)
        #raw_input("Press Enter to continue...")
        print "Training data"
        if toFile:
            #filename = "InputN" + str(trndata.indim) + "HiddenN" + str(HiddenNeurons) + "OutputN" + str(trndata.outdim) + "Momentum"+ str(momentum) + "LearningRate" + str(learningrate) + "Weightdecay" + str(weightdecay)
            root = "/home/anoch/Dropbox/SDU/10 Semester/MSc Project/Data Results/Master/BinarySpeciesTestMSE/"
            filename = "ClassifierSpeciesTest_" + str(iteration) +"_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            folderName = root + "ClassifierSpeciesTest_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            f = open(folderName + "/"+ filename + ".txt", 'w')

            value = "Added Bat Species: " + str(AddBatIDToAdd) + "\n"
            f.write(value)

            value = "Number of bat patterns: " + str(len(trndata)) + "\n"
            f.write(value)

            value = "Number of noise patterns: " + str(AmountPerSpecies) + "\n"
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
            f.write("Output Activation function: Softmax function\n")

        maxEpoch = 100
        for i in range(0,maxEpoch):
            # Train one epoch
            trainer.trainEpochs(10)
            averageError = trainer.testOnData(dataset=tstdata, verbose=False)

            #averageCEE = self.CrossEntropyErrorAveraged(net, tstdata)
            #print "Average Cross Entropy Error: " + str(averageCEE)
            #print "Mean Square Error: " + str(averageError)

            #"""procentError(out, true) return percentage of mismatch between out and target values (lists and arrays accepted) error= ((out - true)/true)*100"""
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

            print("epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,"  test error: %5.2f%%" % tstresult)

            if tstresult < 27.0:
                raw_input("Press Enter to continue...")
                break

            if toFile:
                dataString = str(trainer.totalepochs) + ", " + str(averageError) + ", " + str(trnresult) + ", " + str(tstresult) + "\n"
                f.write(dataString)
        NetworkWriter.writeToFile(net, "ThirdStageClassifier.xml")
        if toFile:
            import numpy as np
            f.close()
            ConfusionMatrix, BatTarget = self.CorrectRatio(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            filename = filename+ "_CR"
            result_file = open(folderName + "/"+ filename + ".txt", 'w')
            result_file.write("[Species]")
            result_file.write(str(BatTarget))
            result_file.write(str(ConfusionMatrix))
            np.savetxt(folderName + "/"+ filename+".csv", ConfusionMatrix, delimiter=",")
            result_file.close()
        self.CorrectRatio(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        print "Done training"

    #Input: A list of the classifier output and the true target
    #Method: calculates the correct ratio based on true and false negative and positive
    #Output: A list of result: [TruePostive, FalsePostive, CorrectRatio], [BatCount]
    def CorrectRatio(self, out, true):

        import numpy as np
        #initilaze with zero
        ConfusionMatrix = np.zeros((7,7))
        #int64 datatype
        ConfusionMatrix = np.array(ConfusionMatrix, dtype=np.int64)
        TotalTest = len(out)
        BatTarget = [0, 0, 0, 0, 0, 0, 0]

        for i in true:
            BatTarget[i] += 1

        out = np.array(out).flatten()
        true = np.array(true).flatten()
        for i in range(0,TotalTest):
            classifierOut = out[i]
            targetOut = true[i]
            ConfusionMatrix[targetOut][classifierOut] += 1

        print BatTarget
        print "\n"
        print ConfusionMatrix
        return ConfusionMatrix


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
                error += float(targetList[i]) * math.log(float(outputList[i])/float(targetList[i]))
        #Return the averaged error
        return (error*-1)/len(outputList)


