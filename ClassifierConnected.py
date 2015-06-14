__author__ = 'Anochjhn Iruthayam'
import h5py
import numpy as np
import pybrain, re
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import random, os
from pybrain.tools.customxml.networkwriter import NetworkWriter #To save a network
from pybrain.tools.customxml.networkreader import NetworkReader #To load a network
import getFunctions

class ClassifierConnected():
    def __init__(self):

        self.pathEventList = []
        self.TrainingSetEventList = []
        self.HDFFile = h5py
        self.ConfusionMatrix = 0
    def saveEventPath(self, name):
        self.pathEventList.append(name)

    def initClasissifer(self, filename):
        print "Initilazing HDF5 database"
        self.HDFFile = h5py.File(filename)
        self.HDFFile.visit(self.saveEventPath)


################################################### FSC ################################################################
    def getHDFInfoFromIDListFSC(self, paths, BatIDList):
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


    def getDistrubedTestDataRUNVERSIONFSC(self, BatIDToAdd):
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
        pathcorr, BatID, pathcorrImg = self.getHDFInfoFromIDListFSC(self.pathEventList, BatIDToAdd)
        EventSize = len(pathcorr)
        currentEvent = 0
        #if EventSize < amount:
        #    amount = EventSize-1
        for i in range(0,EventSize):
            data = self.HDFFile[pathcorr[i]]
            img = self.HDFFile[pathcorrImg[i]]
            pixelAverage.append(img.attrs["AveragePixelValue"])
            minFreq.append(getFunctions.tokFreq(data[0]))
            maxFreq.append(getFunctions.tokFreq(data[1]))
            Durantion.append(getFunctions.toTime(abs(data[2]-data[3])))
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
            fl1.append(getFunctions.toTime(pix1)-getFunctions.toTime(pix0))
            fl2.append(getFunctions.toTime(pix2)-getFunctions.toTime(pix1))
            fl3.append(getFunctions.toTime(pix3)-getFunctions.toTime(pix2))
            fl4.append(getFunctions.toTime(pix4)-getFunctions.toTime(pix3))
            fl5.append(getFunctions.toTime(pix5)-getFunctions.toTime(pix4))
            fl6.append(getFunctions.toTime(pix6)-getFunctions.toTime(pix5))
            fl7.append(getFunctions.toTime(pix7)-getFunctions.toTime(pix6))
            fl8.append(getFunctions.toTime(pix8)-getFunctions.toTime(pix7))
            fl9.append(getFunctions.toTime(pix9)-getFunctions.toTime(pix8))
            fl10.append(getFunctions.toTime(pix10)-getFunctions.toTime(pix9))

            target.append(BatID[i])


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path

    ## Input: Target ID
    ## Converts the Target ID to a number which can be interpreted by the ANN
    ## Returns: returns a modifed target ID
    def convertIDFSC(self, ID):
        if ID == 8:
            newID = 0
        else:
            newID = 1
        return  newID

    def runFirstStageClassifier(self):
        out = []
        true = []
        BatIDToAdd = [1, 2, 3, 5, 6, 10, 11, 12, 14, 8, 9] #1-14 are bats; 8 is noise; 9 is something else
        print "Loading Network.."
        net = NetworkReader.readFrom("C:\Users\Anoch\PycharmProjects\BatClassification\FirstStageClassifier.xml")
        print "Loading feature data..."
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path = self.getDistrubedTestDataRUNVERSIONFSC(BatIDToAdd)
        SAMPLE_SIZE = len(minFreq)
        for i in range(0, SAMPLE_SIZE):
            ClassifierOutput = net.activate([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]])
            ClassifierOutputID = np.argmax(ClassifierOutput)
            currentTarget = self.convertIDFSC(target[i])
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
        self.HDFFile.flush()

################################################### SSC ################################################################
    def getHDFInfoFromIDListSSC(self, paths):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        FSC_VALUE = 1
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # If the first stage classisfier results is 1, then continue. Also exclude some bat calls since they are not used as training
                if 4 != data.attrs["BatID"] and 7 != data.attrs["BatID"] and 13 != data.attrs["BatID"] and 15 != data.attrs["BatID"]:
                    if FSC_VALUE == data.attrs["FSC"]:
                        BatID.append(data.attrs["BatID"])
                        pathcorr.append(path)
                        imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + temp[5] + "/" + temp[6] + "/" + "ArrayImgEvent"
                        pathcorrImg.append(imgPath)

        return pathcorr, BatID, pathcorrImg

    #Output: returns list random picked test data (features)
    def getDistrubedTestDataRUNVERSIONSSC(self):
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
        pathcorr, BatID, pathcorrImg = self.getHDFInfoFromIDListSSC(self.pathEventList)
        EventSize = len(pathcorr)
        currentEvent = 0
        #if EventSize < amount:
        #    amount = EventSize-1
        for i in range(0, EventSize):
            data = self.HDFFile[pathcorr[i]]
            img = self.HDFFile[pathcorrImg[i]]
            pixelAverage.append(img.attrs["AveragePixelValue"])
            minFreq.append(getFunctions.tokFreq(data[0]))
            maxFreq.append(getFunctions.tokFreq(data[1]))
            Durantion.append(getFunctions.toTime(abs(data[2]-data[3])))
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
            fl1.append(getFunctions.toTime(pix1)-getFunctions.toTime(pix0))
            fl2.append(getFunctions.toTime(pix2)-getFunctions.toTime(pix1))
            fl3.append(getFunctions.toTime(pix3)-getFunctions.toTime(pix2))
            fl4.append(getFunctions.toTime(pix4)-getFunctions.toTime(pix3))
            fl5.append(getFunctions.toTime(pix5)-getFunctions.toTime(pix4))
            fl6.append(getFunctions.toTime(pix6)-getFunctions.toTime(pix5))
            fl7.append(getFunctions.toTime(pix7)-getFunctions.toTime(pix6))
            fl8.append(getFunctions.toTime(pix8)-getFunctions.toTime(pix7))
            fl9.append(getFunctions.toTime(pix9)-getFunctions.toTime(pix8))
            fl10.append(getFunctions.toTime(pix10)-getFunctions.toTime(pix9))

            target.append(BatID[i])
        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path
    
     #assign 0 to noise, 1 to single call, 2 to multiple calls
    def convertIDMultiSingleSSC(self, ID):

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
        # ASSIGN SOMETHING ELSE TO SINGLE CALL CLASS
        elif ID == 9:
            newID = 3
        else:
            print "Could not assign the ID " + str(ID) + " to newID"


        return newID

    def runSecondStageClassifier(self):
        out = []
        true = []
        #BatIDToAdd = [1, 2, 3, 5, 6, 10, 11, 12, 14, 8, 9] #1-14 are bats; 8 is noise; 9 is something else
        print "Loading Network.."
        net = NetworkReader.readFrom("C:\Users\Anoch\PycharmProjects\BatClassification\SecondStageClassifier.xml")
        print "Loading feature data with FSC = 1 (Bat calls)"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path = self.getDistrubedTestDataRUNVERSIONSSC()
        SAMPLE_SIZE = len(minFreq)
        for i in range(0, SAMPLE_SIZE):
            ClassifierOutput = net.activate([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i]])
            ClassifierOutputID = np.argmax(ClassifierOutput)
            currentTarget = self.convertIDMultiSingleSSC(target[i])
            out.append(ClassifierOutputID)
            true.append(currentTarget)
            #MAPPING FROM BATID TO TSC value:
            SSC_value = ClassifierOutputID
            # Metadata Setup, get path and write: TSC = value
            ds = self.HDFFile[path[i]]
            ds.attrs["SSC"] = SSC_value
        # Close HDF5 file to save to disk. This is also done to make sure the next stage classifier can open the file
        self.HDFFile.flush()

################################################### TSC ################################################################

    def getHDFInfoFromIDListTSC(self, paths):
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
    def getDistrubedTestDataRUNVERSIONTSC(self):
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
        pathcorr, BatID, pathcorrImg = self.getHDFInfoFromIDListTSC(self.pathEventList)
        EventSize = len(pathcorr)
        currentEvent = 0
        #if EventSize < amount:
        #    amount = EventSize-1
        for i in range(0, EventSize):
            data = self.HDFFile[pathcorr[i]]
            img = self.HDFFile[pathcorrImg[i]]
            pixelAverage.append(img.attrs["AveragePixelValue"])
            minFreq.append(getFunctions.tokFreq(data[0]))
            maxFreq.append(getFunctions.tokFreq(data[1]))
            Durantion.append(getFunctions.toTime(abs(data[2]-data[3])))
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
            fl1.append(getFunctions.toTime(pix1)-getFunctions.toTime(pix0))
            fl2.append(getFunctions.toTime(pix2)-getFunctions.toTime(pix1))
            fl3.append(getFunctions.toTime(pix3)-getFunctions.toTime(pix2))
            fl4.append(getFunctions.toTime(pix4)-getFunctions.toTime(pix3))
            fl5.append(getFunctions.toTime(pix5)-getFunctions.toTime(pix4))
            fl6.append(getFunctions.toTime(pix6)-getFunctions.toTime(pix5))
            fl7.append(getFunctions.toTime(pix7)-getFunctions.toTime(pix6))
            fl8.append(getFunctions.toTime(pix8)-getFunctions.toTime(pix7))
            fl9.append(getFunctions.toTime(pix9)-getFunctions.toTime(pix8))
            fl10.append(getFunctions.toTime(pix10)-getFunctions.toTime(pix9))

            target.append(BatID[i])
        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path

    def convertIDSingleTSC(self, ID):
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


    def runThirdStageClassifier(self):
        out = []
        true = []
        #SingleBatIDToAdd = [1, 2, 3, 5, 6] # for single
        Correct = 0
        print "Loading Network.."
        net = NetworkReader.readFrom("C:\Users\Anoch\PycharmProjects\BatClassification\ThirdStageClassifier.xml")
        print "Loading feature data with SSC = 1 (Single call type)"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, pixelAverage, target, path = self.getDistrubedTestDataRUNVERSIONTSC()
        SAMPLE_SIZE = len(minFreq)
        for i in range(0, SAMPLE_SIZE):
            ClassifierOutput= net.activate([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], pixelAverage[i]])

            ClassifierOutputID = np.argmax(ClassifierOutput)
            currentTarget = self.convertIDSingleTSC(target[i])
            out.append(ClassifierOutputID)
            true.append(currentTarget)

            #MAPPING FROM BATID TO TSC value:
            TSC_value = ClassifierOutputID
            # Metadata Setup, get path and write: TSC = value
            ds = self.HDFFile[path[i]]
            ds.attrs["TSC"] = TSC_value
        self.HDFFile.flush()
        self.ConfusionMatrix =  self.CorrectRatio(out, true)
        return self.ConfusionMatrix

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


################################################### CONNECT ############################################################

    def runClassifiers(self):
        print "FIRST STAGE CLASSIFIER"
        self.runFirstStageClassifier()
        print "SECOND STAGE CLASSIFIER"
        self.runSecondStageClassifier()
        print "THRID STAGE CLASSIFIER"
        return self.runThirdStageClassifier()