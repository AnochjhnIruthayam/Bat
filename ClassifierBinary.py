__author__ = 'Anochjhn Iruthayam'

## Used to classify whetever the a event is a bat call or not. This process will ease the process of
## classifing species of the bats, since lot of the noise will be removed

import h5py, pybrain, re
from pybrain.datasets import ClassificationDataSet
import BatSpecies as BS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import random

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
        self.HDFFile = h5py
        self.Bat = BS.BatSpecies()

    def saveEventPath(self, name):
        self.pathEventList.append(name)

    def initClasissifer(self):
        self.HDFFile = h5py.File("/home/anoch/Documents/BatOutput/BatData.hdf5")
        self.HDFFile.visit(self.saveEventPath)



    ## Scans for all available events
    ## Returns: path where the event data is, target result, path where the event image data is
    def getBatEventHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
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
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
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
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
                # get data from path
                data = self.HDFFile[path]
                # as long as it is not other spice and something else;then add. Include all events and noise
                if data.attrs["BatID"] != 7 and data.attrs["BatID"] != 9:
                    BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)

        return pathcorr, BatID, pathcorrImg

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
        fl10 = []
        fl11 = []
        fl12 = []
        fl13 = []
        fl14 = []
        fl15 = []
        fl16 = []
        fl17 = []
        fl18 = []
        fl19 = []
        fl20 = []
        target = []

        pathcorr, BatID, pathcorrImg = self.getAllEventHDFInformation(self.pathEventList)
        EventSize = len(BatID)
        currentEvent = 0
        randomPathIterator = random.sample(xrange(0,EventSize), amount)
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

            fl11.append(toTime(pix1)-toTime(pix0))
            fl12.append(toTime(pix2)-toTime(pix0))
            fl13.append(toTime(pix3)-toTime(pix0))
            fl14.append(toTime(pix4)-toTime(pix0))
            fl15.append(toTime(pix5)-toTime(pix0))
            fl16.append(toTime(pix6)-toTime(pix0))
            fl17.append(toTime(pix7)-toTime(pix0))
            fl18.append(toTime(pix8)-toTime(pix0))
            fl19.append(toTime(pix9)-toTime(pix0))
            fl20.append(toTime(pix10)-toTime(pix0))

            target.append(BatID[i])


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, fl11, fl12, fl13, fl14, fl15, fl16, fl17, fl18, fl19, fl20, target


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
        fl11 = []
        fl12 = []
        fl13 = []
        fl14 = []
        fl15 = []
        fl16 = []
        fl17 = []
        fl18 = []
        fl19 = []
        fl20 = []
        randomPathIterator = []
        if dataID == 0:
            pathcorr, pathcorrImg = self.getNoiseEventHDFInformation(self.pathEventList)
        if dataID == 1:
            pathcorr, pathcorrImg = self.getBatEventHDFInformation(self.pathEventList)
        EventSize = len(pathcorr)
        currentEvent = 0

        randomPathIterator = random.sample(xrange(0,EventSize), amount)
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

            fl11.append(toTime(pix1)-toTime(pix0))
            fl12.append(toTime(pix2)-toTime(pix0))
            fl13.append(toTime(pix3)-toTime(pix0))
            fl14.append(toTime(pix4)-toTime(pix0))
            fl15.append(toTime(pix5)-toTime(pix0))
            fl16.append(toTime(pix6)-toTime(pix0))
            fl17.append(toTime(pix7)-toTime(pix0))
            fl18.append(toTime(pix8)-toTime(pix0))
            fl19.append(toTime(pix9)-toTime(pix0))
            fl20.append(toTime(pix10)-toTime(pix0))


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, fl11, fl12, fl13, fl14, fl15, fl16, fl17, fl18, fl19, fl20

    ## Input: Target ID
    ## Converts the Target ID to a number which can be interpreted by the ANN
    ## Returns: returns a modifed target ID
    def convertID(self, ID):
        if ID == 8:
            newID = 0
        else:
            newID = 1
        return  newID

    def goClassifer(self):
        print "Initilazing classifier"
        self.initClasissifer()
        #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
        trndata = ClassificationDataSet(23, target=1, nb_classes=2)
        tstdata = ClassificationDataSet(23, target=1, nb_classes=2)
        SAMPLE_SIZE = 160


        print "Adding Bat Events"
        myBat = 1

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, fl11, fl12, fl13, fl14, fl15, fl16, fl17, fl18, fl19, fl20 = self.getTrainingData(SAMPLE_SIZE, myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], fl11[i], fl12[i], fl13[i], fl14[i], fl15[i], fl16[i], fl17[i], fl18[i], fl19[i], fl20[i] ], [myBat])

        print "Adding Noise Events"
        myBat = 0

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, fl11, fl12, fl13, fl14, fl15, fl16, fl17, fl18, fl19, fl20 = self.getTrainingData(SAMPLE_SIZE, myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], fl11[i], fl12[i], fl13[i], fl14[i], fl15[i], fl16[i], fl17[i], fl18[i], fl19[i], fl20[i] ], [myBat])

        print "Adding test data"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, fl11, fl12, fl13, fl14, fl15, fl16, fl17, fl18, fl19, fl20, target = self.getTestData(2000)
        maxSize = len(minFreq)
        for i in range (0, maxSize):
            tempSave = i % 1000
            if tempSave == 0:
                print i
            myBat = self.convertID(target[i])
            tstdata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i], fl11[i], fl12[i], fl13[i], fl14[i], fl15[i], fl16[i], fl17[i], fl18[i], fl19[i], fl20[i]], [ myBat ])

        trndata._convertToOneOfMany( )
        tstdata._convertToOneOfMany( )
        print "Number of training patterns: ", len(trndata)
        print "Input and output dimensions: ", trndata.indim, trndata.outdim
        print "First sample (input, target, class):"
        print trndata['input'][0], trndata['target'][0], trndata['class'][0]
        print "200th sample (input, target, class):"
        print trndata['input'][100], trndata['target'][100], trndata['class'][100]


        #set up the Feed Forward Network
        net = buildNetwork(trndata.indim,10,trndata.outdim, bias=True, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, learningrate=0.001, verbose=True, weightdecay=0)
        print "Training data"

        for i in range(0,1000):
            # Train one epoch
            trainer.trainEpochs(1)


            #"""procentError(out, true) return percentage of mismatch between out and target values (lists and arrays accepted) error= ((out - true)/true)*100"""
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
            print("epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,"  test error: %5.2f%%" % tstresult)
