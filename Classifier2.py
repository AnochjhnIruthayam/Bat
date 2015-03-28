__author__ = 'Anochjhn Iruthayam'
import h5py, pybrain, re
from pybrain.datasets import ClassificationDataSet
import BatSpecies as BS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
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
        self.HDFFile = h5py
        self.Bat = BS.BatSpecies()

    def saveEventPath(self, name):
        self.pathEventList.append(name)

    def initClasissifer(self):
        self.HDFFile = h5py.File("/home/anoch/Documents/BatOutput/BatData.hdf5")
        self.HDFFile.visit(self.saveEventPath)



    def getHDFInformation(self, paths):
        pathcorr = []
        pathcorrImg = []
        BatID = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
                data = self.HDFFile[path]
                #Exclude certain classes/groups
                if data.attrs["BatID"] != 0 and data.attrs["BatID"] != 4 and data.attrs["BatID"] != 7 and data.attrs["BatID"] != 8 and data.attrs["BatID"] != 9:
                    BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)


        return pathcorr, BatID, pathcorrImg

    def getSpecificHDFInformation(self, paths, BatID):
        pathcorr = []
        pathcorrImg = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
                #get data from path
                data = self.HDFFile[path]
                if data.attrs["BatID"] == BatID:
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)

        return pathcorr, pathcorrImg

    def getTestData(self):
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
        for i in range (0, EventSize):
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

    def EnergyPlacementFindMin(self, dataImg):
        height, length = dataImg.shape
        for x in range (0, length):
            for y in range(0,height):
                if dataImg[y, x] > 1:
                    return x

    def EnergyPlacementFindMax(self, dataImg):
        height, length = dataImg.shape
        for x in xrange (length-1, 0, -1):
            for y in xrange(height-1, 0, -1):
                if dataImg[y, x] > 1:
                    return x

    def EnergyPlacementCounter(self, min, max, dataImg):
        height, length = dataImg.shape
        TotalEnergy = 0
        for x in range(min, max):
            for y in range(0, height):
                TotalEnergy += dataImg[y,x]
        return TotalEnergy

    def EnergyPlacement(self, dataImg):
        EP1 = 0
        EP2 = 0
        EP3 = 0
        EP4 = 0
        EP5 = 0
        EP6 = 0
        EP7 = 0
        EP8 = 0
        EP9 = 0
        EP10 = 0
        minX = self.EnergyPlacementFindMin(dataImg)
        maxX = self.EnergyPlacementFindMax(dataImg)
        length = abs(maxX-minX)
        step = length/10
        temp = 0

        EP1 = self.EnergyPlacementCounter(minX + (step*0), minX+(step*1), dataImg)
        EP2 = self.EnergyPlacementCounter(minX+(step*1), minX+(step*2), dataImg)
        EP3 = self.EnergyPlacementCounter(minX+(step*2), minX+(step*3), dataImg)
        EP4 = self.EnergyPlacementCounter(minX+(step*3), minX+(step*4), dataImg)
        EP5 = self.EnergyPlacementCounter(minX+(step*4), minX+(step*5), dataImg)
        EP6 = self.EnergyPlacementCounter(minX+(step*5), minX+(step*6), dataImg)
        EP7 = self.EnergyPlacementCounter(minX+(step*6), minX+(step*7), dataImg)
        EP8 = self.EnergyPlacementCounter(minX+(step*7), minX+(step*8), dataImg)
        EP9 = self.EnergyPlacementCounter(minX+(step*8), minX+(step*9), dataImg)
        EP10 = self.EnergyPlacementCounter(minX+(step*9), minX+(step*10), dataImg)

        return EP1, EP2, EP3, EP4, EP5, EP6, EP7, EP8, EP9, EP10


    def getTrainingData(self, BatID, amount):
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
        currentEvent = 0
        for i in range (0, amount):
            data = self.HDFFile[pathcorr[i]]
            dataImg = self.HDFFile[pathcorrImg[i]]
            self.EnergyPlacement(dataImg)
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


    def convertID(self, ID):
        if ID == 1:
            newID = 0
        if ID == 2:
            newID = 1
        if ID == 3:
            newID = 2
        if ID == 5:
            newID = 3
        if ID == 6:
            newID = 4
        return  newID

    def goClassifer(self):
        self.initClasissifer()
        #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
        trndata = ClassificationDataSet(13, nb_classes=5)
        tstdata = ClassificationDataSet(13, nb_classes=5)
        SAMPLE_SIZE = 100


        print "Adding Eptesicus serotinus"
        myBat = self.Bat.Eptesicus_serotinus

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(myBat, SAMPLE_SIZE)
        myBat = self.convertID(myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [myBat])

        print "Adding Pipistrellus pygmaeus"
        myBat= self.Bat.Pipistrellus_pygmaeus

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(myBat, SAMPLE_SIZE)
        myBat = self.convertID(myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]],[myBat])

        print "Adding Myotis daubeutonii"
        myBat= self.Bat.Myotis_daubeutonii

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(myBat, SAMPLE_SIZE)
        myBat = self.convertID(myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]],[myBat])

        """
        print "Adding Myotis dasycneme"
        myBat= self.Bat.Myotis_dasycneme
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(self, myBat, SAMPLE_SIZE)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]],[myBat])
        """

        print "Adding Pipstrllus nathusii"
        myBat= self.Bat.Pipistrellus_nathusii

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(myBat, SAMPLE_SIZE)
        myBat = self.convertID(myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]],[myBat])


        print "Adding Nyctalus noctula"
        myBat= self.Bat.Nyctalus_noctula

        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getTrainingData(myBat, SAMPLE_SIZE)
        myBat = self.convertID(myBat)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]],[myBat])

        print "Adding test data"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target = self.getTestData()
        maxSize = len(minFreq)
        for i in range (0, maxSize):
            tempSave = i % 1000
            if tempSave == 0:
                print i
            myBat = self.convertID(target[i])
            tstdata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]], [ myBat ])

        trndata._convertToOneOfMany( )
        tstdata._convertToOneOfMany( )
        print "Number of training patterns: ", len(trndata)
        print "Input and output dimensions: ", trndata.indim, trndata.outdim
        print "First sample (input, target, class):"
        print trndata['input'][0], trndata['target'][0], trndata['class'][0]
        print "200th sample (input, target, class):"
        print trndata['input'][200], trndata['target'][200], trndata['class'][200]


        #set up the Feed Forward Network
        net = buildNetwork(trndata.indim,12,trndata.outdim, bias=True, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, learningrate=0.001, verbose=True, weightdecay=0)
        print "Training data"

        for i in range(0,1000):
            trainer.trainEpochs(1)
            trnresult = percentError(trainer.testOnClassData(),
                                     trndata['class'])
            tstresult = percentError(trainer.testOnClassData(
                                     dataset=tstdata), tstdata['class'])
            print("epoch: %4d" % trainer.totalepochs,
                  "  train error: %5.2f%%" % trnresult,
                  "  test error: %5.2f%%" % tstresult)