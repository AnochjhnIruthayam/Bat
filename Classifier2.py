__author__ = 'Anochjhn Iruthayam'
import h5py, pybrain, re
from pybrain.datasets import ClassificationDataSet
import BatSpecies as BS
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
import random, os

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
        print "Initilazing classifier"
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
                #Exclude certain classes/groups 4 because not enough data,
                if data.attrs["BatID"] != 0 and data.attrs["BatID"] != 4 and data.attrs["BatID"] != 7 and data.attrs["BatID"] != 8 and data.attrs["BatID"] != 9 and data.attrs["BatID"] != 10 and data.attrs["BatID"] != 11 and data.attrs["BatID"] != 12 and data.attrs["BatID"] != 13 and data.attrs["BatID"] != 14 and data.attrs["BatID"] != 15:
                    BatID.append(data.attrs["BatID"])
                    pathcorr.append(path)
                    imgPath = temp[0] + "/" + temp[1] + "/" + temp[2] + "/" + temp[3] + "/" + temp[4] + "/" + "ArrayImgEvent"
                    pathcorrImg.append(imgPath)


        return pathcorr, BatID, pathcorrImg

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
                if data.attrs["BatID"] != 7 and data.attrs["BatID"] != 4 and data.attrs["BatID"] != 13 and data.attrs["BatID"] != 15:
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

    def getDistrubedTestData(self, amount):
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
        target = []

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
                target.append(BatSpecies)

        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target

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

        pathcorr, pathcorrImg = self.getSpecificHDFInformation(self.pathEventList, ID)

        EventSize = len(pathcorr)
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


        return minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10


    def convertID(self, ID):

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

    def goClassifer(self, iteration, learningrate, momentum):
        #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
        trndata = ClassificationDataSet(13, nb_classes=11)
        tstdata = ClassificationDataSet(13, nb_classes=11)
        SAMPLE_SIZE = 100
        AmountPerSpecies = 100
        BatIDToAdd = [1, 2, 3, 5, 6, 10, 11, 12, 14]
        TraningDataAmount = 5000
        toFile = True

        print "Adding Bat Species Events"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target = self.getTrainingSpeciesDistributedData(BatIDToAdd, AmountPerSpecies)

        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [self.convertID(target[i])])

        print "Adding noise events"
        NoiseID = 8
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getDistributedData(AmountPerSpecies, NoiseID)
        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [self.convertID(NoiseID)])

        print "Adding something else events"
        SomethingElseID = 9
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10 = self.getDistributedData(AmountPerSpecies, SomethingElseID)
        SAMPLE_SIZE = len(minFreq)
        for i in range (0, SAMPLE_SIZE):
            trndata.addSample([ minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i] ], [self.convertID(SomethingElseID)])


        print "Adding test data"
        minFreq, maxFreq, Durantion, fl1, fl2, fl3, fl4, fl5, fl6, fl7, fl8, fl9, fl10, target = self.getDistrubedTestData(TraningDataAmount)
        maxSize = len(minFreq)
        for i in range (0, maxSize):
            tstdata.addSample([minFreq[i], maxFreq[i], Durantion[i], fl1[i], fl2[i], fl3[i], fl4[i], fl5[i], fl6[i], fl7[i], fl8[i], fl9[i], fl10[i]], [ self.convertID(target[i]) ])

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
        net = buildNetwork(trndata.indim, HiddenNeurons, trndata.outdim, bias=True, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(net, dataset=trndata, momentum=momentum, learningrate=learningrate, verbose=True, weightdecay=weightdecay)
        print "Training data"
        if toFile:
            #filename = "InputN" + str(trndata.indim) + "HiddenN" + str(HiddenNeurons) + "OutputN" + str(trndata.outdim) + "Momentum"+ str(momentum) + "LearningRate" + str(learningrate) + "Weightdecay" + str(weightdecay)
            filename = "ClassifierSpeciesTest_" + str(iteration) +"_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            folderName = "ClassifierSpeciesTest_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
            if not os.path.exists(folderName):
                os.makedirs(folderName)
            f = open(folderName + "/"+ filename + ".txt", 'w')

            value = "Added Bat Species: " + str(BatIDToAdd) + "\n"
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

            f.write("Input Activation function: Sigmoid function\n")
            f.write("Hidden Activation function: Sigmoid function\n")
            f.write("Output Activation function: Softmax function\n")

        maxEpoch = 2
        for i in range(0,maxEpoch):
            # Train one epoch
            trainer.trainEpochs(1)
            if toFile:
                averageError = trainer.testOnData(dataset=tstdata, verbose=False)


            #"""procentError(out, true) return percentage of mismatch between out and target values (lists and arrays accepted) error= ((out - true)/true)*100"""
            trnresult = percentError(trainer.testOnClassData(), trndata['class'])
            tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])

            if maxEpoch-1 == trainer.totalepochs:
                results, BatCount = self.CorrectRatio(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
                filename = "ClassifierSpeciesTest_" + str(iteration) +"_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)+ "_CR"
                folderName = "ClassifierSpeciesTest_MSE_LR_"+str(learningrate) + "_M_"+str(momentum)
                result_file = open(folderName + "/"+ filename + ".txt", 'w')
                result_file.write("[TruePositive, FalsePositive, CorrectRatio, BatCount]\n")
                result_file.write("[Eptesicus sertinus (single call), pipstrellus pygmaeus (single call), myotis daubeutonii (single call), pipistrellus nathusii (single call), nycalus noctula (single call), Eptesicus sertinus (Multi Call), pipstrellus pygmaeus (Multi Call), myotis daubeutonii (Multi Call), pipistrellus nathusii (Multi Call), nycalus noctula (Multi Call)]\n")
                result_file.write(str(results)+"\n")
                result_file.write(str(BatCount))
                result_file.close()

            print("epoch: %4d" % trainer.totalepochs,"  train error: %5.2f%%" % trnresult,"  test error: %5.2f%%" % tstresult)
            if toFile:
                dataString = str(trainer.totalepochs) + ", " + str(averageError) + ", " + str(trnresult) + ", " + str(tstresult) + "\n"
                f.write(dataString)
        if toFile:
            f.close()
        print "Done training"

    #Input: A list of the classifier output and the true target
    #Method: calculates the correct ratio based on true and false negative and positive
    #Output: A list of result: [TruePostive, FalsePostive, CorrectRatio], [BatCount]
    def CorrectRatio(self, out, true):
        TotalTest = len(out)
        TruePostive = 0
        FalsePostive = 0
        TrueBats = 0
        TrueNonBats = 0

        BatCount = [0,0,0,0,0,0,0,0,0,0,0]

        for i in range(0,TotalTest):
            for CBatID in range(0,11):
                if out[i] == CBatID:
                    BatCount[CBatID] += 1
                    if true[i] == CBatID:
                        TruePostive += 1
                        break
                    else:
                        FalsePostive += 1
                        break


        print "True Positive: " + str(TruePostive)
        print "False Positive: " + str(FalsePostive)
        print "True Bats: " + str(TrueBats)
        print "True Non Bats: " + str(TrueNonBats)
        CorrectRatio = float(TruePostive) / float(TotalTest) * 100
        print "Correct Ratio: " + str(CorrectRatio)
        print BatCount
        results = [TruePostive, FalsePostive, CorrectRatio]
        return results, BatCount