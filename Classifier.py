__author__ = 'Anochjhn Iruthayam'
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer
from pybrain.structure import SoftmaxLayer
from pybrain.utilities import percentError
import BatMain

def toTime(timePixel):
    imageLength = 5000.0
    return (1000.0/imageLength)*timePixel

def tokFreq(freqPixel):
    imageWidth = 1025.0
    return (250.0/imageWidth)*(imageWidth-freqPixel)

def ANN_input(event_dir,eventNo):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"
    lookup = "Event"+str(eventNo)

    tree = ET.parse(labelPath)
    root = tree.getroot()

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "minFreq_pixel":
            minFreq_pixel = elem.text
        if elem.tag == "maxFreq_pixel":
            maxFreq_pixel = elem.text
        if elem.tag == "minMiliSec_pixel":
            minMiliSec_pixel = elem.text
        if elem.tag == "maxMiliSec_pixel":
            maxMiliSec_pixel = elem.text

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big0_pixel":
            big0_pixel = elem.text
            break
        else:
            big0_pixel = 0
    for elem in tree.find("Event/"+lookup):

        if elem.tag == "big1_pixel":
            big1_pixel = elem.text
            break
        else:
            big1_pixel = 0
    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big2_pixel":
            big2_pixel = elem.text
            break
        else:
            big2_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big3_pixel":
            big3_pixel = elem.text
            break
        else:
            big3_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big4_pixel":
            big4_pixel = elem.text
            break
        else:
            big4_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big5_pixel":
            big5_pixel = elem.text
            break
        else:
            big5_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big6_pixel":
            big6_pixel = elem.text
            break
        else:
            big6_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big7_pixel":
            big7_pixel = elem.text
            break
        else:
            big7_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big8_pixel":
            big8_pixel = elem.text
            break
        else:
            big8_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big9_pixel":
            big9_pixel = elem.text
            break
        else:
            big9_pixel = 0

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "big10_pixel":
            big10_pixel = elem.text
            break
        else:
            big10_pixel = 0

    minMiliSec = toTime(float(minMiliSec_pixel))
    maxMiliSec = toTime(float(maxMiliSec_pixel))

    minFreq = tokFreq(float(minFreq_pixel))
    maxFreq = tokFreq(float(maxFreq_pixel))
    MiliSec = float(maxMiliSec)-float(minMiliSec)

    T_1 = toTime(float(big1_pixel)-float(big0_pixel))
    T_2 = toTime(float(big2_pixel)-float(big0_pixel))
    T_3 = toTime(float(big3_pixel)-float(big0_pixel))
    T_4 = toTime(float(big4_pixel)-float(big0_pixel))
    T_5 = toTime(float(big5_pixel)-float(big0_pixel))
    T_6 = toTime(float(big6_pixel)-float(big0_pixel))
    T_7 = toTime(float(big7_pixel)-float(big0_pixel))
    T_8 = toTime(float(big8_pixel)-float(big0_pixel))
    T_9 = toTime(float(big9_pixel)-float(big0_pixel))
    T_10 = toTime(float(big10_pixel)-float(big0_pixel))

    t_1 = toTime(float(big1_pixel)-float(big0_pixel))
    t_2 = toTime(float(big2_pixel)-float(big1_pixel))
    t_3 = toTime(float(big3_pixel)-float(big2_pixel))
    t_4 = toTime(float(big4_pixel)-float(big3_pixel))
    t_5 = toTime(float(big5_pixel)-float(big4_pixel))
    t_6 = toTime(float(big6_pixel)-float(big5_pixel))
    t_7 = toTime(float(big7_pixel)-float(big6_pixel))
    t_8 = toTime(float(big8_pixel)-float(big7_pixel))
    t_9 = toTime(float(big9_pixel)-float(big8_pixel))
    t_10 = toTime(float(big10_pixel)-float(big9_pixel))

    return minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10

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

def ANN_Classifier():
    realnonbat = 0
    realthisisBat = 0


    #Set up Classicication Data, 4 input, output is a one dim. and 2 possible outcome or two possible classes
    trndata = ClassificationDataSet(23,target=1, nb_classes=2)
    tstdata = ClassificationDataSet(23,target=1, nb_classes=2)



    rootpath = "/home/anoch/Documents/BatSamples/"
    #get all events
    #event, list_event_dir = get_all_bat_event(rootpath)
    print "Add true event"
    list_event_dir,eventNo = BatMain.getSample(160,1)
    for i in range(0, len(list_event_dir)):
        minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10 = ANN_input(list_event_dir[i],eventNo[i])
        print minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10
        target = c.ANN_outout(list_event_dir[i],eventNo[i])
        print target


        trndata.addSample([minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10],[target])

    print "Add nontrue event"

    list_event_dir, eventNo  = BatMain.getSample(160,0)
    #event, list_event_dir = get_all_bat_event(rootpath)
    for i in range(0, len(list_event_dir)):
        #eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10 = ANN_input(list_event_dir[i], eventNo[i])
        print "Input"
        print minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10
        target = c.ANN_outout(list_event_dir[i], eventNo[i])
        print "Output Target: " + str(target)


        trndata.addSample([minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10], [target])

    event, list_event_dir = BatMain.get_all_bat_event(rootpath)
    for i in range(1000, len(list_event_dir)):
        hej = len(list_event_dir)
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        #Get input data
        minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10 = ANN_input(list_event_dir[i],eventNo)
        print "Input"
        print minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10
        #get output data
        target = c.ANN_outout(list_event_dir[i],eventNo)
        print "Output Target: " + str(target)
        #Add the samples to dataset


        tstdata.addSample([minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10], [target])
    #print "Add training set"

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
    net = buildNetwork(trndata.indim,10,trndata.outdim, bias=True, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, dataset=trndata, momentum=0.1, learningrate=0.001, verbose=True, weightdecay=0)
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
    for i in range(0,2000):
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
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    event, list_event_dir = BatMain.get_all_bat_event(rootpath)
    text_file = open("/home/anoch/Documents/Output23Input.txt", "w")
    for i in range(600, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10 = ANN_input(list_event_dir[i],eventNo)
        real_target = ANN_outout(list_event_dir[i],eventNo)
        print minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10
        print "Sample: " + list_event_dir[i]
        print "EventNo: " + eventNo
        print "Real Target: " + str(real_target)
        pred = net.activate([minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10])
        print pred
        #if bat
        if pred[0] < 0.5 and pred[1] > 0.5:
            thisisBat = thisisBat + 1
            if real_target == 1:
                truePositive = truePositive + 1
                tp = 1
            elif real_target == 0:
                falsePositive = falsePositive + 1
                fp = 1
            print 1
        #if non-bat
        elif pred[0] > 0.5 and pred[1] < 0.5:
            if real_target == 0:
                trueNegative = trueNegative + 1
                tn = 1
            elif real_target == 1:
                falseNegative = falseNegative + 1
                fn = 1
            nonbat = nonbat + 1
            print 0


        if real_target == 1:
            realthisisBat = realthisisBat + 1
        else:
            realnonbat = realnonbat + 1
        print "\n\n"
        data = str(minFreq) + "," + str(maxFreq) + "," + str(MiliSec) + "," + str(T_1) + "," + str(T_2) + "," + str(T_3) + "," + str(T_4) + "," + str(T_5) + "," + str(T_6) + "," + str(T_7) + "," + str(T_8) + "," + str(T_9) + "," + str(T_10) + "," + str(t_1) + "," + str(t_2) + "," + str(t_3) + "," + str(t_4) + "," + str(t_5) + "," + str(t_6) + "," + str(t_7) + "," + str(t_8) + "," + str(t_9) + "," + str(t_10) + "," + str(real_target) + "," + str(tp) + "," + str(tn) + "," + str(fp) + "," + str(fn) + "\n"
        text_file.write(data)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
    text_file.close()
    print "Classifier Result"
    print "non bat: " + str(nonbat)
    print "bat: " + str(thisisBat)
    print "Real result"
    print "non bat: " + str(realnonbat)
    print "bat: " + str(realthisisBat)
    print "True Positive: " + str(truePositive)
    print "True Negative: " + str(trueNegative)
    print "False Positive: " + str(falsePositive)
    print "False Negative: " + str(falseNegative)