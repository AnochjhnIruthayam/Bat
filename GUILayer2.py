__author__ = 'Anochjhn Iruthayam'


import sys
from PyQt4 import QtCore, QtGui
from BatWindow import Ui_BatWindow
import EventExtraction, os, getFunctions, time
import h5py, re
import Classifier2, ClassifierBinary

def toTime(timePixel):
    imageLength = 5000.0
    return (1000.0/imageLength)*timePixel

def tokFreq(freqPixel):
    imageWidth = 1025.0
    return (250.0/imageWidth)*(imageWidth-freqPixel)


class StartQT4(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_BatWindow()
        self.ui.setupUi(self)
        self.ui.frame_BatButtons.hide()
        self.pathEventList = []
        QtCore.QObject.connect(self.ui.pushButton_SetInputDirectory, QtCore.SIGNAL("clicked()"), self.setInputDirectory)
        QtCore.QObject.connect(self.ui.pushButton_SetOutputDirectory, QtCore.SIGNAL("clicked()"), self.setOutputDirectory)
        QtCore.QObject.connect(self.ui.button_start, QtCore.SIGNAL("clicked()"), self.run_analyser)
        QtCore.QObject.connect(self.ui.pushButton_createSpectrogram, QtCore.SIGNAL("clicked()"), self.create_spectrogram)
        QtCore.QObject.connect(self.ui.button_loaddatabase, QtCore.SIGNAL("clicked()"), self.file_dialog)
        QtCore.QObject.connect(self.ui.button_EptesicusSerotinus, QtCore.SIGNAL("clicked()"), self.getValueEptesicusSerotinus)
        QtCore.QObject.connect(self.ui.button_PipistrellusPygmaeus, QtCore.SIGNAL("clicked()"), self.getValuePipistrellusPygmaeus)
        QtCore.QObject.connect(self.ui.button_MyotisDaubentonii, QtCore.SIGNAL("clicked()"), self.getValueMyotisDaubentonii)
        QtCore.QObject.connect(self.ui.button_MyotisDasycneme, QtCore.SIGNAL("clicked()"), self.getValueMyotisDasycneme)
        QtCore.QObject.connect(self.ui.button_pipistrellusNathusii, QtCore.SIGNAL("clicked()"), self.getValuePipistrellusNathusii)
        QtCore.QObject.connect(self.ui.button_NyctalusNoctula, QtCore.SIGNAL("clicked()"), self.getValueNyctalusNoctula)
        QtCore.QObject.connect(self.ui.button_noise, QtCore.SIGNAL("clicked()"), self.getValueNoise)
        QtCore.QObject.connect(self.ui.button_OtherSpecies, QtCore.SIGNAL("clicked()"), self.getValueOtherSpecies)
        QtCore.QObject.connect(self.ui.button_SomethingElse, QtCore.SIGNAL("clicked()"), self.getValueSomethingElse)
        QtCore.QObject.connect(self.ui.checkBox_scaledZoom, QtCore.SIGNAL("clicked()"), self.ScaledZoom)
        QtCore.QObject.connect(self.ui.button_ShowFullSpectrogram, QtCore.SIGNAL("pressed()"), self.ShowFullSpectrogramPressed)
        QtCore.QObject.connect(self.ui.button_ShowFullSpectrogram, QtCore.SIGNAL("released()"), self.resetRelease)
        QtCore.QObject.connect(self.ui.button_ShowMarkedSpectrogram, QtCore.SIGNAL("pressed()"), self.ShowMarkedSpectrogramPressed)
        QtCore.QObject.connect(self.ui.button_ShowMarkedSpectrogram, QtCore.SIGNAL("released()"), self.resetRelease)
        QtCore.QObject.connect(self.ui.button_undo, QtCore.SIGNAL("clicked()"), self.undoLastEvent)
        QtCore.QObject.connect(self.ui.button_save, QtCore.SIGNAL("clicked()"), self.saveCurrentProgress)
        QtCore.QObject.connect(self.ui.button_loaddatabase_browser, QtCore.SIGNAL("clicked()"), self.countSpecies)
        QtCore.QObject.connect(self.ui.button_browser_Next, QtCore.SIGNAL("clicked()"), self.buttonNextHandler)
        QtCore.QObject.connect(self.ui.button_browser_previous, QtCore.SIGNAL("clicked()"), self.buttonPreviousHandler)
        QtCore.QObject.connect(self.ui.comboBox_SelectSpecies, QtCore.SIGNAL("activated(QString)"), self.browserComboxSelectSpeciesHandler)
        QtCore.QObject.connect(self.ui.button_classifier_run, QtCore.SIGNAL("clicked()"), self.runClassifier)
        QtCore.QObject.connect(self.ui.button_classifier_run_binary, QtCore.SIGNAL("clicked()"), self.runBinaryClassifier)

        #QtCore.QObject.connect(self.ui.progressBar)
        self.HDFFile = h5py
        self.EventSize = 0
        self.currentEvent = 0
        self.previousEvent = 0
        self.ProcessCount = 0
        self.MultiCount = 0
        self.day = []
        self.month = []
        self.year = []
        self.file = []
        self.pathcorr = []
        self.eventno = []
        self.SoundFileList = []
        self.OutputDirectory = "/home/anoch/Documents/BatSamplesOutput"
        self.InputDirectory = "/home/anoch/Documents/BatSamplesInput"
        self.ui.label_outputDirectory.setText(self.OutputDirectory)
        self.ui.label_inputDirectory.setText(self.InputDirectory)
        self.classifier = Classifier2.Classifier()
        self.classifierBinary = ClassifierBinary.BinaryClassifier()

        if self.ui.checkBox_scaledZoom.isChecked():
            self.ZoomInParameter = 1
        else:
            self.ZoomInParameter = 0

    def labelBat(self, ID):
        data = self.HDFFile[str(self.pathcorr[self.currentEvent])]
        data.attrs['BatID'] = ID
        if self.scanForNextEvent():
            self.updateEventInfomation()

    def labelCall(self, ID):
        data = self.HDFFile[str(self.pathcorr[self.currentEvent])]
        data.attrs['Call Type'] = ID
        if self.scanForNextEvent():
            self.updateEventInfomation()

    def getValueEptesicusSerotinus(self):
        self.labelBat(1)
    def getValuePipistrellusPygmaeus(self):
        self.labelBat(2)
    def getValueMyotisDaubentonii(self):
        self.labelBat(3)
    def getValueMyotisDasycneme(self):
        self.labelBat(4)
    def getValuePipistrellusNathusii(self):
        self.labelBat(5)
    def getValueNyctalusNoctula(self):
        self.labelBat(6)

    def getValueOtherSpecies(self):
        self.labelBat(7)
    def getValueNoise(self):
        self.labelBat(8)
    def getValueSomethingElse(self):
        self.labelBat(9)

    def getValueEptesicusSerotinusMulti(self):
        self.labelBat(10)
    def getValuePipistrellusPygmaeusMulti(self):
        self.labelBat(11)
    def getValueMyotisDaubentoniiMulti(self):
        self.labelBat(12)
    def getValueMyotisDasycnemeMulti(self):
        self.labelBat(13)
    def getValuePipistrellusNathusiiMulti(self):
        self.labelBat(14)
    def getValueNyctalusNoctulaMulti(self):
        self.labelBat(15)



    def getCallEcholocation(self):
        self.labelCall(1)
    def getCallSocialCall(self):
        self.labelCall(2)
    def getCallForaging(self):
        self.labelCall(3)
    def getCallEcholocationSocialCall(self):
        self.labelCall(4)
    def getCallSomethingElse(self):
        self.labelCall(5)
    def saveCurrentProgress(self):
        self.HDFFile.flush()

    def saveEventPath(self,name):
        self.pathEventList.append(name)

    def undoLastEvent(self):
        if not self.currentEvent == 0:
            self.currentEvent = self.previousEvent
            self.updateEventInfomation()

    def ShowFullSpectrogramPressed(self):
        FullSpecImg = self.OutputDirectory + "/Spectrogram/" + self.file[self.currentEvent] + ".png"
        eventImage = QtGui.QPixmap(FullSpecImg)
        scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.label_imageshow.setPixmap(scaledEventImage)
        #if self.ZoomInParameter == 1:
        #    scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio)
        #    self.ui.label_imageshow.setPixmap(scaledEventImage)
        #else:
        #    self.ui.label_imageshow.setPixmap(eventImage)

    def ShowMarkedSpectrogramPressed(self):
        MarkedSpecImg = self.OutputDirectory + "/SpectrogramMarked/" + self.file[self.currentEvent] + "/SpectrogramAllMarked.png"
        eventImage = QtGui.QPixmap(MarkedSpecImg)
        scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.label_imageshow.setPixmap(scaledEventImage)
        #if self.ZoomInParameter == 1:
        #    scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio)
        #    self.ui.label_imageshow.setPixmap(scaledEventImage)
        #else:
        #    self.ui.label_imageshow.setPixmap(eventImage)

    def resetRelease(self):
        self.updateEventInfomation()

    def ScaledZoom(self):
        if self.ui.checkBox_scaledZoom.isChecked():
            self.ZoomInParameter = 1
            self.updateEventInfomation()
        else:
            self.ZoomInParameter = 0
            self.updateEventInfomation()

    # Overload function
    def keyPressEvent(self, QKeyEvent):
        # if this batbuttons are visible, means we have loaded the data
        if type(QKeyEvent) == QtGui.QKeyEvent:
            if self.ui.tabWidget.currentIndex() == 1:
                print QKeyEvent.key()
                #Check if the label species tab is open
                if self.ui.frame_BatButtons.isVisible():
                    # following numbers are ASCII for 1, 2, 3, 4, 5, 6 and 7
                    ######SINGLE CALL KEY BINDINGS##########
                    if QKeyEvent.key() == 49: # 1
                        self.getValueEptesicusSerotinus()

                    if QKeyEvent.key() == 50: # 2
                        self.getValuePipistrellusPygmaeus()

                    if QKeyEvent.key() == 51: # 3
                        self.getValueMyotisDaubentonii()

                    if QKeyEvent.key() == 52: # 4
                        self.getValueMyotisDasycneme()

                    if QKeyEvent.key() == 53: # 5
                        self.getValuePipistrellusNathusii()

                    if QKeyEvent.key() == 54: # 6
                        self.getValueNyctalusNoctula()

                    if QKeyEvent.key() == 55: # 7
                        self.getValueOtherSpecies()

                    if QKeyEvent.key() == 56: # 8
                        self.getValueNoise()

                    if QKeyEvent.key() == 57: # 9
                        self.getValueSomethingElse()

                    ######MULTI CALLS KEY BINDINGS##########
                    if QKeyEvent.key() == 81: # q
                        self.getValueEptesicusSerotinusMulti()

                    if QKeyEvent.key() == 87: # w
                        self.getValuePipistrellusPygmaeusMulti()

                    if QKeyEvent.key() == 69: # e
                        self.getValueMyotisDaubentoniiMulti()

                    if QKeyEvent.key() == 82: # r
                        self.getValueMyotisDasycnemeMulti()

                    if QKeyEvent.key() == 84: # t
                        self.getValuePipistrellusNathusiiMulti()

                    if QKeyEvent.key() == 89: # y
                        self.getValueNyctalusNoctulaMulti()

                    ####OTHER OPTIONS KEY BINDINGS#############
                    if QKeyEvent.key() == 90: # z
                        if self.ui.checkBox_scaledZoom.isChecked():
                            self.ui.checkBox_scaledZoom.setChecked(False)
                            self.ScaledZoom()
                        else:
                            self.ui.checkBox_scaledZoom.setChecked(True)
                            self.ScaledZoom()
                    if QKeyEvent.key() == 83: # s
                        self.ShowFullSpectrogramPressed()
                    if QKeyEvent.key() == 77: # m
                        self.ShowMarkedSpectrogramPressed()
                    if QKeyEvent.key() == 85:
                        self.undoLastEvent()
                # Check if lavel action tab is open
            if self.ui.tabWidget.currentIndex() == 2:
                if QKeyEvent.key() == 49:
                    self.getCallEcholocation()
                if QKeyEvent.key() == 50:
                    self.getCallSocialCall()
                if QKeyEvent.key() == 51:
                    self.getCallForaging()
                if QKeyEvent.key() == 52:
                    self.getCallEcholocationSocialCall()
                if QKeyEvent.key() == 53:
                    self.getCallSomethingElse()



    # Overload function
    def keyReleaseEvent(self, QKeyEvent):
        if self.ui.tabWidget.currentIndex() == 1:
            if self.ui.frame_BatButtons.isVisible():
                if type(QKeyEvent) == QtGui.QKeyEvent:
                    if QKeyEvent.key() == 83:
                        self.resetRelease()
                    if QKeyEvent.key() == 77:
                        self.resetRelease()

    def runClassifier(self):
        self.classifier.goClassifer()

    def runBinaryClassifier(self):
        self.classifierBinary.initClasissifer()
        learningRate = [0.001, 0.001, 0.01, 0.01, 0.1, 0.1, 0.1]
        momentum = [0.01, 0.001, 0.01, 0.001, 0.1, 0.01, 0.001]
        for setting in range (0, len(learningRate)):
            for i in range(0, 5):
                self.classifierBinary.goClassifer(i, learningRate[setting], momentum[setting] )


    def getHDFInformation(self, paths):
        day = []
        month = []
        year = []
        file = []
        pathcorr = []
        eventno = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
                #get data from path
                year.append(temp[0])
                month.append(temp[1])
                day.append(temp[2])
                data = self.HDFFile[path]
                hour = str(data.attrs["Hour"])
                minute = str(data.attrs["Minute"])
                second = str(data.attrs["Second"])
                #offset = str(data.attrs["Offset"])
                channel = str(data.attrs["Recording Channel"])
                if len(hour) == 1:
                    hour = "0" + hour
                if len(minute) == 1:
                    minute = "0" + minute
                if len(second) == 1:
                    second = "0" + second
                tempOffset = re.split('_', temp[3])
                offset = tempOffset[1]
                #if len(offset) != 20:
                #    for i in range(0,20):
                #        offset = "0" + offset
                #        if len(offset) == 20:
                #            break
                filename = "date_" + temp[2] + "_" + temp[1] + "_" + temp[0] + "_time_" + hour + "_" + minute + "_" + second +"_ch_" + channel +  "_offset_" + offset
                file.append(filename)
                #file.append(temp[3])
                eventnoTemp = re.split('_',temp[4])
                eventno.append(eventnoTemp[1])
                pathcorr.append(path)

        return day, month, year, file, eventno, pathcorr

    def getSpecificHDFInformation(self, paths, BatID):
        day = []
        month = []
        year = []
        file = []
        pathcorr = []
        eventno = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            if len(temp) == 6 and temp[5] == "FeatureDataEvent":
                #get data from path
                data = self.HDFFile[path]
                if data.attrs["BatID"] == BatID:
                    year.append(temp[0])
                    month.append(temp[1])
                    day.append(temp[2])
                    hour = str(data.attrs["Hour"])
                    minute = str(data.attrs["Minute"])
                    second = str(data.attrs["Second"])
                    #offset = str(data.attrs["Offset"])
                    channel = str(data.attrs["Recording Channel"])
                    if len(hour) == 1:
                        hour = "0" + hour
                    if len(minute) == 1:
                        minute = "0" + minute
                    if len(second) == 1:
                        second = "0" + second
                    tempOffset = re.split('_', temp[3])
                    offset = tempOffset[1]
                    #if len(offset) != 20:
                    #    for i in range(0,20):
                    #        offset = "0" + offset
                    #        if len(offset) == 20:
                    #            break
                    filename = "date_" + temp[2] + "_" + temp[1] + "_" + temp[0] + "_time_" + hour + "_" + minute + "_" + second +"_ch_" + channel +  "_offset_" + offset
                    file.append(filename)
                    #file.append(temp[3])
                    eventnoTemp = re.split('_',temp[4])
                    eventno.append(eventnoTemp[1])
                    pathcorr.append(path)

        return day, month, year, file, eventno, pathcorr

    def setEventImage(self, event, eventno):
        root = self.OutputDirectory + "/SpectrogramMarked/"
        path  = root + event + "/Event" + eventno + ".png"
        eventImage = QtGui.QPixmap(path)
        if self.ZoomInParameter == 1:
            scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio)
            self.ui.label_imageshow.setPixmap(scaledEventImage)
        else:
            self.ui.label_imageshow.setPixmap(eventImage)

    def updateEventInfomation(self):
        data = self.HDFFile[str(self.pathcorr[self.currentEvent])]
        self.setEventImage(self.file[self.currentEvent], self.eventno[self.currentEvent])
        dateToShow = self.day[self.currentEvent] + "-" + self.month[self.currentEvent] + "-" + self.year[self.currentEvent]
        self.ui.label_Date.setText(dateToShow)

        # Recontruction of original filename
        originalFilename = "sr_500000_ch_4_offset_" + str(data.attrs["Offset"])

        self.ui.label_EventName.setText(originalFilename)
        self.ui.label_EventNo.setText(self.eventno[self.currentEvent])
        self.ui.progressBar_eventLabel.setValue(self.currentEvent)

        self.ui.label_time.setText(self.timeHandler(data))
        self.ui.label_currentStatus.setText("Current Event Status: " + str(self.currentEvent + 1) + " out of " + str(self.EventSize) + ". Process Count: " + str(self.ProcessCount))
        minFreq = tokFreq(data[0])
        maxFreq = tokFreq(data[1])
        bat_suggestion = ""
        if minFreq > 20 and minFreq <30 and maxFreq > 30 and maxFreq < 60:
            bat_suggestion += "(1) "
        if minFreq > 45 and minFreq <60 and maxFreq > 70 and maxFreq < 100:
            bat_suggestion += "(2) "
        if minFreq > 25 and minFreq <38 and maxFreq > 70 and maxFreq < 90:
            bat_suggestion += "(3) "
        if minFreq > 20 and minFreq <30 and maxFreq > 70 and maxFreq < 80:
            bat_suggestion += "(4) "
        if minFreq > 27 and minFreq <44 and maxFreq > 57 and maxFreq < 68:
            bat_suggestion += "(5) "
        if minFreq > 15 and minFreq <30 and maxFreq > 15 and maxFreq < 50:
            bat_suggestion += "(6) "
        self.ui.label_BatSuggestion.setText("Bat species suggestion: " + bat_suggestion)
        duration = str(toTime(abs(data[2]-data[3])))
        self.ui.label_MinFreq.setText(str(minFreq))
        self.ui.label_maxFreq.setText(str(maxFreq))

        self.ui.label_Duration.setText(duration)
        self.ui.label_FrontLine_1.setText(str(data[4]))
        self.ui.label_FrontLine_2.setText(str(data[5]))
        self.ui.label_FrontLine_3.setText(str(data[6]))
        self.ui.label_FrontLine_4.setText(str(data[7]))
        self.ui.label_FrontLine_5.setText(str(data[8]))
        self.ui.label_FrontLine_6.setText(str(data[9]))
        self.ui.label_FrontLine_7.setText(str(data[10]))
        self.ui.label_FrontLine_8.setText(str(data[11]))
        self.ui.label_FrontLine_9.setText(str(data[12]))
        self.ui.label_FrontLine_10.setText(str(data[13]))
        self.ui.label_FrontLine_11.setText(str(data[14]))

    def timeHandler(self, data):
        hour = str(data.attrs["Hour"])
        minute = str(data.attrs["Minute"])
        second = str(data.attrs["Second"])
        if len(hour) == 1:
            hour = "0" + hour
        if len(minute) == 1:
            minute = "0" + minute
        if len(second) == 1:
            second = "0" + second
        return hour + ":" + minute + ":" + second

    def lastEventHandler(self):
        #when we are finished with labelling, hide buttons for bats and show N/A on labels
        self.ui.progressBar_eventLabel.setValue(self.currentEvent)
        self.ui.frame_BatButtons.hide()
        self.ui.label_imageshow.setText("Labelling Done!")
        self.ui.label_Date.setText("N/A")
        self.ui.label_EventName.setText("N/A")
        self.ui.label_EventNo.setText("N/A")
        self.ui.label_time.setText("N/A")

        self.ui.label_MinFreq.setText("N/A")
        self.ui.label_maxFreq.setText("N/A")
        self.ui.label_Duration.setText("N/A")
        self.ui.label_FrontLine_1.setText("N/A")
        self.ui.label_FrontLine_2.setText("N/A")
        self.ui.label_FrontLine_3.setText("N/A")
        self.ui.label_FrontLine_4.setText("N/A")
        self.ui.label_FrontLine_5.setText("N/A")
        self.ui.label_FrontLine_6.setText("N/A")
        self.ui.label_FrontLine_7.setText("N/A")
        self.ui.label_FrontLine_8.setText("N/A")
        self.ui.label_FrontLine_9.setText("N/A")
        self.ui.label_FrontLine_10.setText("N/A")
        self.ui.label_FrontLine_11.setText("N/A")
        self.HDFFile.close()




    def scanForNextEvent2(self):
        self.previousEvent = self.currentEvent
        SAVE_ITERATION = 5
        self.ProcessCount += 1
        # save to disk for every 5th event
        tempSave = self.ProcessCount % SAVE_ITERATION
        if tempSave == 0:
            self.HDFFile.flush()
        self.currentEvent += 1
        return True
        #for i in range(self.currentEvent, self.EventSize):
            #data = self.HDFFile[str(self.pathcorr[i])]
            ###############################################
            ###############################################CHANGES!!
            ###############################################
            #if data.attrs['BatID'] == 0:

                #FrontLineValue = 0
                # We check if the full frontline has a value, if not its noise
                #for FrontLineIndex in range(4,14):
                #    FrontLineValue += data[FrontLineIndex]
                #if FrontLineValue > 0:
                #    self.currentEvent = i
                #    return True
                #else:
                #    data.attrs['BatID'] = 8 # 8 is the ID for noise
        #self.lastEventHandler()
        #return False
    def scanForNextEvent(self):
        self.previousEvent = self.currentEvent
        SAVE_ITERATION = 5
        self.ProcessCount += 1
        # save to disk for every 5th event
        tempSave = self.ProcessCount % SAVE_ITERATION
        if tempSave == 0:
            self.HDFFile.flush()
        for i in range(self.currentEvent, self.EventSize):
            data = self.HDFFile[str(self.pathcorr[i])]
            if data.attrs['BatID'] == 0:
                FrontLineValue = 0
                #We check if the full frontline has a value, if not its noise
                for FrontLineIndex in range(4,14):
                    FrontLineValue += data[FrontLineIndex]
                if FrontLineValue > 0:
                    self.currentEvent = i
                    return True
                else:
                    data.attrs['BatID'] = 8 # 8 is the ID for noise
        self.lastEventHandler()
        return False

    def setInputDirectory(self):
        self.InputDirectory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Input Directory"))
        if os.path.exists(self.InputDirectory):
            self.ui.label_inputDirectory.setText(self.InputDirectory)
        else:
            self.ui.label_inputDirectory.setText("None Selected!")

    def setOutputDirectory(self):
        self.OutputDirectory = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Output Directory"))
        if os.path.exists(self.OutputDirectory):
            self.ui.label_outputDirectory.setText(self.OutputDirectory)
        else:
            self.ui.label_outputDirectory.setText("None Selected!")

    def countSpecies(self):
        self.filepath = QtGui.QFileDialog.getOpenFileName(self, "Open HDF5 File",'', "HDF5 Files (*.hdf5 *.h5)")

        self.HDFFile = h5py.File(str(self.filepath))
        self.HDFFile.visit(self.saveEventPath)

        self.ui.comboBox_SelectSpecies.addItem("Eptesicus serotinus")
        self.ui.comboBox_SelectSpecies.addItem("Pipistrellus pygmaeus")
        self.ui.comboBox_SelectSpecies.addItem("Myotis daubeutonii")
        self.ui.comboBox_SelectSpecies.addItem("Myotis dasycneme")
        self.ui.comboBox_SelectSpecies.addItem("Pipistrellus nathusii")
        self.ui.comboBox_SelectSpecies.addItem("Nyctalus noctula")
        self.ui.comboBox_SelectSpecies.addItem("Eptesicus serotinus (Multi)")
        self.ui.comboBox_SelectSpecies.addItem("Pipistrellus pygmaeus (Multi)")
        self.ui.comboBox_SelectSpecies.addItem("Myotis daubeutonii (Multi)")
        self.ui.comboBox_SelectSpecies.addItem("Myotis dasycneme (Multi)")
        self.ui.comboBox_SelectSpecies.addItem("Pipistrellus nathusii (Multi)")
        self.ui.comboBox_SelectSpecies.addItem("Nyctalus noctula (Multi)")
        self.ui.comboBox_SelectSpecies.addItem("Other spicies")
        self.ui.comboBox_SelectSpecies.addItem("Noise")
        self.ui.comboBox_SelectSpecies.addItem("Something else")

    def browserComboxSelectSpeciesHandler(self, text):
        self.ui.label_imageshow_browser.setText("Loading...")
        SelectedSpecies = 0
        if text == "Eptesicus serotinus":
            SelectedSpecies = 1
        if text == "Pipistrellus pygmaeus":
            SelectedSpecies = 2
        if text == "Myotis daubeutonii":
            SelectedSpecies = 3
        if text == "Myotis dasycneme":
            SelectedSpecies = 4
        if text == "Pipistrellus nathusii":
            SelectedSpecies = 5
        if text == "Nyctalus noctula":
            SelectedSpecies = 6
        if text == "Eptesicus serotinus (Multi)":
            SelectedSpecies = 10
        if text == "Pipistrellus pygmaeus (Multi)":
            SelectedSpecies = 11
        if text == "Myotis daubeutonii (Multi)":
            SelectedSpecies = 12
        if text == "Myotis dasycneme (Multi)":
            SelectedSpecies = 13
        if text == "Pipistrellus nathusii (Multi)":
            SelectedSpecies = 14
        if text == "Nyctalus noctula (Multi)":
            SelectedSpecies = 15
        if text == "Other spicies":
            SelectedSpecies = 7
        if text == "Noise":
            SelectedSpecies = 8
        if text == "Something else":
            SelectedSpecies = 9
        #make sure if we have an empty list
        self.day[:] = []
        self.month[:] = []
        self.year[:] = []
        self.file[:] = []
        self.eventno[:] = []
        self.pathcorr[:] = []


        self.day, self.month, self.year, self.file, self.eventno, self.pathcorr = self.getSpecificHDFInformation(self.pathEventList, SelectedSpecies)
        self.EventSize = len(self.day)
        self.currentEvent = 0
        thisFilePath, thisFile = os.path.split(str(self.filepath))
        self.OutputDirectory = thisFilePath
        self.ui.label_outputDirectory.setText(self.OutputDirectory)
        self.updateBrowserEventInfomation()


    def buttonPreviousHandler(self):
        if self.currentEvent != 0:
            self.currentEvent -= 1
            self.updateBrowserEventInfomation()

    def buttonNextHandler(self):
        if self.currentEvent != self.EventSize:
            self.currentEvent += 1
            self.updateBrowserEventInfomation()

    def setBrowserEventImage(self, event, eventno):
        root = self.OutputDirectory + "/SpectrogramMarked/"
        path  = root + event + "/Event" + eventno + ".png"
        eventImage = QtGui.QPixmap(path)
        self.ui.label_imageshow_browser.setPixmap(eventImage)

    def updateBrowserEventInfomation(self):
        data = self.HDFFile[str(self.pathcorr[self.currentEvent])]
        self.setBrowserEventImage(self.file[self.currentEvent], self.eventno[self.currentEvent])
        dateToShow = self.day[self.currentEvent] + "-" + self.month[self.currentEvent] + "-" + self.year[self.currentEvent]
        self.ui.label_Date.setText(dateToShow)

        # Recontruction of original filename
        originalFilename = "sr_500000_ch_4_offset_" + str(data.attrs["Offset"])
        self.ui.label_browser_currentstatus.setText("Current Event Status: " + str(self.currentEvent + 1) + " out of " + str(self.EventSize) + ". Process Count: " + str(self.ProcessCount))
        self.ui.label_EventName_3.setText(originalFilename)
        self.ui.label_EventNo_3.setText(self.eventno[self.currentEvent])

        self.ui.label_time_3.setText(self.timeHandler(data))
        minFreq = tokFreq(data[0])
        maxFreq = tokFreq(data[1])
        duration = str(toTime(abs(data[2]-data[3])))
        self.ui.label_MinFreq_3.setText(str(minFreq))
        self.ui.label_maxFreq_3.setText(str(maxFreq))

        self.ui.label_Duration_3.setText(duration)
        self.ui.label_FrontLine_23.setText(str(data[4]))
        self.ui.label_FrontLine_24.setText(str(data[5]))
        self.ui.label_FrontLine_25.setText(str(data[6]))
        self.ui.label_FrontLine_26.setText(str(data[7]))
        self.ui.label_FrontLine_27.setText(str(data[8]))
        self.ui.label_FrontLine_28.setText(str(data[9]))
        self.ui.label_FrontLine_29.setText(str(data[10]))
        self.ui.label_FrontLine_30.setText(str(data[11]))
        self.ui.label_FrontLine_31.setText(str(data[12]))
        self.ui.label_FrontLine_32.setText(str(data[13]))
        self.ui.label_FrontLine_33.setText(str(data[14]))

    def file_dialog(self):
        self.filepath = QtGui.QFileDialog.getOpenFileName(self, "Open HDF5 File",'', "HDF5 Files (*.hdf5 *.h5)")

        from os.path import isfile
        if isfile(self.filepath):

            self.HDFFile = h5py.File(str(self.filepath))
            self.HDFFile.visit(self.saveEventPath)
            ###############################################
            ###############################################CHANGES!!
            ###############################################
            #self.day, self.month, self.year, self.file, self.eventno, self.pathcorr = self.getSpecificHDFInformation(self.pathEventList, 9)
            self.day, self.month, self.year, self.file, self.eventno, self.pathcorr = self.getHDFInformation(self.pathEventList)
            self.EventSize = len(self.day)
            self.currentEvent = 0
            self.ui.progressBar_eventLabel.setMinimum(self.currentEvent)
            self.ui.progressBar_eventLabel.setMaximum(self.EventSize-1)
            thisFilePath, thisFile = os.path.split(str(self.filepath))
            self.OutputDirectory = thisFilePath
            self.ui.label_outputDirectory.setText(self.OutputDirectory)
            self.ui.label_database_name.setText(thisFile)
            self.ui.frame_BatButtons.show()
            self.scanForNextEvent()
            self.updateEventInfomation()
        else:
            self.ui.label_database_name.setText("None selected")

    def create_spectrogram(self):
        SampleRate = self.ui.spinBox_SampleRate.value()
        SearchDirectory = self.InputDirectory + "/"
        SaveDirectory = self.OutputDirectory + "/"
        channel = 1 # default
        #Check radioButton for which channel we should make spectrogram of
        if self.ui.radioButton_channel_1.isChecked():
            channel = 1
        if self.ui.radioButton_channel_2.isChecked():
            channel = 2
        if self.ui.radioButton_channel_3.isChecked():
            channel = 3
        if self.ui.radioButton_channel_4.isChecked():
            channel = 4
        self.SoundFileList = getFunctions.getFileListDepthScan(SearchDirectory, ".s16")
        self.ui.progressBar_analyse.setMinimum(0)
        self.ui.progressBar_analyse.setMaximum(len(self.SoundFileList))


        Count = 0
        for soundfile in self.SoundFileList:
            self.ui.textEdit_overview.setText("Creating Spectrogram for " + soundfile + " at channel " + str(channel))
            EventExtraction.createSpectrogram(soundfile, SearchDirectory, SaveDirectory, channel, SampleRate)
            self.ui.progressBar_analyse.setValue(Count)
            Count += 1
        self.ui.progressBar_analyse.setValue(Count)
        self.ui.textEdit_overview.setText("Creating Spectrogram Done!")


    def run_analyser(self):
        rootpath = self.OutputDirectory

        recordedAt = str(self.ui.lineEdit_recordedAt.text())

        SearchPath = rootpath + "/Spectrogram/"
        SavePath = rootpath + "/SpectrogramMarked/"
        try:
            self.ui.textEdit_overview.setText("Loading Spectrogram Files...")
            sampleList = getFunctions.getFileList(SearchPath,".png")
            self.ui.textEdit_overview.setText("Loading Spectrogram Files... Done! Found " + str(len(sampleList)) + " files")
            maxSize = len(sampleList)
            self.ui.progressBar_analyse.setMinimum(0)
            self.ui.progressBar_analyse.setMaximum(maxSize)
            progressCount = 0

            for eventFile in sampleList:
                self.ui.textEdit_overview.setText("Analyzing " + os.path.splitext((eventFile))[0] + "\n")
                self.ui.label_FilesFoundProgress.setText(str(progressCount) + " out of " + str(maxSize))
                EventExtraction.findEvent(self.OutputDirectory, eventFile, recordedAt)
                progressCount += 1
                self.ui.progressBar_analyse.setValue(progressCount)
            self.ui.textEdit_overview.setText("Event extraction done!")
        except:
            self.ui.textEdit_overview.setText("Loading Failed! Make sure the Input / Output directory are set correct")



if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	myapp = StartQT4()
	myapp.show()
	sys.exit(app.exec_())