__author__ = 'Anochjhn Iruthayam'


import sys
from PyQt4 import QtCore, QtGui
from BatWindow import Ui_BatWindow
import EventExtraction, os, getFunctions, time
import h5py, re
import ClassifierThirdStage, ClassifierSecondStage, HDF5Handler, cv2, ClassifierFirstStage, ClassifierConnected
import numpy as np

def toTime(timePixel):
    imageLength = 5000.0
    return (1000.0/imageLength)*timePixel

def tokFreq(freqPixel):
    imageWidth = 1025.0
    return (250.0/imageWidth)*(imageWidth-freqPixel)

class GenerateSpecThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

    def setup(self,SoundFileList, SearchDirectory, SaveDirectory, channel, SampleRate):
        self.SearchDirectory = SearchDirectory
        self.SaveDirectory = SaveDirectory
        self.channel = channel
        self.SampleRate = SampleRate
        self.SoundFileList = SoundFileList

    def __del__(self):
        self.wait()

    def run(self):
        for soundfile in self.SoundFileList:
            EventExtraction.createSpectrogram(soundfile, self.SearchDirectory, self.SaveDirectory, self.channel, self.SampleRate)
            self.emit(QtCore.SIGNAL("progGenSpec()"))

class AnalyzeThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

    def setup(self,OutputDirectory, sampleList, recordedAt, projectName, InputDirectory):
        self.OutputDirectory = OutputDirectory
        self.sampleList = sampleList
        self.recordedAt = recordedAt
        self.projectName = projectName
        self.InputDirectory = InputDirectory

    def __del__(self):
        self.wait()

    def run(self):
        for eventFile in self.sampleList:
            EventExtraction.findEvent(self.OutputDirectory, eventFile, self.recordedAt, self.projectName, self.InputDirectory)
            self.emit(QtCore.SIGNAL("analyzeSpec()"))

class ClassifierThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

    def setup(self, classifier, databasePath, iteration, learningrate, momentum, toFile):
        self.classifier = classifier
        self.databasePath = databasePath
        self.iteration = iteration
        self.learningrate = learningrate
        self.momentum = momentum
        self.toFile = toFile

    def __del__(self):
        self.wait()

    def run(self):
        self.emit(QtCore.SIGNAL("clasProg()"))
        self.classifier.initClasissifer(self.databasePath)
        self.emit(QtCore.SIGNAL("train()"))
        self.classifier.goClassifer(self.iteration, self.learningrate, self.momentum, self.toFile)

class ClassifierRunThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

    def setup(self, function, databasePath):
        self.function = function
        self.databasePath = databasePath

    def __del__(self):
        self.wait()

    def run(self):
        self.emit(QtCore.SIGNAL("clasProg()"))
        self.function.initClasissifer(self.databasePath)
        self.emit(QtCore.SIGNAL("classRun()"))
        ConfusionMatrix = self.function.runClassifier()
        #ConfusionMatrix = self.function.ConfusionMatrix
        self.emit(QtCore.SIGNAL("output(PyQt_PyObject)"), ConfusionMatrix)

class ClassifierConnectedRunThread(QtCore.QThread):
    def __init__(self):
        QtCore.QThread.__init__(self)

    def setup(self, function, databasePath):
        self.function = function
        self.databasePath = databasePath

    def __del__(self):
        self.wait()

    def run(self):
        self.emit(QtCore.SIGNAL("clasProg()"))
        self.function.initClasissifer(self.databasePath)
        self.emit(QtCore.SIGNAL("classFirst()"))
        self.function.runFirstStageClassifier()
        self.emit(QtCore.SIGNAL("classSecond()"))
        self.function.runSecondStageClassifier()
        self.emit(QtCore.SIGNAL("classThird()"))
        self.function.runThirdStageClassifier()
        ConfusionMatrix = self.function.ConfusionMatrix
        self.emit(QtCore.SIGNAL("output(PyQt_PyObject)"), np.array(ConfusionMatrix))

class StartQT4(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_BatWindow()
        self.ui.setupUi(self)
        self.ui.frame_BatButtons.hide()
        self.pathEventList = []
        self.threadPool = []

        ### FIXES WINDOWS TASKBAR HANDLER, WHICH SHOWS THE CORRECT ICON ###
        import ctypes
        myappid = 'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


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
        #Connect train classifier buttons
        QtCore.QObject.connect(self.ui.button_classifierFirstStage_train, QtCore.SIGNAL("clicked()"), self.trainFirstStageClassifier)
        QtCore.QObject.connect(self.ui.button_classifierSecondStage_train, QtCore.SIGNAL("clicked()"), self.trainSecondStageClassifier)
        QtCore.QObject.connect(self.ui.button_classifierThirdStage_train, QtCore.SIGNAL("clicked()"), self.trainThirdStageClassifier)

        #Connect run classifier buttons
        QtCore.QObject.connect(self.ui.button_classifierFirstStage_run, QtCore.SIGNAL("clicked()"), self.runFirstStageClassifier)
        QtCore.QObject.connect(self.ui.button_classifierSecondStage_run, QtCore.SIGNAL("clicked()"), self.runSecondStageClassifier)
        QtCore.QObject.connect(self.ui.button_classifierThirdStage_run, QtCore.SIGNAL("clicked()"), self.runThirdStageClassifier)
        QtCore.QObject.connect(self.ui.button_classiferConnected_run, QtCore.SIGNAL("clicked()"), self.runConnectedClassifiers)

        QtCore.QObject.connect(self.ui.button_classifier_database, QtCore.SIGNAL("clicked()"), self.file_dialog_classifier)

        QtCore.QObject.connect(self.ui.button_loaddatabaseReconstruct, QtCore.SIGNAL("clicked()"), self.file_dialog2)
        QtCore.QObject.connect(self.ui.pushButton_SetOutputDirectory_Reconstruct, QtCore.SIGNAL("clicked()"), self.setOutputDirectory)
        QtCore.QObject.connect(self.ui.button_Recontructor, QtCore.SIGNAL("clicked()"), self.imageRecontructor)
        self.ui.progressBar_classifier.hide()
        ### GENERATE SPECTROGRAM THREAD ###
        self.specGenThread = GenerateSpecThread()
        self.connect(self.specGenThread, QtCore.SIGNAL("started()"), self.update_SpecProgStarted)
        self.connect(self.specGenThread, QtCore.SIGNAL("finished()"), self.update_SpecProgFinished)
        self.connect(self.specGenThread, QtCore.SIGNAL("progGenSpec()"), self.update_SpecProg)

        #### ANALYZE THREAD ###
        self.analyzeThread = AnalyzeThread()
        self.connect(self.analyzeThread, QtCore.SIGNAL("started()"), self.update_analyzeProgStarted)
        self.connect(self.analyzeThread, QtCore.SIGNAL("finished()"), self.update_analyzeProgFinished)
        self.connect(self.analyzeThread, QtCore.SIGNAL("analyzeSpec()"), self.update_analyzeProg)

        ### CLASSIFIER THREAD ###
        self.classifierThread = ClassifierThread()
        self.connect(self.classifierThread, QtCore.SIGNAL("started()"), self.update_disableAllButtons)
        self.connect(self.classifierThread, QtCore.SIGNAL("finished()"), self.update_classifierProgFinished)
        self.connect(self.classifierThread, QtCore.SIGNAL("clasProg()"), self.update_classifierProgStarted)

        self.connect(self.classifierThread, QtCore.SIGNAL("train()"), self.update_classfierInfo)

        ### CLASSIFIER RUN THREAD ###
        self.classifierRunThreadFSC = ClassifierRunThread()
        self.connect(self.classifierRunThreadFSC, QtCore.SIGNAL("finished()"), self.update_classifierRunProg)
        self.connect(self.classifierRunThreadFSC, QtCore.SIGNAL("started()"), self.update_disableAllButtons)
        self.connect(self.classifierRunThreadFSC, QtCore.SIGNAL("clasProg()"), self.update_classifierProgStarted)
        self.connect(self.classifierRunThreadFSC, QtCore.SIGNAL("classRun()"), self.update_classfierRunInfo)
        self.connect(self.classifierRunThreadFSC, QtCore.SIGNAL("output(PyQt_PyObject)"), self.tableConfusionMatrixHandlerFSC)

        self.classifierRunThreadSSC = ClassifierRunThread()
        self.connect(self.classifierRunThreadSSC, QtCore.SIGNAL("finished()"), self.update_classifierRunProg)
        self.connect(self.classifierRunThreadSSC, QtCore.SIGNAL("started()"), self.update_disableAllButtons)
        self.connect(self.classifierRunThreadSSC, QtCore.SIGNAL("clasProg()"), self.update_classifierProgStarted)
        self.connect(self.classifierRunThreadSSC, QtCore.SIGNAL("classRun()"), self.update_classfierRunInfo)
        self.connect(self.classifierRunThreadSSC, QtCore.SIGNAL("output(PyQt_PyObject)"), self.tableConfusionMatrixHandlerSSC)

        self.classifierRunThreadTSC = ClassifierRunThread()
        self.connect(self.classifierRunThreadTSC, QtCore.SIGNAL("finished()"), self.update_classifierRunProg)
        self.connect(self.classifierRunThreadTSC, QtCore.SIGNAL("started()"), self.update_disableAllButtons)
        self.connect(self.classifierRunThreadTSC, QtCore.SIGNAL("clasProg()"), self.update_classifierProgStarted)
        self.connect(self.classifierRunThreadTSC, QtCore.SIGNAL("classRun()"), self.update_classfierRunInfo)
        self.connect(self.classifierRunThreadTSC, QtCore.SIGNAL("output(PyQt_PyObject)"), self.tableConfusionMatrixHandlerTSC)

        self.classifierConnectedRunThread = ClassifierConnectedRunThread()
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("finished()"), self.update_classifierRunProg)
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("started()"), self.update_disableAllButtons)
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("clasProg()"), self.update_classifierProgStarted)
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("classFirst()"), self.update_first)
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("classSecond()"), self.update_second)
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("classThird()"), self.update_third)
        self.connect(self.classifierConnectedRunThread, QtCore.SIGNAL("output(PyQt_PyObject)"), self.tableConfusionMatrixHandlerTSC)


        #QtCore.QObject.connect(self.ui.progressBar)
        self.HDFFile = h5py
        self.EventSize = 0
        self.currentEvent = 0
        self.previousEvent = 0
        self.ProcessCount = 0
        self.MultiCount = 0
        self.SpecProgressCount = 0
        self.analyzeProgessCount = 0
        self.day = []
        self.month = []
        self.year = []
        self.file = []
        self.pathcorr = []
        self.eventno = []
        self.SoundFileList = []
        self.OutputDirectory = "C:\Users\Anoch\Documents\PreOutput"#"/home/anoch/Documents/BatSamplesOutput"
        self.InputDirectory = "C:\Users\Anoch\Documents\PreInput"#"/home/anoch/Documents/BatSamplesInput"
        self.DatabasePath = "C:\Users\Anoch\Documents\BatOutput\BatData.hdf5"#"/home/anoch/Documents/BatOutput/BatData.hdf5"
        self.ui.label_classifier_databaseDirectory.setText(self.DatabasePath)
        self.ui.label_outputDirectory.setText(self.OutputDirectory)
        self.ui.label_inputDirectory.setText(self.InputDirectory)
        self.third_stage_classifier = ClassifierThirdStage.Classifier()
        self.second_stage_classifier = ClassifierSecondStage.Classifier()

        self.first_stage_classifier = ClassifierFirstStage.BinaryClassifier()
        self.second_stage_classifier = ClassifierSecondStage.Classifier()
        self.third_stage_classifier = ClassifierThirdStage.Classifier()

        self.first_stage_classifier_run = ClassifierFirstStage.BinaryClassifier()
        self.second_stage_classifier_run = ClassifierSecondStage.Classifier()
        self.third_stage_classifier_run = ClassifierThirdStage.Classifier()

        self.connected_classifier_run = ClassifierConnected.ClassifierConnected()


        self.ui.button_OtherSpecies.hide()
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

    def imageRecontructor(self):
        if self.ui.tabWidget.currentIndex() == 3:
            day, month, year, file, eventno, pathcorr = self.getHDFInformationRecontructImage(self.pathEventList, 1)
            max = len(pathcorr)
            for i in range(0, max):
                Imgdata = self.HDFFile[pathcorr[i]]
                image = HDF5Handler.imageRecontructFromHDF5(Imgdata)
                cv2.imwrite(self.OutputDirectory + "/Spectrogram/" + file[i] + ".png", image)

    def update_classifierProgStarted(self):
        self.ui.textEdit_classifier_overview.setText("Initilazing database")
        self.ui.progressBar_classifier.show()
    def update_classifierProgFinished(self):
        self.ui.button_classifier_database.setEnabled(True)
        self.ui.button_classiferConnected_run.setEnabled(True)
        self.ui.button_classifierFirstStage_train.setEnabled(True)
        self.ui.button_classifierFirstStage_run.setEnabled(True)
        self.ui.button_classifierSecondStage_train.setEnabled(True)
        self.ui.button_classifierSecondStage_run.setEnabled(True)
        self.ui.button_classifierThirdStage_train.setEnabled(True)
        self.ui.button_classifierThirdStage_run.setEnabled(True)
        self.ui.textEdit_classifier_overview.setText("Training network... Done")
        self.ui.progressBar_classifier.hide()
    def update_classifierRunProg(self):
        self.ui.button_classifier_database.setEnabled(True)
        self.ui.button_classiferConnected_run.setEnabled(True)
        self.ui.button_classifierFirstStage_train.setEnabled(True)
        self.ui.button_classifierFirstStage_run.setEnabled(True)
        self.ui.button_classifierSecondStage_train.setEnabled(True)
        self.ui.button_classifierSecondStage_run.setEnabled(True)
        self.ui.button_classifierThirdStage_train.setEnabled(True)
        self.ui.button_classifierThirdStage_run.setEnabled(True)
        self.ui.textEdit_classifier_overview.setText("Done!")
        self.ui.progressBar_classifier.hide()
    def update_classfierInfo(self):
        self.ui.textEdit_classifier_overview.setText("Training network...")
    def update_classfierRunInfo(self):
        self.ui.textEdit_classifier_overview.setText("Running network...")
    def update_first(self):
        self.ui.textEdit_classifier_overview.setText("Running first stage network...")
    def update_second(self):
        self.ui.textEdit_classifier_overview.setText("Running second stage network...")
    def update_third(self):
        self.ui.textEdit_classifier_overview.setText("Running third stage network...")
    def update_disableAllButtons(self):
        self.ui.button_classifier_database.setEnabled(False)
        self.ui.button_classiferConnected_run.setEnabled(False)
        self.ui.button_classifierFirstStage_train.setEnabled(False)
        self.ui.button_classifierFirstStage_run.setEnabled(False)
        self.ui.button_classifierSecondStage_train.setEnabled(False)
        self.ui.button_classifierSecondStage_run.setEnabled(False)
        self.ui.button_classifierThirdStage_train.setEnabled(False)
        self.ui.button_classifierThirdStage_run.setEnabled(False)

    def trainFirstStageClassifier(self):

        self.classifierThread.setup(self.first_stage_classifier, self.DatabasePath, 0, 0.001,0.01, False)
        self.classifierThread.start()
        #self.first_stage_classifier.initClasissifer(self.DatabasePath)
        #self.ui.textEdit_classifier_overview.setText("Training network...")
        #self.first_stage_classifier.goClassifer(0, 0.001, 0.1, False)
        #self.ui.textEdit_classifier_overview.setText("Training network... Done")
        ## Testing Purpose ##
        """
        learningRate    = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
        momentum        = [0.001, 0.01, 0.1, 0.001, 0.01, 0.1, 0.001, 0.01, 0.1]
        for setting in range (0, len(learningRate)):
            for i in range (0, 5):
                self.first_stage_classifier.goClassifer(i, learningRate[setting], momentum[setting], True)
        """
    def trainSecondStageClassifier(self):
        self.classifierThread.setup(self.second_stage_classifier, self.DatabasePath, 0,0.001,0.001,False)
        self.classifierThread.start()


        #self.ui.textEdit_classifier_overview.setText("Initilazing database")
        #self.second_stage_classifier.initClasissifer(self.DatabasePath)
        #self.ui.textEdit_classifier_overview.setText("Training network...")
        #self.second_stage_classifier.goClassifer(0,0.001,0.001,False)
        #self.ui.textEdit_classifier_overview.setText("Training network... Done")
        """
        learningRate    = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
        momentum        = [0.001, 0.01, 0.1, 0.001, 0.01, 0.1, 0.001, 0.01, 0.1]
        for setting in range (0, len(learningRate)):
            for i in range(0,5):
                self.second_stage_classifier.goClassifer(i, learningRate[setting], momentum[setting], True)
        """

    def trainThirdStageClassifier(self):
        self.classifierThread.setup(self.third_stage_classifier, self.DatabasePath, 0, 0.001, 0.010, False)
        self.classifierThread.start()



        #self.ui.textEdit_classifier_overview.setText("Initilazing database")
        #self.third_stage_classifier.initClasissifer(self.DatabasePath)
        #self.ui.textEdit_classifier_overview.setText("Training network...")
        #self.third_stage_classifier.goClassifer(0, 0.001, 0.010, False)
        #self.ui.textEdit_classifier_overview.setText("Training network... Done")
        """
        learningRate    = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
        momentum        = [0.001, 0.01, 0.1, 0.001, 0.01, 0.1, 0.001, 0.01, 0.1]
        for setting in range (0, len(learningRate)):
            for i in range(0,5):
                self.third_stage_classifier.goClassifer(i, learningRate[setting], momentum[setting], True)
        """

    def runFirstStageClassifier(self):
        #self.ui.textEdit_classifier_overview.setText("Initilazing database")
        #self.first_stage_classifier_run.initClasissifer(self.DatabasePath)
        self.classifierRunThreadFSC.setup(self.first_stage_classifier_run, self.DatabasePath)
        self.classifierRunThreadFSC.start()
        #self.classifierRunThread.wait()
        #ConfusionMatrix = self.first_stage_classifier_run.runClassifier()

        #self.tableConfusionMatrixHandlerFSC(ConfusionMatrix)
    def runSecondStageClassifier(self):
        self.classifierRunThreadSSC.setup(self.second_stage_classifier_run, self.DatabasePath)
        self.classifierRunThreadSSC.start()
        #self.second_stage_classifier_run.initClasissifer(self.DatabasePath)
        #ConfusionMatrix, BatTarget = self.second_stage_classifier_run.runClassifier()
        #cursor = QtGui.QTextCursor(self.ui.textEdit_classifier_overview.document())

        #cursor.insertText("Confusion Matrix\n" + str(ConfusionMatrix) + "\nTarget\n" + str(BatTarget))
        #self.tableConfusionMatrixHandlerSSC(ConfusionMatrix)
    def runThirdStageClassifier(self):
        self.classifierRunThreadTSC.setup(self.third_stage_classifier_run, self.DatabasePath)
        self.classifierRunThreadTSC.start()
        #ConfusionMatrix, BatTarget = self.third_stage_classifier_run.runClassifier()
        #cursor = QtGui.QTextCursor(self.ui.textEdit_classifier_overview.document())
        #cursor.insertText("Confusion Matrix\n" + str(ConfusionMatrix) + "\nTarget\n" + str(BatTarget))
        #self.tableConfusionMatrixHandlerTSC(ConfusionMatrix)

    def runConnectedClassifiers(self):
        self.classifierConnectedRunThread.setup(self.connected_classifier_run, self.DatabasePath)
        self.classifierConnectedRunThread.start()
        #self.connected_classifier_run.initClasissifer(self.DatabasePath)
        #ConfusionMatrix, BatTarget = self.connected_classifier_run.runClassifiers()
        #cursor = QtGui.QTextCursor(self.ui.textEdit_classifier_overview.document())
        #cursor.insertText("Confusion Matrix\n" + str(ConfusionMatrix) + "\nTarget\n" + str(BatTarget))

        #ConfusionMatrix = np.zeros((7,7))
        #ConfusionMatrix = np.array(ConfusionMatrix, dtype=np.int64)
        #count = 0
        #for row in range(0,7):
        #    for colomn in range(0,7):
        #        ConfusionMatrix[row][colomn] = count
        #        count += 1
        #self.tableConfusionMatrixHandlerTSC(ConfusionMatrix)

    def tableConfusionMatrixHandlerTSC(self, ConfusionMatrix):
        # SET UP Labels for the table
        self.ui.tableWidget_ConfusionMatrix.clear()
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("EpSe")
        self.ui.tableWidget_ConfusionMatrix.setItem(2,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("EpSe")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,2,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("PiPy")
        self.ui.tableWidget_ConfusionMatrix.setItem(3,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("PiPy")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,3,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("MyDau")
        self.ui.tableWidget_ConfusionMatrix.setItem(4,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("MyDau")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,4,data)


        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("PiNa")
        self.ui.tableWidget_ConfusionMatrix.setItem(5,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("PiNa")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,5,data)


        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("NyNo")
        self.ui.tableWidget_ConfusionMatrix.setItem(6,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("NyNo")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,6,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Noise")
        self.ui.tableWidget_ConfusionMatrix.setItem(7,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Noise")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,7,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("SeEl")
        self.ui.tableWidget_ConfusionMatrix.setItem(8,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("SeEl")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,8,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Target")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,9,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("BL [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,10,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("CCR [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,11,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Prec [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,12,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Reca [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,13,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Out")
        self.ui.tableWidget_ConfusionMatrix.setItem(0,6,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("True")
        self.ui.tableWidget_ConfusionMatrix.setItem(4,0,data)
        stringMatrix = str(ConfusionMatrix)
        print stringMatrix
        target = [0,0,0,0,0,0,0]
        RowMatrix = [0]
        for row in range(0,7):
            for colomn in range(0,7):
                ## WORK AROUND FOR BUG IN MILTIDIMENSION ARRAY
                RowMatrix[0] = ConfusionMatrix[row]
                target[row] += ConfusionMatrix[row][colomn]
                data = QtGui.QTableWidgetItem(str(ConfusionMatrix[row][colomn]))
                self.ui.tableWidget_ConfusionMatrix.setItem(row+2,colomn+2,data)
            data = QtGui.QTableWidgetItem(str(target[row]))
            self.ui.tableWidget_ConfusionMatrix.setItem(row+2,colomn+3,data)
        largestIndex = np.argmax(target)
        maxValue = target[largestIndex]
        sumTarget = sum(target)
        baseline =  (float(maxValue)/float(sumTarget))*100
        data = QtGui.QTableWidgetItem(str("%.2f" % baseline))
        self.ui.tableWidget_ConfusionMatrix.setItem(2,10,data)
        diagonalSum = 0
        for diag in range(0,7):
            diagonalSum += ConfusionMatrix[diag][diag]
        CCR = (float(diagonalSum)/float(sumTarget))*100
        data = QtGui.QTableWidgetItem(str("%.2f" % CCR))
        self.ui.tableWidget_ConfusionMatrix.setItem(2,11,data)

        classifierTarget = [0,0,0,0,0,0,0]
        precision = [0,0,0,0,0,0,0]
        for colomn in range(0,7):
            for row in range(0,7):
                classifierTarget[colomn] += ConfusionMatrix[row][colomn]
            precision = (float(ConfusionMatrix[colomn][colomn])/float(classifierTarget[colomn]))*100
            data = QtGui.QTableWidgetItem(str("%.2f" % precision))
            self.ui.tableWidget_ConfusionMatrix.setItem(colomn+2,12,data)

        for diag in range(0,7):
            recall = (float(ConfusionMatrix[diag][diag])/target[diag])*100
            data = QtGui.QTableWidgetItem(str("%.2f" % recall))
            self.ui.tableWidget_ConfusionMatrix.setItem(diag+2,13,data)

    def tableConfusionMatrixHandlerSSC(self, ConfusionMatrix):
        # SET UP Labels for the table
        self.ui.tableWidget_ConfusionMatrix.clear()
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Noise")
        self.ui.tableWidget_ConfusionMatrix.setItem(2,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Noise")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,2,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Single")
        self.ui.tableWidget_ConfusionMatrix.setItem(3,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Single")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,3,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Multi")
        self.ui.tableWidget_ConfusionMatrix.setItem(4,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Multi")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,4,data)


        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("SoEl")
        self.ui.tableWidget_ConfusionMatrix.setItem(5,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("SoEl")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,5,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Target")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,6,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("BL [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,7,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("CCR [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,8,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Prec [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,9,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Reca [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,10,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Out")
        self.ui.tableWidget_ConfusionMatrix.setItem(0,6,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("True")
        self.ui.tableWidget_ConfusionMatrix.setItem(4,0,data)

        target = [0,0,0,0]
        dimension = len(target)
        for row in range(0,dimension):
            for colomn in range(0,dimension):
                target[row] += ConfusionMatrix[row][colomn]
                data = QtGui.QTableWidgetItem(str(ConfusionMatrix[row][colomn]))
                self.ui.tableWidget_ConfusionMatrix.setItem(row+2,colomn+2,data)
            data = QtGui.QTableWidgetItem(str(target[row]))
            self.ui.tableWidget_ConfusionMatrix.setItem(row+2,colomn+3,data)

        # BASELINE
        largestIndex = np.argmax(target)
        maxValue = target[largestIndex]
        sumTarget = sum(target)
        baseline =  (float(maxValue)/float(sumTarget))*100
        data = QtGui.QTableWidgetItem(str("%.2f" % baseline))
        self.ui.tableWidget_ConfusionMatrix.setItem(2,7,data)

        # CCR
        diagonalSum = 0
        for diag in range(0,dimension):
            diagonalSum += ConfusionMatrix[diag][diag]
        CCR = (float(diagonalSum)/float(sumTarget))*100
        data = QtGui.QTableWidgetItem(str("%.2f" % CCR))
        self.ui.tableWidget_ConfusionMatrix.setItem(2,8,data)

        classifierTarget = [0,0,0,0]
        for colomn in range(0,dimension):
            for row in range(0,dimension):
                classifierTarget[colomn] += ConfusionMatrix[row][colomn]
            precision = (float(ConfusionMatrix[colomn][colomn])/float(classifierTarget[colomn]))*100
            data = QtGui.QTableWidgetItem(str("%.2f" % precision))
            self.ui.tableWidget_ConfusionMatrix.setItem(colomn+2,9,data)

        for diag in range(0,dimension):
            recall = (float(ConfusionMatrix[diag][diag])/target[diag])*100
            data = QtGui.QTableWidgetItem(str("%.2f" % recall))
            self.ui.tableWidget_ConfusionMatrix.setItem(diag+2,10,data)

    def tableConfusionMatrixHandlerFSC(self, ConfusionMatrix):
        # SET UP Labels for the table
        self.ui.tableWidget_ConfusionMatrix.clear()
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(True)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Bat")
        self.ui.tableWidget_ConfusionMatrix.setItem(2,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Bat")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,2,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Noise")
        self.ui.tableWidget_ConfusionMatrix.setItem(3,1,data)
        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Noise")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,3,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Target")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,4,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("BL [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,5,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("CCR [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,6,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Prec [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,7,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Reca [%]")
        self.ui.tableWidget_ConfusionMatrix.setItem(1,8,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("Out")
        self.ui.tableWidget_ConfusionMatrix.setItem(0,5,data)

        data = QtGui.QTableWidgetItem()
        data.setFont(font)
        data.setText("True")
        self.ui.tableWidget_ConfusionMatrix.setItem(2,0,data)

        target = [0,0]
        dimension = len(target)
        for row in range(0,dimension):
            for colomn in range(0,dimension):
                target[row] += ConfusionMatrix[row][colomn]
                data = QtGui.QTableWidgetItem(str(ConfusionMatrix[row][colomn]))
                self.ui.tableWidget_ConfusionMatrix.setItem(row+2,colomn+2,data)
            data = QtGui.QTableWidgetItem(str(target[row]))
            self.ui.tableWidget_ConfusionMatrix.setItem(row+2,colomn+3,data)

        # BASELINE
        largestIndex = np.argmax(target)
        maxValue = target[largestIndex]
        sumTarget = sum(target)
        baseline =  (float(maxValue)/float(sumTarget))*100
        data = QtGui.QTableWidgetItem(str("%.2f" % baseline))
        self.ui.tableWidget_ConfusionMatrix.setItem(2,5,data)

        # CCR
        diagonalSum = 0
        for diag in range(0,dimension):
            diagonalSum += ConfusionMatrix[diag][diag]
        CCR = (float(diagonalSum)/float(sumTarget))*100
        data = QtGui.QTableWidgetItem(str("%.2f" % CCR))
        self.ui.tableWidget_ConfusionMatrix.setItem(2,6,data)

        classifierTarget = [0,0]
        for colomn in range(0,dimension):
            for row in range(0,dimension):
                classifierTarget[colomn] += ConfusionMatrix[row][colomn]
            precision = (float(ConfusionMatrix[colomn][colomn])/float(classifierTarget[colomn]))*100
            data = QtGui.QTableWidgetItem(str("%.2f" % precision))
            self.ui.tableWidget_ConfusionMatrix.setItem(colomn+2,7,data)

        for diag in range(0,dimension):
            recall = (float(ConfusionMatrix[diag][diag])/target[diag])*100
            data = QtGui.QTableWidgetItem(str("%.2f" % recall))
            self.ui.tableWidget_ConfusionMatrix.setItem(diag+2,8,data)

    def file_dialog_classifier(self):
        filepath = QtGui.QFileDialog.getOpenFileName(self,"Open HDF5 File",'', "HDF5 Files (*.hdf5 *.h5)")

        from os.path import isfile
        if isfile(filepath):
            filename = filepath
            self.DatabasePath = str(filename)
            self.ui.label_classifier_databaseDirectory.setText(filename)
            #self.HDFFile = h5py.File(str(filepath))
            #self.HDFFile.visit(self.saveEventPath)
        else:
            self.ui.label_classifier_databaseDirectory.setText("None selected")

    def getHDFInformationRecontructImage(self, paths, imgType):
        day = []
        month = []
        year = []
        file = []
        pathcorr = []
        eventno = []
        for path in paths:
            temp = re.split('/', path)
            # if there are 5 elements in the array, means that this one has an event
            if imgType == 1:
                lookFor = "ArrayImgSpectrogram"
                index = 4
                length = 5
            if imgType == 2:
                lookFor = "ArrayImgMarkedSpectrogram"
                index = 4
                length = 5
            if imgType == 3:
                lookFor = "ArrayImgEvent"
                index = 5
                length = 6
            if len(temp) == length and temp[index] == lookFor:
                #get data from path
                year.append(temp[2])
                month.append(temp[3])
                day.append(temp[4])
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
                tempOffset = re.split('_', temp[5])
                offset = tempOffset[1]
                #if len(offset) != 20:
                #    for i in range(0,20):
                #        offset = "0" + offset
                #        if len(offset) == 20:
                #            break
                filename = "date_" + temp[4] + "_" + temp[3] + "_" + temp[2] + "_time_" + hour + "_" + minute + "_" + second +"_ch_" + channel +  "_offset_" + offset
                file.append(filename)
                #file.append(temp[3])
                eventnoTemp = re.split('_',temp[6])
                eventno.append(eventnoTemp[1])
                pathcorr.append(path)

        return day, month, year, file, eventno, pathcorr

    def getHDFInformation(self, paths):
        day = []
        month = []
        year = []
        file = []
        pathcorr = []
        eventno = []
        for path in paths:
            temp = re.split('/', path)
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                #get data from path
                year.append(temp[2])
                month.append(temp[3])
                day.append(temp[4])
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
                tempOffset = re.split('_', temp[5])
                offset = tempOffset[1]
                #if len(offset) != 20:
                #    for i in range(0,20):
                #        offset = "0" + offset
                #        if len(offset) == 20:
                #            break
                filename = "date_" + temp[4] + "_" + temp[3] + "_" + temp[2] + "_time_" + hour + "_" + minute + "_" + second +"_ch_" + channel +  "_offset_" + offset
                file.append(filename)
                #file.append(temp[3])
                eventnoTemp = re.split('_',temp[6])
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
            index = 7
            length = 8
            if len(temp) == length and temp[index] == "FeatureDataEvent":
                #get data from path
                data = self.HDFFile[path]
                if data.attrs["BatID"] == BatID:
                    year.append(temp[2])
                    month.append(temp[3])
                    day.append(temp[4])
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
                    tempOffset = re.split('_', temp[5])
                    offset = tempOffset[1]
                    #if len(offset) != 20:
                    #    for i in range(0,20):
                    #        offset = "0" + offset
                    #        if len(offset) == 20:
                    #            break
                    filename = "date_" + temp[4] + "_" + temp[3] + "_" + temp[2] + "_time_" + hour + "_" + minute + "_" + second +"_ch_" + channel +  "_offset_" + offset
                    file.append(filename)
                    #file.append(temp[3])
                    eventnoTemp = re.split('_',temp[6])
                    eventno.append(eventnoTemp[1])
                    pathcorr.append(path)

        return day, month, year, file, eventno, pathcorr

    def setEventImage(self, event, eventno):
        root = self.OutputDirectory + "/SpectrogramMarked/"
        pathTemp  = root + event + "/Event" + eventno + ".png"
        from os.path import isfile
        if isfile(pathTemp):
            path = pathTemp
        else:
            #get one hour UP
            TimeTemp = re.split('_', event)
            hour = int(TimeTemp[5]) + 1
            strHour = str(hour)
            newEvent = TimeTemp[0] + "_" + TimeTemp[1] + "_" + TimeTemp[2] + "_" + TimeTemp[3] + "_" + TimeTemp[4] + "_" + strHour + "_" + TimeTemp[6] + "_" + TimeTemp[7] + "_" + TimeTemp[8] + "_" + TimeTemp[9] + "_" + TimeTemp[10] + "_" + TimeTemp[11]
            pathTemp  = root + newEvent + "/Event" + eventno + ".png"
            if isfile(pathTemp):
                path = pathTemp
            else:
                # Get one hour DOWN
                TimeTemp = re.split('_', event)
                hour = int(TimeTemp[5]) - 1
                strHour = str(hour)
                newEvent = TimeTemp[0] + "_" + TimeTemp[1] + "_" + TimeTemp[2] + "_" + TimeTemp[3] + "_" + TimeTemp[4] + "_" + strHour + "_" + TimeTemp[6] + "_" + TimeTemp[7] + "_" + TimeTemp[8] + "_" + TimeTemp[9] + "_" + TimeTemp[10] + "_" + TimeTemp[11]
                path  = root + newEvent + "/Event" + eventno + ".png"
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
        self.ui.progressBar_eventLabel.setValue(self.currentEvent+1)
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
        pathTemp  = root + event + "/Event" + eventno + ".png"
        from os.path import isfile
        if isfile(pathTemp):
            path = pathTemp
        else:
            #get one hour UP
            TimeTemp = re.split('_', event)
            hour = int(TimeTemp[5]) + 1
            strHour = str(hour)
            newEvent = TimeTemp[0] + "_" + TimeTemp[1] + "_" + TimeTemp[2] + "_" + TimeTemp[3] + "_" + TimeTemp[4] + "_" + strHour + "_" + TimeTemp[6] + "_" + TimeTemp[7] + "_" + TimeTemp[8] + "_" + TimeTemp[9] + "_" + TimeTemp[10] + "_" + TimeTemp[11]
            pathTemp  = root + newEvent + "/Event" + eventno + ".png"
            if isfile(pathTemp):
                path = pathTemp
            else:
                # Get one hour DOWN
                TimeTemp = re.split('_', event)
                hour = int(TimeTemp[5]) - 1
                strHour = str(hour)
                newEvent = TimeTemp[0] + "_" + TimeTemp[1] + "_" + TimeTemp[2] + "_" + TimeTemp[3] + "_" + TimeTemp[4] + "_" + strHour + "_" + TimeTemp[6] + "_" + TimeTemp[7] + "_" + TimeTemp[8] + "_" + TimeTemp[9] + "_" + TimeTemp[10] + "_" + TimeTemp[11]
                path  = root + newEvent + "/Event" + eventno + ".png"

        eventImage = QtGui.QPixmap(path)
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

    def file_dialog2(self):
        filepath = QtGui.QFileDialog.getOpenFileName(self,"Open HDF5 File",'', "HDF5 Files (*.hdf5 *.h5)")

        from os.path import isfile
        if isfile(filepath):

            self.HDFFile = h5py.File(str(filepath))
            self.HDFFile.visit(self.saveEventPath)
        else:
            self.ui.textEdit_overview_Recontructor.setText("None selected")

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
            self.ui.button_OtherSpecies.hide()
            self.scanForNextEvent()
            self.updateEventInfomation()
        else:
            self.ui.label_database_name.setText("None selected")


    def create_spectrogramBACKUP(self):
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
        first = True
        for soundfile in self.SoundFileList:
            self.ui.textEdit_overview.setText("Creating Spectrogram for " + soundfile + " at channel " + str(channel))
            #self.threadPool.append(GenericThread(EventExtraction.createSpectrogram, soundfile, SearchDirectory, SaveDirectory, channel, SampleRate))
            #worker = GenericThread(EventExtraction.createSpectrogram, soundfile, SearchDirectory, SaveDirectory, channel, SampleRate)
            #worker.start()
            print "Thread setup"

            self.specGenThread.setup(EventExtraction.createSpectrogram, soundfile, SearchDirectory, SaveDirectory, channel, SampleRate)
            print "Thread start"
            self.specGenThread.start()
            print "Thread wait"
            self.specGenThread.wait()
            print "Thread finished"
            #worker.wait()
            #worker.run(EventExtraction.createSpectrogram(soundfile, SearchDirectory, SaveDirectory, channel, SampleRate))
            #self.threadPool[len(self.threadPool)-1].start()
            #self.connect(worker, QtCore.SIGNAL("finished()"), self.update_SpecProg)
            #if worker.isFinished():
            #    worker.finished()
            #    self.ui.progressBar_analyse.setValue(Count)
            #    Count += 1
        self.SpecProgressCount += 1
        self.ui.progressBar_analyse.setValue(self.SpecProgressCount)
        self.ui.textEdit_overview.setText("Creating Spectrogram Done!")

    def create_spectrogram(self):
        self.ui.pushButton_createSpectrogram.setEnabled(False)
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
        self.specGenThread.setup(self.SoundFileList, SearchDirectory, SaveDirectory, channel, SampleRate)
        self.specGenThread.start()

        Count = 0
        first = True

        #self.ui.progressBar_analyse.setValue(self.SpecProgressCount)
        #self.ui.textEdit_overview.setText("Creating Spectrogram Done!")

    def update_SpecProg(self):
        #print "Im updating"
        self.SpecProgressCount += 1
        self.ui.progressBar_analyse.setValue(self.SpecProgressCount)

    def update_SpecProgFinished(self):
        self.ui.pushButton_createSpectrogram.setEnabled(True)
        self.ui.textEdit_overview.setText("Creating Spectrogram Done!")
    def update_SpecProgStarted(self):
        self.ui.textEdit_overview.setText("Creating Spectrogram Started!")


    def run_analyser(self):
        self.ui.button_start.setEnabled(False)
        rootpath = self.OutputDirectory
        self.analyzeProgessCount
        recordedAt = str(self.ui.lineEdit_recordedAt.text())
        projectName = str(self.ui.lineEdit_projectName.text())

        SearchPath = rootpath + "/Spectrogram/"
        SavePath = rootpath + "/SpectrogramMarked/"
        try:
            #self.ui.textEdit_overview.setText("Loading Spectrogram Files...")
            sampleList = getFunctions.getFileList(SearchPath,".png")
            #self.ui.textEdit_overview.setText("Loading Spectrogram Files... Done! Found " + str(len(sampleList)) + " files")
            maxSize = len(sampleList)
            self.ui.progressBar_analyse.setMinimum(0)
            self.ui.progressBar_analyse.setMaximum(maxSize)
            progressCount = 0
            self.analyzeThread.setup(self.OutputDirectory, sampleList, recordedAt, projectName, self.InputDirectory)
            self.analyzeThread.start()

            #for eventFile in sampleList:
            #    self.ui.textEdit_overview.setText("Analyzing " + os.path.splitext((eventFile))[0] + "\n")
            #    self.ui.label_FilesFoundProgress.setText(str(progressCount) + " out of " + str(maxSize))

            #    EventExtraction.findEvent(self.OutputDirectory, eventFile, recordedAt, projectName, self.InputDirectory)
            #    progressCount += 1
            #    self.ui.progressBar_analyse.setValue(progressCount)
            #self.ui.textEdit_overview.setText("Event extraction done!")
        except:
            self.ui.textEdit_overview.setText("Loading Failed! Make sure the Input / Output directory are set correct")

    def update_analyzeProg(self):
        self.analyzeProgessCount += 1
        self.ui.progressBar_analyse.setValue(self.analyzeProgessCount)

    def update_analyzeProgFinished(self):
        self.ui.button_start.setEnabled(True)
        self.analyzeProgessCount += 1
        self.ui.progressBar_analyse.setValue(self.analyzeProgessCount)
        self.ui.textEdit_overview.setText("Event extraction done!")

    def update_analyzeProgStarted(self):
        self.ui.textEdit_overview.setText("Event extraction Started!")



if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	myapp = StartQT4()
	myapp.show()
	sys.exit(app.exec_())