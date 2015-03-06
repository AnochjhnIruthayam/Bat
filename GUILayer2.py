__author__ = 'Anochjhn Iruthayam'


import sys
from PyQt4 import QtCore, QtGui
from BatWindow import Ui_BatWindow
import EventExtraction, os, getFunctions, time
import h5py, re


def toTime(timePixel):
    imageLength = 5000.0
    return (1000.0/imageLength)*timePixel

def tokFreq(freqPixel):
    imageWidth = 1025.0
    return (250.0/imageWidth)*(imageWidth-freqPixel)



def getHDFInformation(paths):
    day = []
    month = []
    year = []
    file = []
    pathcorr = []
    eventno = []
    for path in paths:
        temp = re.split('/', path)
        # if there are 5 elements in the array, means that this one has an event
        if len(temp) > 4:
            #get data from path
            year.append(temp[0])
            month.append(temp[1])
            day.append(temp[2])
            file.append(temp[3])
            eventnoTemp = re.split('_',temp[4])
            eventno.append(eventnoTemp[1])
            pathcorr.append(path)

    return day, month, year, file, eventno, pathcorr



class StartQT4(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_BatWindow()
        self.ui.setupUi(self)
        self.ui.frame_BatButtons.hide()
        self.pathEventList = []
        QtCore.QObject.connect(self.ui.button_start, QtCore.SIGNAL("clicked()"), self.run_analyser)
        QtCore.QObject.connect(self.ui.button_loaddatabase, QtCore.SIGNAL("clicked()"), self.file_dialog)
        QtCore.QObject.connect(self.ui.button_EptesicusSerotinus, QtCore.SIGNAL("clicked()"), self.getValueEptesicusSerotinus)
        QtCore.QObject.connect(self.ui.button_PipistrellusPygmaeus, QtCore.SIGNAL("clicked()"), self.getValuePipistrellusPygmaeus)
        QtCore.QObject.connect(self.ui.button_MyotisDaubentonii, QtCore.SIGNAL("clicked()"), self.getValueMyotisDaubentonii)
        QtCore.QObject.connect(self.ui.button_MyotisDasycneme, QtCore.SIGNAL("clicked()"), self.getValueMyotisDasycneme)
        QtCore.QObject.connect(self.ui.button_pipistrellusNathusii, QtCore.SIGNAL("clicked()"), self.getValuePipistrellusNathusii)
        QtCore.QObject.connect(self.ui.button_NyctalusNoctula, QtCore.SIGNAL("clicked()"), self.getValueNyctalusNoctula)
        QtCore.QObject.connect(self.ui.button_noise, QtCore.SIGNAL("clicked()"), self.getValueNoise)
        QtCore.QObject.connect(self.ui.button_OtherSpecies, QtCore.SIGNAL("clicked()"), self.getValueOtherSpecies)

        #QtCore.QObject.connect(self.ui.progressBar)
        self.HDFFile = h5py
        self.EventSize = 0
        self.currentEvent = 0
        self.day = []
        self.month = []
        self.year = []
        self.file = []
        self.pathcorr = []
        self.eventno = []

    def labelBat(self, ID):
        data = self.HDFFile[str(self.pathcorr[self.currentEvent])]
        data.attrs['BatID'] = ID
        self.scanForNextEvent()
        self.updateEventInfomation()

    def getValueEptesicusSerotinus(self):
        self.labelBat(1)
    def getValueMyotisDasycneme(self):
        self.labelBat(2)
    def getValueMyotisDaubentonii(self):
        self.labelBat(3)
    def getValueNyctalusNoctula(self):
        self.labelBat(4)
    def getValuePipistrellusNathusii(self):
        self.labelBat(5)
    def getValuePipistrellusPygmaeus(self):
        self.labelBat(6)
    def getValueOtherSpecies(self):
        self.labelBat(7)
    def getValueNoise(self):
        self.labelBat(8)

    def saveEventPath(self,name):
        self.pathEventList.append(name)

    def setEventImage(self, event, eventno):
        root = "/home/anoch/Documents/BatSamples/SpectrogramMarked/"
        path  = root + event + "/Event" + eventno + ".png"
        eventImage = QtGui.QPixmap(path)
        scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio)
        self.ui.label_imageshow.setPixmap(scaledEventImage)

    def updateEventInfomation(self):

        data = self.HDFFile[str(self.pathcorr[self.currentEvent])]
        self.setEventImage(self.file[self.currentEvent],self.eventno[self.currentEvent])
        dateToShow = self.day[self.currentEvent] + "-" + self.month[self.currentEvent] + "-" + self.year[self.currentEvent]
        self.ui.label_Date.setText(dateToShow)
        self.ui.label_EventName.setText(self.file[self.currentEvent])
        self.ui.label_EventNo.setText(self.eventno[self.currentEvent])


        duration = str(toTime(abs(data[2]-data[3])))
        self.ui.label_MinFreq.setText(str(tokFreq(data[0])))
        self.ui.label_maxFreq.setText(str(tokFreq(data[1])))
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



    def scanForNextEvent(self):
        for i in range(self.currentEvent, self.EventSize):
            data = self.HDFFile[str(self.pathcorr[i])]
            if data.attrs['BatID'] == 0:
                self.currentEvent = i
                self.ui.progressBar_eventLabel.setValue(self.currentEvent)
                break


    def file_dialog(self):
        self.filepath = QtGui.QFileDialog.getOpenFileName(self, "Open HDF5 File",'', "HDF5 Files (*.hdf5 *.h5)")

        from os.path import isfile
        if isfile(self.filepath):

            self.HDFFile = h5py.File(str(self.filepath))
            self.HDFFile.visit(self.saveEventPath)
            self.day, self.month, self.year, self.file, self.eventno, self.pathcorr = getHDFInformation(self.pathEventList)
            self.EventSize = len(self.day)
            self.currentEvent = 0
            self.ui.progressBar_eventLabel.setMinimum(self.currentEvent)
            self.ui.progressBar_eventLabel.setMaximum(self.EventSize)
            filename = os.path.basename(str(self.filepath))
            self.ui.label_database_name.setText(filename)
            self.ui.frame_BatButtons.show()
            self.scanForNextEvent()
            self.updateEventInfomation()
        else:
            self.ui.label_database_name.setText("None selected")



    def run_analyser(self):
        rootpath = "/home/anoch/Documents/BatSamples/"

        SearchPath = rootpath + "Spectrogram/"
        SavePath = rootpath + "SpectrogramMarked/"
        sampleList = getFunctions.getFileList(SearchPath,".png")
        maxSize = len(sampleList)
        self.ui.progressBar_analyse.setMinimum(0)
        self.ui.progressBar_analyse.setMaximum(maxSize)
        progressCount = 0

        for eventFile in sampleList:
            #print "Analyzing " + os.path.splitext((eventFile))[0]
            #self.ui.textEdit_overview.setText("Analyzing " + os.path.splitext((eventFile))[0] + "\n")
            #topX to endX is the time range, while topX and bottomY is the frequency range
            EventExtraction.findEvent(SearchPath, eventFile, SavePath)
            progressCount += 1
            self.ui.progressBar_analyse.setValue(progressCount)

        #self.ui.textEdit_overview.setText("Event extraction done!")
        #print "Event extraction done!"


if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	myapp = StartQT4()
	myapp.show()
	sys.exit(app.exec_())