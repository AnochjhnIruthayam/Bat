__author__ = 'Anochjhn Iruthayam'


import sys
from PyQt4 import QtCore, QtGui
from BatWindow import Ui_BatWindow
import EventExtraction, os, getFunctions, time

class ExtractThread(QtCore.QThread):
    def __init__(self, function):
        QtCore.QThread.__init__(self)
        self.function = function

    def __del__(self):
        self.wait()

    def run(self):
        self.function()

        return

class StartQT4(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_BatWindow()
        self.ui.setupUi(self)
        self.ui.progressBar.setValue(0)
        self.threadPool = []
        QtCore.QObject.connect(self.ui.button_start,QtCore.SIGNAL("clicked()"), self.testrun)
        QtCore.QObject.connect(self.ui.button_loaddatabase,QtCore.SIGNAL("clicked()"), self.file_dialog)
        #QtCore.QObject.connect(self.ui.progressBar)
    def file_dialog(self):
        self.filepath = QtGui.QFileDialog.getOpenFileName(self, "Open HDF5 File",'', "HDF5 Files (*.hdf5 *.h5)")

        eventImage = QtGui.QPixmap("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000007665339500/Event4.png")
        scaledEventImage = eventImage.scaled(self.ui.label_imageshow.size(), QtCore.Qt.KeepAspectRatio )
        self.ui.label_imageshow.setPixmap(scaledEventImage)


        from os.path import isfile
        if isfile(self.filepath):
            filename = os.path.basename(str(self.filepath))
            self.ui.label_database_name.setText(filename)


    def run_analyser(self):
        rootpath = "/home/anoch/Documents/BatSamples/"

        SearchPath = rootpath + "Spectrogram/"
        SavePath = rootpath + "SpectrogramMarked/"
        sampleList = getFunctions.getFileList(SearchPath,".png")
        maxSize = len(sampleList)
        #self.ui.progressBar.setMinimum(0)
        #self.ui.progressBar.setMaximum(maxSize)
        progressCount = 0

        for eventFile in sampleList:
            #print "Analyzing " + os.path.splitext((eventFile))[0]
            #self.ui.textEdit_overview.setText("Analyzing " + os.path.splitext((eventFile))[0] + "\n")
            #topX to endX is the time range, while topX and bottomY is the frequency range
            EventExtraction.findEvent(SearchPath, eventFile, SavePath)
            progressCount += 1
            #self.ui.progressBar.setValue(progressCount)

        #self.ui.textEdit_overview.setText("Event extraction done!")
        #print "Event extraction done!"

    def testrun(self):
        createthread = ExtractThread(self.run_analyser)
        createthread.start()
        #self.threadPool.append( ExtractThread(self.run_analyser))
        #self.threadPool[len(self.threadPool)-1].start()


if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	myapp = StartQT4()
	myapp.show()
	sys.exit(app.exec_())