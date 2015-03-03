__author__ = 'Anochjhn Iruthayam'


import sys
from PyQt4 import QtCore, QtGui
from BatWindow import Ui_BatWindow
import EventExtraction, os, getFunctions, time



class StartQT4(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self,parent)
        self.ui = Ui_BatWindow()
        self.ui.setupUi(self)
        self.ui.progressBar.setValue(0)
        QtCore.QObject.connect(self.ui.button_start,QtCore.SIGNAL("clicked()"), self.run_analyser)
        #QtCore.QObject.connect(self.ui.progressBar)
    def run_analyser(self):
        rootpath = "/home/anoch/Documents/BatSamples/"

        SearchPath = rootpath + "Spectrogram/"
        SavePath = rootpath + "SpectrogramMarked/"
        sampleList = getFunctions.getFileList(SearchPath,".png")
        maxSize = len(sampleList)
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(maxSize)
        progressCount = 0
        for eventFile in sampleList:
            #print "Analyzing " + os.path.splitext((eventFile))[0]
            self.ui.textEdit_overview.setText("Analyzing " + os.path.splitext((eventFile))[0] + "\n")
            #topX to endX is the time range, while topX and bottomY is the frequency range
            EventExtraction.findEvent(SearchPath, eventFile, SavePath)
            progressCount += 1
            self.ui.progressBar.setValue(progressCount)

        self.ui.textEdit_overview.setText("Event extraction done!")
        #print "Event extraction done!"

if __name__ == "__main__":
	app = QtGui.QApplication(sys.argv)
	myapp = StartQT4()
	myapp.show()
	sys.exit(app.exec_())