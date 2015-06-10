__author__ = 'Anochjhn Iruthayam'
import os

###########################USED BY GUI#############
def getFileList(path, extension):
    sampleList = []
    for file in os.listdir(path):
        if file.endswith(extension):
            sampleList.append(file)
    return sampleList

def getFileListDepthScan(path, extension):
    sampleList = []
    for root, dirs, files in os.walk(path):
        if len(files) > 0:
            for file in files:
                if file.endswith(extension):
                    currentPath = root +"/" + file
                    sampleList.append(currentPath)
    return sampleList
###################################3
















def getAllEvents(rootpath):
    import EventExtraction
    SearchPath = rootpath + "/Spectrogram/"
    SavePath = rootpath + "/SpectrogramMarked/"
    sampleList = getFileList(SearchPath,".png")
    for eventFile in sampleList:
        print "Analyzing " + os.path.splitext((eventFile))[0]
        #topX to endX is the time range, while topX and bottomY is the frequency range
        EventExtraction.findEvent(SearchPath, eventFile, SavePath)
    print "Event extraction done!"

def get_all_bat_event(rootpath):

    path = rootpath + "SpectrogramMarked/"
    listdirectoryTEMP =  os.listdir(path)
    #eventlist = getFileList(path, ".png")
    listdirectory = []
    list_event = []
    list_event_dir = []
    temppath = ""
    for dir in listdirectoryTEMP:
        if not ".png" in dir:
            if not "~" in dir:
                listdirectory.append(dir)
                #print dir

    for dir in listdirectory:
        eventlist = getFileList(path+dir, ".png")
        for current_event in eventlist:
            list_event.append(current_event)
            list_event_dir.append(dir)
    #print list_event
    #print list_event_dir
    return list_event, list_event_dir

