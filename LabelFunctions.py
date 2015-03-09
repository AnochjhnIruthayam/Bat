__author__ = 'Anochjhn Iruthayam'
import xml.etree.ElementTree as ET # phone  home
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import numpy as np
import EventExtraction as ee
import h5py, re



def eventHDFLabel(eventName, minMiliSec_pixel, maxFreq_pixel, maxMiliSec_pixel, minFreq_pixel, savePath, eventNum, OutputDirectory, InputDirectory):
    SAVE_ITERATION = 10
    myHDFFile = OutputDirectory + "/BatData.hdf5"
    f = h5py.File(myHDFFile)
    for i in eventNum:
        # save to disk for every 10th event
        tempSave = i % SAVE_ITERATION
        if tempSave == 0:
            f.flush()
        currentSubject = savePath + eventName + "/Event" + str(i) + ".png"
        points = ee.getFrontLineFeature(currentSubject)
        #toHDF(bottomY[i],topY[i],topX[i],endX[i],points,eventName,i)

        temp = re.split('_', eventName)
        day = temp[1]
        month = temp[2]
        year = temp[3]
        hour = temp[5]
        min = temp[6]
        sec = temp[7]
        channel = temp[9]
        offset = temp[11]


        #soundfile = InputDirectory + "/" + eventName + ".s16"

        #print "Creating Database"

        #Meta data for weather
        temperature = 0
        humidity = 0
        windspeed = 0
        weathercondition = 0
        bat_id = 0


        #testdata = np.random.random(14)

        #Process which zero pads feature data, if there is not enough data
        test_zero = 0
        zero_int = 0
        for point in points:
            test_zero += point
            zero_int += 1
        if test_zero <= 0 or zero_int < 11:
            points = np.zeros(11)

        #Feature data
        data = np.array([minFreq_pixel[i], maxFreq_pixel[i],minMiliSec_pixel[i], maxMiliSec_pixel[i],points[0],points[1],points[2],points[3],points[4],points[5],points[6],points[7],points[8],points[9],points[10]])
        #print data.shape


        #currentfile = soundfile
        #day,month,year,hour,min,sec = ee.get_time_for_modified_files(currentfile)
        timeName = hour + ":" + min + ":" + sec
        DirectoryString = year + "/" + month + "/" + day  + "/" + timeName + "_" + offset
        e = DirectoryString in f
        print DirectoryString
        print "\n\n"
        if not e:
            out = f.create_group(DirectoryString)
        else:
            out = f[DirectoryString]
        dbName = "FeatureDataEvent_" + str(i)
        out[dbName] = data
        out[dbName].attrs["Day"] = int(day)
        out[dbName].attrs["Month"] = int(month)
        out[dbName].attrs["Year"] = int(year)
        out[dbName].attrs["Hour"] = int(hour)
        out[dbName].attrs["Minute"] = int(min)
        out[dbName].attrs["Second"] = int(sec)
        out[dbName].attrs["Offset"] = int(offset)
        out[dbName].attrs["Recording Channel"] = int(channel)
        out[dbName].attrs["Temperature"] = temperature
        out[dbName].attrs["Humidity"] = humidity
        out[dbName].attrs["Wind Speed"] = windspeed
        out[dbName].attrs["Weather Condition"] = weathercondition
        out[dbName].attrs["BatID"] = bat_id
    f.close()



def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
#Label Event

def eventLabel2OLD(eventName, topX, topY, endX, bottomY, event_img_path, savePath,imgHeight,imgLength, eventNum):
    import EventExtraction as ee
    top = Element('top')
    comment = Comment("Event information for: " + eventName)
    top.append(comment)

    event = SubElement(top, 'Event')
    for i in eventNum:
        currentSubject = savePath + eventName + "/Event" + str(i) + ".png"
        points = ee.getFrontLineFeature(currentSubject)

        subEvent = SubElement(event, "Event" + str(i))

        ee.hdfgroup(bottomY[i],topY[i],topX[i],endX[i],points,eventName,i)


        minFreq = SubElement(subEvent,"minFreq_pixel")
        minFreq.text = str(bottomY[i])

        maxFreq = SubElement(subEvent,"maxFreq_pixel")
        maxFreq.text = str(topY[i])

        minMiliSec = SubElement(subEvent,"minMiliSec_pixel")
        minMiliSec.text = str(topX[i])

        maxMiliSec = SubElement(subEvent,"maxMiliSec_pixel")
        maxMiliSec.text = str(endX[i])
        it = 0
        for point in points:
            big = SubElement(subEvent,"big" +str(it)+"_pixel")
            big.text = str(point)
            it += 1


        batID = SubElement(subEvent, "batID")
        batID.text = "NOT CLASSIFIED"

        MainEventSoundFile = SubElement(subEvent,"MainEventSoundFile")
        MainEventSoundFile.text = eventName


    #print prettify(top)
    tree = ET.ElementTree(top)
    tree.write(savePath + eventName + "/label.xml")