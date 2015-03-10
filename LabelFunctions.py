__author__ = 'Anochjhn Iruthayam'
import xml.etree.ElementTree as ET # phone  home
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment
import numpy as np
import h5py
import re

import EventExtraction as ee



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