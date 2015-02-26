__author__ = 'Anochjhn Iruthayam'
import xml.etree.ElementTree as ET # phone  home
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, Comment, tostring


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
#Label Event
def eventLabel(eventName, topX, topY, endX, bottomY, event_img_path, savePath,imgHeight,imgLength, eventNum):
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
        # big0 = SubElement(subEvent,"big0_pixel")
        # big0.text = str(points[0])
        #
        # big1 = SubElement(subEvent,"big1_pixel")
        # big1.text = str(points[1])
        #
        # big2 = SubElement(subEvent,"big2_pixel")
        # big2.text = str(points[2])
        #
        # big3 = SubElement(subEvent,"big3_pixel")
        # big3.text = str(points[3])
        #
        # big4 = SubElement(subEvent,"big4_pixel")
        # big4.text = str(points[4])
        #
        # big5 = SubElement(subEvent,"big5_pixel")
        # big5.text = str(points[5])
        #
        # big6 = SubElement(subEvent,"big6_pixel")
        # big6.text = str(points[6])
        #
        # big7 = SubElement(subEvent,"big7_pixel")
        # big7.text = str(points[7])
        #
        # big8 = SubElement(subEvent,"big8_pixel")
        # big8.text = str(points[8])
        #
        # big9 = SubElement(subEvent,"big9_pixel")
        # big9.text = str(points[9])
        #
        # big10 = SubElement(subEvent,"big10_pixel")
        # big10.text = str(points[10])

        batID = SubElement(subEvent, "batID")
        batID.text = "NOT CLASSIFIED"

        MainEventSoundFile = SubElement(subEvent,"MainEventSoundFile")
        MainEventSoundFile.text = eventName


    #print prettify(top)
    tree = ET.ElementTree(top)
    tree.write(savePath + eventName + "/label.xml")