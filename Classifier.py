__author__ = 'Anochjhn Iruthayam'

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
