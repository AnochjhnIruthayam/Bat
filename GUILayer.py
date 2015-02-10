__author__ = 'Anochjhn Iruthayam'


import Tkinter as tk
import Image, ImageTk
import xml.etree.ElementTree as ET # phone  home

def event_non_bat(image, event_dir, eventNo, top):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"

    print "NON BAT CALL Registered and written to XML file: " + labelPath
    print "Image: " + image

    tree = ET.parse(labelPath)
    root = tree.getroot()
    lookup = "Event"+str(eventNo)

    #Look for BatID tag
    for BatID in tree.iter(lookup+'/batID'):
        #Mark as NONBAT
        BatID.text = str(0)
    for elem in tree.find("Event/"+lookup):
        if elem.tag == "batID":
            elem.text = str(0)
        print "\t"+elem.tag, elem.text

    tree.write(labelPath)
    top.destroy()

def event_bat(image, event_dir, eventNo, top):
    rootpath = "/home/anoch/Documents/BatSamples/"
    labelPath =  rootpath + "SpectrogramMarked/" + event_dir + "/label.xml"
    print "BAT CALL Registered and written to XML file: " + labelPath
    print "Image: " + image

    #Read XML file
    tree = ET.parse(labelPath)
    root = tree.getroot()
    lookup = "Event"+str(eventNo)

    for elem in tree.find("Event/"+lookup):
        if elem.tag == "batID":
            elem.text = str(1)
        print "\t"+elem.tag, elem.text

    tree.write(labelPath)
    top.destroy()
    #root.findall(".")
    #print tree.findall("./Event/Event"+str(eventNo)).find("minFreq").text

def keyLeft(event):
    print "ARROW BAT CALL Registered and written to XML file: " + event

def keyRight(event):
    print "ARROW NON BAT CALL Registered and written to XML file: " + event



def GUIClassifier(rootpath, image, event_dir, eventNo):

    top = tk.Tk()
    top.minsize(width=300, height=300)
    image = rootpath + "SpectrogramMarked/" + event_dir +"/"+  image
    print image
    #IMAGE
    im = Image.open(image)
    tkimage = ImageTk.PhotoImage(im)
    tk.Label(top, image=tkimage).pack()
    i = 0
    #BUTTONS
    btn_bat = tk.Button(top, text ="BAT", command = lambda: event_bat(image, event_dir, eventNo, top))
    btn_bat.pack(side=tk.TOP)

    nonbat_btn = tk.Button(top, text ="NONBAT", command = lambda: event_non_bat(image, event_dir, eventNo, top))
    nonbat_btn.pack(side=tk.TOP)
    #top.bind('<Left>',keyLeft(event_bat(image, event_dir, eventNo, top)))
    #top.bind('<Right>', keyRight(event_non_bat(image, event_dir, eventNo, top)))
    #top.focus_set()
    top.mainloop()


def GUI(rootpath):
    all_events, event_dir = get_all_bat_event(rootpath)

    for i in range(0, len(all_events)):
        print str(i) +" out of " + str(len(all_events))
        eventNo= ''.join(x for x in all_events[i] if x.isdigit())
        GUIClassifier(rootpath, all_events[i],event_dir[i], eventNo)