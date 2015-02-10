__author__ = 'Anochjhn Iruthayam'
import BatMain
import Classifier as c
def saveData():
    print "Saving to file.."
    rootpath = "/home/anoch/Documents/BatSamples/"
    event, list_event_dir = BatMain.get_all_bat_event(rootpath)
    text_file = open("/home/anoch/Documents/Output.txt", "w")
    for i in range(0, len(list_event_dir)):
        eventNo= ''.join(x for x in event[i] if x.isdigit())
        minFreq, maxFreq, MiliSec, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, t_1, t_2, t_3, t_4, t_5, t_6, t_7, t_8, t_9, t_10 = c.ANN_input(list_event_dir[i],eventNo)
        target = c.ANN_outout(list_event_dir[i],eventNo)
        #data = str(minFreq) +"," + str(maxFreq)+"," + str(MiliSec)+"," + str(pixels)+"," +str(target) + "\n"
        #text_file.write(data)
    text_file.close()
    print "Save Done!"