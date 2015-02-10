__author__ = 'Anochjhn Iruthayam'



import cv2
import os
import EventExtraction as ee
import getFunctions
import LabelFunctions

# import MultiNEAT as NEAT #not as neat I thought it would be! BASTARD!! Still remember ET to phone home
# from pybrain.tools.shortcuts import buildNetwork

# Set up global frequency band. Set to the range of BatClassification Calls aka. 13 Khz to 75 KHz into Pixel values
#getHeightMin = 500
#getHeightMax = 980



#Scan the spectrogram with sliding window.
#def scanHorizontal(img)
#    getHeight, getWidth  = img.shape












#####################################################MAIN###############################################################

def main():
    rootpath = "/home/anoch/Documents/BatSamples/"

    #createSpectrogram(rootpath)
    getFunctions.getAllEvents(rootpath)
    #img = cv2.imread("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000029921020500SpectrogramAllMarked.png",0)
    #verticalScan2(img)
    #GUI(rootpath)
    #ANN_SupervisedBackPro()
    #ANN_Classifier()
    #saveData()
    ######################################FOR THE NEW FEATURE EXTRACTION -- WORKS#########################################
    #poly =  getFrontLineFeature("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000029923979000/Event0.png")
    #print poly, len(poly)
    #poly =  getFrontLineFeature("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000007842559000/Event1.png")
    #print poly, len(poly)
    #bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000041706671000/Event1.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000018916088000/Event5.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event0.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event3.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event4.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event8.png")
    # bestFit("/home/anoch/Documents/BatSamples/SpectrogramMarked/sr_500000_ch_4_offset_00000000008460585000/Event9.png")

#run main
main()