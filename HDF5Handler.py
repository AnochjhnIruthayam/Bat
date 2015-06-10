__author__ = 'Anochjhn Iruthayam'
import h5py
import itertools
import re
import numpy as np

import cv2

import EventExtraction as ee


def imageToHDF5_path(path):
    readIMG = cv2.imread(path, 0)
    return np.vstack(itertools.imap(np.uint8, readIMG))


def imageToHDF5_img(readImg):
    return np.vstack(itertools.imap(np.uint8, readImg))

def loadSoundFile(soundFile):
    f = open(soundFile, "r")
    soundArray = np.fromfile(f, dtype=np.int16)
    return soundArray


def imageRecontructFromHDF5(ArrayDataImg):
    image_2d = np.vstack(itertools.imap(np.uint8, ArrayDataImg))
    height, width = image_2d.shape
    image_3d = np.reshape(image_2d, (height, width, 1))

    return image_3d

def eventHDFLabel(eventName, minMiliSec_pixel, maxFreq_pixel, maxMiliSec_pixel, minFreq_pixel, savePath, eventNum, OutputDirectory, imgSpectrogram, imgMarkedSpectrogram, imgEvent, recordedAt, projectName, InputDirectory):
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
        # toHDF(bottomY[i],topY[i],topX[i],endX[i],points,eventName,i)

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
        call_type = 0


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
        data = np.array(
            [minFreq_pixel[i], maxFreq_pixel[i], minMiliSec_pixel[i], maxMiliSec_pixel[i], points[0], points[1],
             points[2], points[3], points[4], points[5], points[6], points[7], points[8], points[9], points[10]])
        #print data.shape


        #currentfile = soundfile
        #day,month,year,hour,min,sec = ee.get_time_for_modified_files(currentfile)
        timeName = hour + ":" + min + ":" + sec
        DirectoryString = year + "/" + month + "/" + day + "/" + timeName + "_" + offset
        DirectoryString = projectName + "/" + recordedAt + "/" + year + "/" + month + "/" + day + "/" + timeName + "_" + offset

        e = DirectoryString in f

        if not e:
            out = f.create_group(DirectoryString)

            # Gets the pixels array and saves it to HDF5
            ArrayImgSpectrogram = imageToHDF5_img(imgSpectrogram)
            # Create dataset with ArrayImgSpectrogram specifications
            ds_Spectrogram = out.create_dataset("ArrayImgSpectrogram", ArrayImgSpectrogram.shape, dtype=ArrayImgSpectrogram.dtype, compression="gzip")
            # Save data to dataset
            ds_Spectrogram[:] = ArrayImgSpectrogram
            out["ArrayImgSpectrogram"].attrs["Image Format"] = ".PNG"
            out["ArrayImgSpectrogram"].attrs["Bit Depth"] = 8
            out["ArrayImgSpectrogram"].attrs["Interlace"] = 0
            out["ArrayImgSpectrogram"].attrs["Grayscale"] = "TRUE"
            out["ArrayImgSpectrogram"].attrs["Alpha"] = "FALSE"

            # Gets the pixels array and saves it to HDF5
            ArrayImgMarkedSpectrogram = imageToHDF5_img(imgMarkedSpectrogram)
            ds_MarkedSpectrogram = out.create_dataset("ArrayImgMarkedSpectrogram", ArrayImgMarkedSpectrogram.shape, dtype=ArrayImgMarkedSpectrogram.dtype , compression="gzip")
            ds_MarkedSpectrogram[:] = ArrayImgMarkedSpectrogram
            out["ArrayImgMarkedSpectrogram"].attrs["Image Format"] = ".PNG"
            out["ArrayImgMarkedSpectrogram"].attrs["Bit Depth"] = 8
            out["ArrayImgMarkedSpectrogram"].attrs["Interlace"] = 0
            out["ArrayImgMarkedSpectrogram"].attrs["Grayscale"] = "TRUE"
            out["ArrayImgMarkedSpectrogram"].attrs["Alpha"] = "FALSE"

            # Get .s16 file and save it to dataset
            #Reconstruct orginial .s16 filename
            filename = InputDirectory + "/sr_500000_ch_4_offset_" + offset + ".s16"
            #Convert binary file to array
            ArraySound = loadSoundFile(filename)
            ds_sound = out.create_dataset("SensorData", ArraySound.shape, dtype=ArraySound.dtype, compression="gzip")
            ds_sound[:] = ArraySound
            out["SensorData"].attrs["Sampling Frequency"] = 500000
            out["SensorData"].attrs["Number of Channels"] = 4


        SubDirectory = DirectoryString + "/Event_" + str(i)
        e2 = SubDirectory in f
        if not e2:
            EventGroup = f.create_group(SubDirectory)

            # Gets the pixels array and saves it to HDF5
            ArrayImgEvent = imageToHDF5_img(imgEvent)
            ds_Event = EventGroup.create_dataset("ArrayImgEvent", ArrayImgEvent.shape, dtype=ArrayImgEvent.dtype , compression="gzip")
            ds_Event[:] = ArrayImgEvent
            EventGroup["ArrayImgEvent"].attrs["Image Format"] = ".PNG"
            EventGroup["ArrayImgEvent"].attrs["Bit Depth"] = 8
            EventGroup["ArrayImgEvent"].attrs["Interlace"] = 0
            EventGroup["ArrayImgEvent"].attrs["Grayscale"] = "TRUE"
            EventGroup["ArrayImgEvent"].attrs["Alpha"] = "FALSE"


            dbName = "FeatureDataEvent"
            ds_FeatureDataEvent = EventGroup.create_dataset(dbName, data.shape, dtype=data.dtype, compression="gzip")
            ds_FeatureDataEvent[:] = data
            EventGroup[dbName].attrs["Day"] = int(day)
            EventGroup[dbName].attrs["Month"] = int(month)
            EventGroup[dbName].attrs["Year"] = int(year)
            EventGroup[dbName].attrs["Hour"] = int(hour)
            EventGroup[dbName].attrs["Minute"] = int(min)
            EventGroup[dbName].attrs["Second"] = int(sec)
            tempOffset = re.split('-', offset)
            EventGroup[dbName].attrs["Offset"] = int(tempOffset[0])
            #if len(tempOffset) > 1:
            #    EventGroup[dbName].attrs["Offset Number"] = int(offset[1])
            #else:
            #    EventGroup[dbName].attrs["Offset Number"] = 0
            EventGroup[dbName].attrs["Recording Channel"] = int(channel)
            EventGroup[dbName].attrs["Sample Rate"] = 500000
            EventGroup[dbName].attrs["Temperature"] = temperature
            EventGroup[dbName].attrs["Humidity"] = humidity
            EventGroup[dbName].attrs["Wind Speed"] = windspeed
            EventGroup[dbName].attrs["Weather Condition"] = weathercondition
            EventGroup[dbName].attrs["BatID"] = bat_id
            EventGroup[dbName].attrs["Call Type"] = call_type
            EventGroup[dbName].attrs["Recorded At"] = recordedAt

            #Data already existing, then overwrite
            """
            ArrayImgSpectrogram = imageToHDF5_img(imgSpectrogram)
            f[DirectoryString + "/ArrayImgSpectrogram"] = ArrayImgSpectrogram
            f[DirectoryString + "/ArrayImgSpectrogram"].attrs["Image Format"] = ".PNG"
            f[DirectoryString + "/ArrayImgSpectrogram"].attrs["Bit Depth"] = 8
            f[DirectoryString + "/ArrayImgSpectrogram"].attrs["Interlace"] = 0
            f[DirectoryString + "/ArrayImgSpectrogram"].attrs["Grayscale"] = "TRUE"
            f[DirectoryString + "/ArrayImgSpectrogram"].attrs["Alpha"] = "FALSE"

            ArrayImgMarkedSpectrogram = imageToHDF5_img(imgMarkedSpectrogram)
            f[DirectoryString + "/ArrayImgMarkedSpectrogram"] = ArrayImgMarkedSpectrogram
            f[DirectoryString + "/ArrayImgMarkedSpectrogram"].attrs["Image Format"] = ".PNG"
            f[DirectoryString + "/ArrayImgMarkedSpectrogram"].attrs["Bit Depth"] = 8
            f[DirectoryString + "/ArrayImgMarkedSpectrogram"].attrs["Interlace"] = 0
            f[DirectoryString + "/ArrayImgMarkedSpectrogram"].attrs["Grayscale"] = "TRUE"
            f[DirectoryString + "/ArrayImgMarkedSpectrogram"].attrs["Alpha"] = "FALSE"
            """

    f.close()