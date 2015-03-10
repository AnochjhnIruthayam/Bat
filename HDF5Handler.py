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


def imageRecontructFromHDF5(ArrayDataImg):
    image_2d = np.vstack(itertools.imap(np.uint8, ArrayDataImg))
    height, width = image_2d.shape
    image_3d = np.reshape(image_2d, (height, width, 1))

    return image_3d


def eventHDFLabel(eventName, minMiliSec_pixel, maxFreq_pixel, maxMiliSec_pixel, minFreq_pixel, savePath, eventNum,
                  OutputDirectory, imgSpectrogram, imgMarkedSpectrogram, imgEvent):
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
        e = DirectoryString in f

        if not e:
            out = f.create_group(DirectoryString)
        else:
            out = f[DirectoryString]

        ArrayImgEvent = imageToHDF5_img(imgEvent)
        ds_Event = out.create_dataset("ArrayImgEvent_" + str(i), ArrayImgEvent.shape, compression="gzip")
        ds_Event[:] = ArrayImgEvent

        dbName = "FeatureDataEvent_" + str(i)
        ds_FeatureDataEvent = out.create_dataset(dbName, data.shape, compression="gzip")
        ds_FeatureDataEvent[:] = data
        out[dbName].attrs["Day"] = int(day)
        out[dbName].attrs["Month"] = int(month)
        out[dbName].attrs["Year"] = int(year)
        out[dbName].attrs["Hour"] = int(hour)
        out[dbName].attrs["Minute"] = int(min)
        out[dbName].attrs["Second"] = int(sec)
        out[dbName].attrs["Offset"] = int(offset)
        out[dbName].attrs["Recording Channel"] = int(channel)
        out[dbName].attrs["Sample Rate"] = 500000
        out[dbName].attrs["Temperature"] = temperature
        out[dbName].attrs["Humidity"] = humidity
        out[dbName].attrs["Wind Speed"] = windspeed
        out[dbName].attrs["Weather Condition"] = weathercondition
        out[dbName].attrs["BatID"] = bat_id
    f.close()