# Import required modules
import argparse
import time
import os
import math
from shutil import copyfile


############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(description="Analysis face data")
parser.add_argument(
    "--dataset",
    type=str,
    default="./database/",
    help="path to input directory of faces + images",
)
args = parser.parse_args()

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def list_dirs(basePath):
    output = []
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        output.extend(dirNames)
        break
    return output


def list_images_dirs(basePath, contains=None):
    # return the set of files that are valid
    return list_file_dirs(basePath, validExts=image_types, contains=contains)


def list_file_dirs(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the folder name of the image and yield it
                yield rootDir[rootDir.rfind("/") + 1 :].lower()


def list_images_paths(basePath, contains=None):
    # return the set of files that are valid
    return list_file_paths(basePath, validExts=image_types, contains=contains)


def list_file_paths(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the folder name of the image and yield it
                yield rootDir


def transTime(tick, string):
    if tick >= 60000:
        mins = math.floor(tick / 60000)
        secs = math.floor((tick - mins * 60000) / 1000)
        msec = tick - mins * 60000 - secs * 1000
        print(
            string
            + str(mins)
            + " mins "
            + str(secs)
            + " secs "
            + str(math.floor(msec))
            + " ms"
        )
    elif tick >= 1000:
        secs = math.floor(tick / 1000)
        msec = tick - secs * 1000
        print(string + str(secs) + " secs " + str(math.floor(msec)) + " ms")
    else:
        print(string + str(math.floor(tick)) + " ms")


def main():
    # get start time
    start_time = time.time()

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    images = list(list_images(args.dataset))
    imagePaths = list(list_images_paths(args.dataset))

    # initialize our lists of extracted facial embeddings and
    # corresponding people names
    knownNames = list(list_images_dirs(args.dataset))

    # initialize the total number of faces processed
    total = 0

    # initialize the number of faces of a people
    count = 0
    count_temp = ""

    # start face data manipulating
    for (i, image) in enumerate(images):
        if count_temp in imagePaths[i]:
            count += 1
        else:
            count = 1

        newName = imagePaths[i] + "/" + knownNames[i] + "-" + str(count) + ".jpg"
        os.rename(image, newName)

        newPath = imagePaths[i] + "-" + str(count) + ".jpg"
        copyfile(newName, newPath)

        count_temp = imagePaths[i]
        total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} images...".format(total))

    # calculate processing time
    tick = (time.time() - start_time) * 1000
    transTime(tick, "[INFO] Total process time: ")


if __name__ == "__main__":
    main()
