from os.path import exists
from os import mkdir

def getAvailableFolder(path):
    num = 1
    while(exists(path + "_" + str(num))):
        num += 1
    loc = path + "_" + str(num)
    createDir(loc)
    return loc

def folderExist(path):
    return exists(path)
def createDir(path):
    mkdir(path)
    return path