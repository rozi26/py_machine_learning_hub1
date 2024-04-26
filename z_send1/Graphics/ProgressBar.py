from datetime import datetime, timedelta
class ProgressBar():
    
    def __init__(self, items_to_load):
        self.length = items_to_load
        self.item = 1
        self.longest = 0
        self.startTime = datetime.now()
    def progress(self):
        num = "{:,}".format(self.item)
        end = "{:,}".format(self.length)
        text = "progress: " + str(num) + "\\" + str(end) + " (" + getPresent(self.item,self.length) + ")"
        
        seconds = round((datetime.now() - self.startTime).total_seconds())
        timeLeft = astimateSecandsLeft(self.item,self.length,seconds)
        text += " [" + str(timedelta(seconds=seconds)) + "] ~A[" + str(timedelta(seconds=timeLeft)) + "]"
        
        if(len(text) > self.longest):
            self.longest = len(text)
        else:
            text += " "*(self.longest - len(text))
        print(f"\r{text}", end="")
        self.item += 1
def getPresent(a,b):
    return str(round((a * 100) / (b),2)) + "%"
def astimateSecandsLeft(item, length, time):
    if(time == 0): return 0
    return round((length - item) * (time / item))
    