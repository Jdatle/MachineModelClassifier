# John Le's Project 1 (ASU CSE 572 FALL A 2022)

# File Incoming Files

fileCGM = "CGMData.csv"
fileInsulin = "InsulinData.csv"

# File Outgoing

fileResult = "Results.csv"

# FUNCTION DEFINITIONS

# Determine When Manual turns off and Auto Starts

def ManOffAutoStart(filepath):
    infile = open(filepath)
    message = "AUTO MODE ACTIVE PLGM OFF"
    counter = 0

    for line in infile:

        if line.find(message) == -1:
            counter = counter
        else:
            if counter == 1:
                counter = counter
            else:
                counter = counter + 1
                string = line.split(",")
                autoStartTime = [string[1], string[2]]
    return autoStartTime

# Parse single CGM line

def ParseLine(string):
    string = string.split(",")
    parseLine = [string[1], string[2], string[30]]
    return parseLine

# Create Big List of imported Data

def fileToList(filepath):
    infile = open(filepath)
    list = []
    next(infile)
    for line in infile:
        list.append(ParseLine(line))
    infile.close()
    return list

# list of dates with 288 entries per day AND zero blank data

def listOfDatesFullData(list):
    listAllDates = []
    dataPointsPerData = 0
    currentDate = list[0][0]

    for i in list:
        if i[0] == currentDate:
            dataPointsPerData = dataPointsPerData + 1
        else:
            listAllDates.append([currentDate, dataPointsPerData])
            currentDate = i[0]
            dataPointsPerData = 1
        if i == list[-1]:
            listAllDates.append([currentDate, dataPointsPerData])

    listDates288 = []
    for i in listAllDates:
        if i[1] == 288:
            listDates288.append(i[0])

    listDatesToRemove = []
    for i in listDates288:
        for j in list:
            if i == j[0]:
                if j[2] == '' or j[2] == 'NaN' :
                    listDatesToRemove.append(i)
    listDatesToRemove = [*set(listDatesToRemove)]

    listDatesFullData = []
    for i in listDates288:
        if i in listDatesToRemove:
            listDatesFullData = listDatesFullData
        else:
            listDatesFullData.append(i)
    return listDatesFullData

# list of dates manual & Auto

def listOfDatesManualAuto(list, date):
    listManualDates = []
    listAutoDates = []
    day = date[0]
    for i in list:
        if i == NewerDay(i, day):
            listManualDates.append(i)
        else:
            listAutoDates.append(i)
    return listManualDates, listAutoDates

# Function to determine which is the newest day

def NewerDay(date1, date2):
    date1Total = date1.split("/")
    date2Total = date2.split("/")
    year1 = int(date1Total[2])
    month1 = int(date1Total[0])
    day1 = int(date1Total[1])
    year2 = int(date2Total[2])
    month2 = int(date2Total[0])
    day2 = int(date2Total[1])

    if year1 > year2:
        return date1
    elif year1 < year2:
        return date2
    elif month1 > month2:
        return date1
    elif month1 < month2:
        return date2
    elif day1 > day2:
        return date1
    elif day1 < day2:
        return date2
    else:
        return date1

# Function to determine CGM Range (0-5)

def CGM_Range(cgm):
    if cgm > 250:
        return 1
    elif cgm > 180:
        return 0
    elif cgm >= 70 and cgm <= 150:
        return 3
    elif cgm >= 70 and cgm <=180:
        return 2
    elif cgm < 54:
        return 5
    elif cgm < 70:
        return 4

# Function to Calculate percentages for a given day

def PercentageCalc(day, list):
    metricCounter = [ [0,0,0,0,0,0], [0,0,0,0,0,0],[0,0,0,0,0,0] ]
    for i in list:
        if i[0] == day:
            cgm = int(i[2])
            timeInterval = TimeInterval(i[1])
            cgmRange = CGM_Range(cgm)
            metricCounter[2][cgmRange] += 1
            metricCounter[timeInterval][cgmRange] += 1
    metricPercentage = [[],[],[]]
    # metricPercentage[0] = [x /72 for x in metricCounter[0]]
    # metricPercentage[1] = [x /216 for x in metricCounter[1]]
    metricPercentage[0] = [x /288 for x in metricCounter[0]]
    metricPercentage[1] = [x /288 for x in metricCounter[1]]
    metricPercentage[2] = [x /288 for x in metricCounter[2]]
    return metricPercentage        
    

# Function to determine which time interval

def TimeInterval(time):
    timeTotal = time.split(":")
    hours = int(timeTotal[0])
    minutes = int(timeTotal[1])
    seconds = int(timeTotal[2])
    if hours > 6:
        return 1    # Day Time
    elif hours < 6:
        return 0    # Night Time
    elif minutes > 0:
        return 1    # Day Time
    elif seconds > 0:
        return 1    # Day Time
    else:
        return 0    # Nighttime


# Function to take a list of dates, datalist and output required row

def MultiPercentageCalc(listOfDates, listofData):
    allMatrix = []
    for i in listOfDates:
        allMatrix.append(PercentageCalc(i,listofData))
    
    results = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in allMatrix:
        column = 0
        for j in i:
            for k in j:
                results[column] = results[column] + k
                column += 1
    numberOfDates = len(listOfDates)
    results = [x / numberOfDates for x in results]
    return results



# Start of calling stuff

#Import all data into a list
listData = fileToList(fileCGM)

#Determine which dates have 288 entries and zero empty data slots
listDatesFullData = listOfDatesFullData(listData)

#Create a list of manual dates and auto dates
listManualDates, listAutoDates = listOfDatesManualAuto(listDatesFullData, ManOffAutoStart(fileInsulin))

#Format data for Results.csv
row1 = MultiPercentageCalc(listManualDates,listData)
row2 = MultiPercentageCalc(listAutoDates, listData)

row1 = [x * 100 for x in row1]
row2 = [x * 100 for x in row2]

row1String = ','.join(map(str,row1))
row2String = ','.join(map(str,row2))

outfile = open(fileResult, 'w')
outfile.write(row1String + "\n" + row2String)

outfile.close()
