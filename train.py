# John Le's Project 2 (ASU CSE 572 FALL A 2022)

# File Incoming Files

from cmath import exp, pi
import numpy as np
from numpy.fft import fft
import pickle

fileCGM = "CGMData.csv"
fileInsulin = "InsulinData.csv"
fileCGM2 = "CGM_patient2.csv"
fileInsulin2 = "Insulin_patient2.csv"

# FUNCTION DEFINITIONS


# # Parse single Insulin line

def ParseLineInsulin(string):
    string = string.split(",")
    parseLine = [string[1], string[2], string[24]]   
    return parseLine

def ParseLineInsulin2(string):
    string = string.split(",")
    parseLine = [string[2], string[3], string[25]]   
    return parseLine

def RemoveEmpties(list):
    resultList = []
    for i in list:
        if i[2] != '' and i[2] != '0':
            line = [i[0],i[1]]
            resultList.append(line)
    return resultList

# # Create Big List of imported Insulin Data

def fileInsulinToList(filepath):
    infile = open(filepath)
    list = []
    next(infile)
    for line in infile:
        list.append(ParseLineInsulin(line))
    infile.close()
    return list

def fileInsulinToList2(filepath):
    infile = open(filepath)
    list = []
    next(infile)
    for line in infile:
        list.append(ParseLineInsulin2(line))
    infile.close()
    return list


# # Create Big List of imported CGM Data

def fileCGMToList(filepath):
    infile = open(filepath)
    list = []
    next(infile)
    for line in infile:
        list.append(ParseLineCGM(line))
    infile.close()
    return list

def ParseLineCGM(string):
    string = string.split(",")
    parseLine = [string[1], string[2], string[30]]   
    return parseLine

def fileCGMToList2(filepath):
    infile = open(filepath)
    list = []
    next(infile)
    for line in infile:
        list.append(ParseLineCGM2(line))
    infile.close()
    return list

def ParseLineCGM2(string):
    string = string.split(",")
    parseLine = [string[2], string[3], string[31]]   
    return parseLine

# Return latest Time

def LatestTime(time1, time2):
    timeTotal1 = time1.split(":")
    hours1 = int(timeTotal1[0])
    minutes1 = int(timeTotal1[1])
    seconds1 = int(timeTotal1[2])
    timeTotal2 = time2.split(":")
    hours2 = int(timeTotal2[0])
    minutes2 = int(timeTotal2[1])
    seconds2 = int(timeTotal2[2])

    if hours1 > hours2:
        return time1
    elif hours1 < hours2:
        return time2
    elif minutes1 > minutes2:
        return time1
    elif minutes1 < minutes2:
        return time2
    elif seconds1 > seconds2:
        return time1
    elif seconds1 < seconds2:
        return time2
    else:
        return time1

# Given a meal time, returns pre meal and post meal range

def MealTimeRange(time):
    timeTotal = time.split(":")
    hours = int(timeTotal[0])
    minutes = int(timeTotal[1])
    seconds = int(timeTotal[2])
    postHours = hours + 2
    postMinutes = minutes

    if minutes >= 30:
        preHours = hours
        preMinutes = minutes - 30
    elif minutes < 30:
        preHours = hours - 1
        preMinutes = 60 - (30 - minutes)

    if postHours >= 24:
        postHours = "EndOfDay"
    
    preMealTime = str(preHours) + ":" + str(preMinutes) + ":" + str(seconds)
    postMealTime = str(postHours) + ":" + str(postMinutes) + ":" + str(seconds)

    resultList = [preMealTime, postMealTime]
    return resultList

# Given a list of meal times, Makes a list of meals that don't overlap in meal time range

def PotentialMealTimeFinder(list):
    resultList = []
    nextMeal = list[0]
    currentMeal = list[1]
    for i in list:
        if i == list[0] or i == list[1]:
            pass
        elif LatestTime(i[1],"0:29:59") == "0:29:59":
            pass
        elif LatestTime(i[1],"22:00:00") == i[1]:
            pass        
        else:
            preMeal = i
            if currentMeal[0] == preMeal[0] and currentMeal[0] == nextMeal[0]:  # IF all dates are equal
                mealRange = MealTimeRange(currentMeal[1])
                if LatestTime(preMeal[1], mealRange[0]) == preMeal[1]:
                    pass
                elif LatestTime(nextMeal[1], mealRange[1]) == mealRange[1]:
                    pass
                else:
                    # resultline = currentMeal[0] + "," + currentMeal[1] + "," + mealRange[0] + "," + mealRange[1]
                    resultline = [currentMeal[0], currentMeal[1],mealRange[0],mealRange[1]]
                    resultList.append(resultline)
                    nextMeal = currentMeal
                    currentMeal = preMeal
            else:
                mealRange = MealTimeRange(currentMeal[1])
                # resultline = currentMeal[0] + "," + currentMeal[1] + "," + mealRange[0] + "," + mealRange[1]
                resultline = [currentMeal[0], currentMeal[1],mealRange[0],mealRange[1]]
                resultList.append(resultline)
                nextMeal = currentMeal
                currentMeal = preMeal            
    return resultList   

def NoMealTimeKeepOut(list):
    resultList = []
    for i in list:
        mealDate = i[0]
        mealTime = i[1]
        mealTimePlus2 = MealTimeRange(mealTime)[1]
        resultList.append([mealDate, mealTime,mealTimePlus2])
    return resultList

# Given a time, start time, end time, return True or False if Time is within window

def WithinTimeInterval(time, timeStart, timeEnd):
    totalTime = time.split(":")
    hours = int(totalTime[0])
    mins = int(totalTime[1])
    secs = int(totalTime[2])    
    totalTimeStart = timeStart.split(":")
    hoursStart = int(totalTimeStart[0])
    minsStart = int(totalTimeStart[1])
    secsStart = int(totalTimeStart[2])    
    totalTimeEnd = timeEnd.split(":")
    hoursEnd = int(totalTimeEnd[0])
    minsEnd = int(totalTimeEnd[1])
    secsEnd = int(totalTimeEnd[2])    

    if hoursStart < hours and hours < hoursEnd:
        return True
    elif hoursStart > hours or hours > hoursEnd:
        return False
    elif hoursStart == hours:
        if minsStart < mins:
            return True
        elif minsStart > mins:
            return False
        else:
            if secsStart < secs:
                return True
            elif secsStart > secs:
                return False
    elif hoursEnd == hours:
        if minsEnd > mins:
            return True
        elif minsEnd < mins:
            return False
        else:
            if secsEnd > secs:
                return True
            elif secsEnd < secs:
                return False
    return True


# Determine if CGM data is Meal or No Meal Data

def ClassifyCGM(listCGM, potentialMealList):
    resultListMeal = []
    resultListNoMeal = []
    noMealListIntervals = []
    j = 0
    jMax = len(listCGM) - 1

    jStart = 0
    jEnd = 0

    for i in potentialMealList:
        found = False
        # print(i[0])
        while found == False:
            if j >= jMax:
                found = True
                break
            if i[0] == listCGM[j][0]:   # If Dates are the same
                if WithinTimeInterval(listCGM[j][1],i[2],i[3]) == True:
                    jEnd = j
                    if jStart > 0:
                        noMealListIntervals.append([jStart,jEnd])
                    resultLine = []
                    missingValue = False
                    for k in range(j,j+30,1):
                        if listCGM[k][2] == '' or listCGM[k][2] == 'NaN':
                            missingValue = True
                        else:
                            resultLine.append(float(listCGM[k][2]))
                        # resultLine.append(listCGM[k][2])
                    j = j + 31
                    jStart = j
                    if missingValue == False:
                        resultLine.reverse()
                        resultListMeal.append(resultLine)
                    # print("found")
                    found = True
                else:
                    j += 1
            else:
                j += 1

    for i in noMealListIntervals:
        j = i[0]
        while i[1]-j >= 24:
            resultLine = []
            missingValue = False
            for k in range(j,j+24,1):
                if listCGM[k][2] == '' or listCGM[k][2] == 'NaN':
                    missingValue = True
                else:
                    resultLine.append(float(listCGM[k][2]))
                # resultLine.append(listCGM[k][2])
            if missingValue == False:
                resultLine.reverse()
                resultListNoMeal.append(resultLine)
            j = j + 24


    return resultListMeal, resultListNoMeal

def FeatureExtractor(list):
    mealSlot = 0
    mealCGM = 0
    classification = 0
    if len(list) > 24:
        mealSlot = 5
        classification = 1
    mealCGM = list[mealSlot]
    
    # Max CGM, Delta CGM, Normalize CGM difference
    maxCGM = max(list)
    deltaCGM = maxCGM - mealCGM
    tau = (maxCGM - mealCGM) / mealCGM

    # first and 2nd derivative avg from mealSlot to Max (8 numbers)
    dCGM_dt = 0
    d2CGM_dt = 0
    slopes = []
    slopes2 = []
    n = 8   # of slots from meal time to max (Assume 45 mins / 6)
    
    for i in range(mealSlot,mealSlot + n, 1):
        slope = list[i+1] -list[i]
        slopes.append(slope)
    dCGM_dt = sum(slopes)/len(slopes)

    for i in range(0, n - 1, 1):
        slope2 = slopes[i+1] - slopes[i]
        slopes2.append(slope2)
    d2CGM_dt = sum(slopes2)/len(slopes2)

    # Fast Fourier Transform

    samplingRate = 100
    fftList = fft(list)
    lengthFFT = len(fftList)
    n = np.arange(lengthFFT)
    timeStep = lengthFFT / samplingRate
    frequency = n / timeStep
    absFFT = np.abs(fftList)

    pf1 = absFFT[1]
    f1 = frequency[1]
    pf2 = absFFT[2]
    f2 = frequency[2]

    features = [deltaCGM, tau, dCGM_dt, d2CGM_dt, pf1, f1, pf2, f2, classification]

    return features


# Transpose the data matrix

def TransposeDataMatrix(matrix):
    resultsMatrix = []

    for k in range(0,len(matrix[0]),1):
        resultsMatrix.append([])

    for i in matrix:
        for k in range(0,len(matrix[0]),1):
            resultsMatrix[k].append(i[k])
    return resultsMatrix


# Partition the attributes into n paritions based on values

def partitionAttributes(matrix, nPartitions):
    n = nPartitions
    kMax = int(n) + 1
    matrixTP = TransposeDataMatrix(matrix)
    resultsMatrix = []

    for k in range(0,len(matrixTP),1):
        resultsMatrix.append([])

    for i in range(0,len(matrixTP),1):

        if matrixTP[i] == matrixTP[-1] and max(matrixTP[-1]) == 1:
            pass
        else:
            maxAtt = max(matrixTP[i])
            minAtt = min(matrixTP[i])
            rangeInterval = maxAtt - minAtt
            interval = rangeInterval/n

            for k in range(1,kMax,1):

                lowerLimit = minAtt + (k-1)*interval
                upperLimit = minAtt + k*interval

                for j in range(0,len(matrixTP[i]),1):
                    if matrixTP[i][j] == maxAtt:
                        matrixTP[i][j] = n
                    elif lowerLimit <= matrixTP[i][j] and matrixTP[i][j] < upperLimit:
                        matrixTP[i][j] = k
            
                
    resultMatrix = TransposeDataMatrix(matrixTP)
    return resultMatrix


# Probability Calcs of a single class

def ProbsClass(classX,matrix):
    # assumes class is in last column
    foundCounter = 0
    total = len(matrix)
    for i in matrix:
        if i[-1] == classX:
            foundCounter += 1
    result = foundCounter / total
    return result

# Probability Calc of an attribute, given class

def ProbAttGivenClass(attValue, attColumn, classX, matrix):
    n = 1
    maxAttValue = max(matrix[attColumn])
    minAttValue = min(matrix[attColumn])
    c = (maxAttValue - minAttValue + 1)*n
    foundAttandClass = 0 + n
    foundClass = 0 + c
    for i in matrix:
        if i[-1] == classX:
            foundClass += 1
        if i[-1] == classX and i[attColumn] == attValue:
            foundAttandClass += 1
    result = foundAttandClass / foundClass
    return result

def ProbAttContGivenClass(attValue, attColumn, classX, matrix):
    list_Att_and_Class_Matching = []
    for i in matrix:
        if i[-1] == classX:
            list_Att_and_Class_Matching.append(i[attColumn])
    n = len(list_Att_and_Class_Matching)
    mean = sum(list_Att_and_Class_Matching) / n
    
    sSquare = 0
    for i in list_Att_and_Class_Matching:
        sSquare = (i - mean)**2
    
    sSquare = sSquare / n
    s = sSquare**0.5

    result = (1/( ((2*pi)**0.5)*s))**exp( -(  ((attValue-mean)**2)/(2*sSquare)  ) )

    return result

# Classify Machine

def Classify(profileX, matrix):
    result = 0
    probClass0 = ProbsClass(0,matrix)
    probClass1 = ProbsClass(1,matrix)
    # print(probClass1.real)
    # print(probClass0.real)
    for i in range(0,len(profileX),1):
        probs0 = ProbAttContGivenClass(profileX[i],i,0,matrix)
        # print(probs0.real)
        probClass0 = probClass0*probs0
        
    for i in range(0,len(profileX),1):
        probs1 = ProbAttContGivenClass(profileX[i],i,1,matrix)
        # print(probs1.real)
        probClass1 = probClass1*probs1


    if probClass1 >= probClass0:
        result = 1
    return result

def ClassifyMulti(matrixTest, matrixLabel):
    result = []
    for i in matrixTest:
        resultLine = Classify(i,matrixLabel)
        result.append(resultLine)
    return result
        

def ClassifySplit(profileX, matrix):
    result = 0
    probClass0 = ProbsClass(0,matrix)
    probClass1 = ProbsClass(1,matrix)
    # print(probClass1.real)
    # print(probClass0.real)
    for i in range(0,len(profileX),1):
        probs0 = ProbAttGivenClass(profileX[i],i,0,matrix)
        # print(probs0.real)
        probClass0 = probClass0*probs0
        
    for i in range(0,len(profileX),1):
        probs1 = ProbAttGivenClass(profileX[i],i,1,matrix)
        # print(probs1.real)
        probClass1 = probClass1*probs1


    if probClass1 >= probClass0:
        result = 1
    return result

def ClassifySplitMulti(matrixTest, matrixLabel):
    result = []
    for i in matrixTest:
        resultLine = ClassifySplit(i,matrixLabel)
        result.append(resultLine)
    return result



#PROJECT II Calling Stuff

#Import all data into a list
listDataInsulin1 = RemoveEmpties(fileInsulinToList(fileInsulin))
listDataCGM1 = fileCGMToList(fileCGM)
listDataInsulin2 = RemoveEmpties(fileInsulinToList2(fileInsulin2))
listDataCGM2 = fileCGMToList2(fileCGM2)

potentialMealTimeList1 = PotentialMealTimeFinder(listDataInsulin1)
potentialMealTimeList2 = PotentialMealTimeFinder(listDataInsulin2)

# Create lists of meals and noMeals CGM readings
mealCGMList1, noMealCGMList1 = ClassifyCGM(listDataCGM1,potentialMealTimeList1)
mealCGMList2, noMealCGMList2 = ClassifyCGM(listDataCGM2,potentialMealTimeList2)

# # Combine the two data points
mealCGM = mealCGMList1 + mealCGMList2
noMealCGM = noMealCGMList1 + noMealCGMList2

# mealCGM = mealCGMList1
# noMealCGM = noMealCGMList1

#Create Test CGM PRACTICE TEST
# testCGM = mealCGMList2 + noMealCGMList2
# fileTest = "test.csv"
# outfile = open(fileTest, 'w')
# for i in testCGM:
#     for j in range(0,len(i),1):
#         # if j == len(i)-1:
#         #     outfile.write(str(i[j]))
#         # else:
#             outfile.write(str(i[j])+",")
#     outfile.write("\n")
# outfile.close()


#Create Data Matrix with FeatureExtracted Data and their Classifications
dataMatrix = []
for i in mealCGM:
    inputLine = FeatureExtractor(i)
    dataMatrix.append(inputLine)

for i in noMealCGM:
    inputLine = FeatureExtractor(i)
    dataMatrix.append(inputLine)

##################################
#  TESTING STUFF #
####################################

# Create Test Features and Remove Classification
# testMatrix = []

# for i in testCGM:
#     testMatrix.append(FeatureExtractor(i))

# for i in range(0,len(testMatrix),1):
#     testMatrix[i].pop(-1)


#
# test = ClassifyMulti(testMatrix,dataMatrix)
# print(test)

# testMatrix2 = []

# for i in testCGM:
#     inputLine = FeatureExtractor(i)
#     testMatrix2.append(inputLine)

# for i in range(0,len(testMatrix2),1):
#     testMatrix2[i].pop(-1)

# testMatrixAtt = partitionAttributes(testMatrix2,6)
# dataMatrixAtt = partitionAttributes(dataMatrix,6)

# test2 = ClassifySplitMulti(testMatrixAtt,dataMatrixAtt)
# print(test2)

##################################################################

dataMatrixAtt = partitionAttributes(dataMatrix,5)

# Time to Pickle

with open('dataMatrixAtt.pkl','wb') as dataPickle:
    pickle.dump(dataMatrixAtt,dataPickle)



# MORE RANDOM TESTING

# dataMatrixAtt = partitionAttributes(dataMatrix,6)
# print(dataMatrixAtt[0])
# print(dataMatrixAtt[-1])

# print(dataMatrixAtt[0])

# print(ProbsClass(1,dataMatrixAtt))
# print(ProbsClass(0,dataMatrixAtt))

# print(ProbAttGivenClass(1,0,1,dataMatrixAtt))
# print(ProbAttGivenClass(2,0,1,dataMatrixAtt))
# print(ProbAttGivenClass(3,0,1,dataMatrixAtt))
# print(ProbAttGivenClass(4,0,1,dataMatrixAtt))
# print(ProbAttGivenClass(5,0,1,dataMatrixAtt))
# print(ProbAttGivenClass(6,0,1,dataMatrixAtt))

# print(ProbAttContGivenClass(55,0,1,dataMatrix))
# print(ProbAttContGivenClass(55,0,0,dataMatrix))

# test = dataMatrix[0].copy()
# test.pop(-1)
# print(dataMatrix[0])
# print(test)

# print(Classify(test,dataMatrix))