# John Le's Project 3 (ASU CSE 572 FALL A 2022)

from cmath import exp, pi
from decimal import ROUND_UP
from multiprocessing.dummy import Array
from re import M
import numpy as np
from numpy.fft import fft
import math

# File Incoming Files
fileCGM = "CGMData.csv"
fileInsulin = "InsulinData.csv"

# FUNCTION DEFINITIONS

## STUFF FROM PROJECT II

# # Parse single Insulin line

def ParseLineInsulin(string):
    string = string.split(",")
    parseLine = [string[1], string[2], string[24]]   
    return parseLine

def RemoveEmpties(list):
    resultList = []
    for i in list:
        if i[2] != '' and i[2] != '0':
            line = [i[0],i[1],i[2]]
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
                    resultline = [currentMeal[0], currentMeal[1],mealRange[0],mealRange[1],i[2]]
                    resultList.append(resultline)
                    nextMeal = currentMeal
                    currentMeal = preMeal
            else:
                mealRange = MealTimeRange(currentMeal[1])
                # resultline = currentMeal[0] + "," + currentMeal[1] + "," + mealRange[0] + "," + mealRange[1]
                resultline = [currentMeal[0], currentMeal[1],mealRange[0],mealRange[1],i[2]]
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
    resultListCarbs = []
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
                        resultListCarbs.append(i[4])
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


    return resultListMeal, resultListNoMeal, resultListCarbs


## FEATURE EXTRACTOR

def FeatureExtractor(list,number):
    mealSlot = 0
    mealCGM = 0
    classification = float(number)
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


# Classify Carb input into bins

def PartitionCarbs(matrix):
    matrixResult = matrix.copy()
    binSize = 20
    minCarbs = 999
    maxCarbs = 0

    for i in dataMatrix:
        minCarbs = min(i[8],minCarbs)
        maxCarbs = max(i[8],maxCarbs)
    
    nBins = math.ceil((maxCarbs-minCarbs)/20)

    for k in range(0,nBins,1):
        lowerLimit = minCarbs + k*binSize
        upperLimit = minCarbs + (k+1)*binSize
        for j in range(0,len(matrixResult),1):
            if matrixResult[j][8] >= lowerLimit and matrixResult[j][8] < upperLimit:
                matrixResult[j][8] = k+1
    return matrixResult

def PartitionCarbs2(matrix):
    matrixResult = matrix.copy()
    binSize = 20
    minCarbs = 999
    maxCarbs = 0

    for i in dataMatrix:
        minCarbs = min(i[8],minCarbs)
        maxCarbs = max(i[8],maxCarbs)
    
    nBins = math.ceil((maxCarbs-minCarbs)/20)

    for k in range(0,nBins,1):
        lowerLimit = minCarbs + k*binSize
        upperLimit = minCarbs + (k+1)*binSize
        for j in range(0,len(matrixResult),1):
            if matrixResult[j][8] >= lowerLimit and matrixResult[j][8] < upperLimit:
                matrixResult[j][8] = k+1
    return matrixResult    


# Calculate Distance between two data entries

def DistanceCalc(array1,array2,nAttributes):
    sumSquare = 0
    for i in range(0,nAttributes,1):
        sumSquare += (array1[i] - array2[i])**2
    distance = sumSquare**.5
    return distance


# Perform kMeans
def kMeans(dataMatrix):
    tpMatrix = TransposeDataMatrix(dataMatrix)
    k = max(tpMatrix[8])
    clusters = []
    nAtts = 8

    #Get Initial cluster locations (Assume 1st found on list)
    # for j in range(0,k,1):
    #     clusters.append(dataMatrix[j])

    #Try average of initial spots
    clusters = ClusterCGs(dataMatrix,nAtts,k)

    #Add extra column for cluster assignment

    matrixPlusCol = dataMatrix
    for i in matrixPlusCol:
        i.append(999)

    matrixAssigned = []
    for i in matrixPlusCol:
        matrixAssigned.append(AssignCluster(i,nAtts,clusters))
    
    # test = ClusterCGs(matrixAssigned,nAtts,k)

    currentClusters = clusters.copy()
    nextClusters = ClusterCGs(matrixAssigned,nAtts,k)
    iterations = 0
    while currentClusters != nextClusters:
        currentClusters = nextClusters.copy()
        newMatrixAssigned = []
        for i in matrixPlusCol:
            newMatrixAssigned.append(AssignCluster(i,nAtts,nextClusters))
        nextClusters = ClusterCGs(newMatrixAssigned,nAtts,k)
        iterations += 1

    # print(iterations)
    return newMatrixAssigned, nextClusters

    # Assign cluster

def AssignCluster(array,nAttributes,clusterArray):
    k = len(clusterArray)
    kMin = 999
    minDistance = 99999
    for i in range(0,len(clusterArray),1):
        distance = DistanceCalc(clusterArray[i],array,nAttributes)
        if distance <= minDistance:
            minDistance = distance
            kMin = i+1
    array[nAttributes+1] = kMin
    return array

# Calculate Cluster CGs

def ClusterCGs(arrayWithClusters,nAttributes,k):
    clustersByBin = []
    for i in range(0,k,1):
        arraySingleCluster = []
        for j in range(0,len(arrayWithClusters),1):
            if arrayWithClusters[j][-1] == i + 1:
                arraySingleCluster.append(arrayWithClusters[j])
        clustersByBin.append(arraySingleCluster)
    # print(clustersByBin[2][5])

    clusterList = []
    for i in range(0,k,1):
        singleClusterCG = []
        tpMatrix = TransposeDataMatrix(clustersByBin[i])
        for j in range(0,nAttributes,1):
            length = len(tpMatrix[j])
            sum = 0
            for m in range(0,length,1):
                sum += tpMatrix[j][m]
            average = sum/length
            singleClusterCG.append(average)
        singleClusterCG.append(i+1)
        clusterList.append(singleClusterCG)
    return clusterList

# Calculate Entropy

def CalcEntropyPurity(list):
    tpMatrix = TransposeDataMatrix(list)
    nBins = max(tpMatrix[-2])
    totalList = len(list)
    
    purityTotal = 0
    entropyTotal = 0

    for i in range(0,nBins,1):
        clusterI = i+1
        totalClusterI = 0
        maxPurityI = 0
        for o in list:
            if o[-1] == clusterI:
                totalClusterI += 1
        
        entropyClusterI = 0

        for j in range(0,nBins,1):
            binJ = j+1
            totalBinJ = 0
            for m in list:
                if m[-1] == clusterI and m[8] == binJ:
                    totalBinJ += 1
            pBinJ = totalBinJ / totalClusterI
            maxPurityI = max(pBinJ,maxPurityI)
            entropy_clusterI_binJ = CalcEntropySingle(pBinJ) 
            # print(totalBinJ)
            # print(totalClusterI)
            # print(entropy_clusterI_binJ)
            entropyClusterI += entropy_clusterI_binJ

        entropyTotal += (totalClusterI/totalList)*entropyClusterI
        purityTotal += (totalClusterI/totalList)*maxPurityI
        # print(totalClusterI)
        # print(totalList)
        # print(entropyClusterI)
    
    return entropyTotal, purityTotal


def CalcEntropySingle(prob):
    if prob == 0:
        return 0
    else:
        result = -prob*np.log2(prob)
        return result

def SumSquaredErrors(list,cluster):
    SumSquared = 0
    for i in list:
        for j in cluster:
            if i[-1] == j[-1]:
                distance = DistanceCalc(i,j,8)
                SumSquared += distance**2

    return SumSquared


def DBSCAN(dataMatrix2,eps,minPts):
    nAtt = 8
    distanceMatrix = CalcDistanceMatrix(dataMatrix2,nAtt)
    
    # list of labels core, border, or noise
    neighborList = []
    for i in range(0,len(distanceMatrix),1):
        counter = 0
        ptsWithinEps = []
        for j in range(0,len(distanceMatrix[i]),1):
            if distanceMatrix[i][j] <= eps:
                counter += 1
                ptsWithinEps.append(j)
        numberAndNeighbors = [i,counter,ptsWithinEps]
        neighborList.append(numberAndNeighbors)

    corePts = []
    for i in neighborList:
        if i[1] >= minPts:
            corePts.append(i)
    
    borderPts = []
    for i in range(0,len(neighborList),1):
        found = False
        jMax = len(corePts)
        j = 0
        while found == False and j < jMax:
            if neighborList[i][0] == corePts[j][0]:
                pass
            elif neighborList[i][0] in corePts[j][2]:
                borderPts.append([i,corePts[j][0]])
                found = True
            j += 1

    coreClusterList = []
    for i in range(0,len(corePts),1):
        # print("i="+str(i))
        for j in range(0,len(corePts),1):
            # print("j="+str(j))
            ptI = corePts[i][0]
            ptJ = corePts[j][0]
            distance = DistanceCalc(dataMatrix2[ptI],dataMatrix2[ptJ],nAtt)
            if i == j:
                pass
            elif distance <= eps:
                if len(coreClusterList) == 0:
                    coreClusterList.append([ptI,ptJ])
                    # print("added first pts")
                found = False
                mMax = len(coreClusterList)
                m = 0
                while found == False and m < mMax:
                # for m in range(0,len(coreClusterList),1):
                # for m in range(0,2,1):
                    # print("m="+str(m))
                    if ptI in coreClusterList[m] and ptJ in coreClusterList[m]:
                        found = True
                    elif ptI in coreClusterList[m] and (ptJ in coreClusterList[m]) == False:
                        coreClusterList[m].append(ptJ)
                        found = True
                        # print("added ptJ")
                    elif ptJ in coreClusterList[m] and (ptI in coreClusterList[m]) == False:
                        coreClusterList[m].append(ptI)
                        found = True
                        # print("added ptI")
                    m += 1
                if found == False:
                    coreClusterList.append([ptI,ptJ])
                    # print("added Both points")
    clusterArray = []
    for i in coreClusterList:
        singleCluster = []
        for j in i:
            singleCluster.append(dataMatrix2[j])
        clusterArray.append(singleCluster)

    # # Classify Clusters
    clusterClassify = [0 for _ in range(len(coreClusterList))]
    for i in range(0,len(clusterArray),1):
        classifyCounter = [0 for _ in range(len(coreClusterList))]
        for j in clusterArray[i]:
            classifyCounter[j[-1] - 1] += 1
        clusterClassify[i] = i + 1
    
    resultList = []
    for i in range(0,len(clusterArray),1):
        classification = clusterClassify[i]
        # print("i="+str(i))
        for j in range(0,len(clusterArray[i]),1):
            # print("J="+str(j))
            clusterArray[i][j].append(classification)
            # resultLine = clusterArray[i][j]
            # print(resultLine)
            # resultLine.append(classification)
            # print(resultLine)
            resultList.append(clusterArray[i][j])
    
    clusterCG = ClusterCGs(resultList,nAtt,5)
    return resultList, clusterCG
            

def CalcDistanceMatrix(dataMatrix,nAtt):
    resultMatrix = []
    for i in dataMatrix:
        rowI = []
        for j in dataMatrix:
            distanceIJ = DistanceCalc(i,j,nAtt)
            rowI.append(distanceIJ)
        resultMatrix.append(rowI)
    return resultMatrix
    


    # PROJECT III Calling stuff

# Import all data into a list
listDataInsulin1 = RemoveEmpties(fileInsulinToList(fileInsulin))
listDataCGM1 = fileCGMToList(fileCGM)

potentialMealTimeList1 = PotentialMealTimeFinder(listDataInsulin1)

# Create lists of meals and noMeals CGM readings
mealCGMList1, noMealCGMList1, mealCarbsList1 = ClassifyCGM(listDataCGM1,potentialMealTimeList1)

#Create Data Matrix with FeatureExtracted Data and their Classifications (Carb input)
dataMatrix = []

for i in range(0,len(mealCGMList1),1):
    inputLine = FeatureExtractor(mealCGMList1[i],mealCarbsList1[i])
    dataMatrix.append(inputLine)


#add bin partitions
dataMatrixBin = PartitionCarbs(dataMatrix)


# Perform kmeans
kMeansList, kMeansClusterCGs = kMeans(dataMatrixBin)

# Remove k means clustering from data matix & make dataMatrixBin2
dataMatrixBin2 = []
for i in dataMatrixBin:
    singleArray = []
    for j in range(0,len(i),1):
        if j == len(i)-1:
            pass
        else:
            singleArray.append(i[j])
    dataMatrixBin2.append(singleArray)

# Perform DBSCAN
DBSCANList, DBSCANClusterCGS = DBSCAN(dataMatrixBin2,130,8)


#Calculate entropy, purity and SSE
entropy_kmeans, purity_kmeans = CalcEntropyPurity(kMeansList)
SSE_kmeans = SumSquaredErrors(kMeansList,kMeansClusterCGs)
entropy_DBSCAN, purity_DBSCAN = CalcEntropyPurity(DBSCANList)
SSE_DBSCAN = SumSquaredErrors(DBSCANList,DBSCANClusterCGS)

#Create result vector
outputArray = [SSE_kmeans,SSE_DBSCAN,entropy_kmeans,entropy_DBSCAN,purity_kmeans,purity_DBSCAN]
# print(outputArray)
 
output = ''
for i in range(0,len(outputArray),1):
    if i == len(outputArray)-1:
        output += str(outputArray[i])
    else:
        output += str(outputArray[i]) + ","


# Create submission
fileResult = "Result.csv"

outfile = open(fileResult, 'w')
outfile.write(output)
outfile.close()