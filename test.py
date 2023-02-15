import pickle
import numpy as np
from numpy.fft import fft


#################################
# FEATURE EXTRACTOR, Partition Divider, Probability Calcs, & Classifier
#################################

######################################################################################
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

    # features = [deltaCGM, tau, dCGM_dt, d2CGM_dt, pf1, f1, pf2, f2, classification]
    features = [deltaCGM, tau, dCGM_dt, d2CGM_dt, pf1, f1, pf2, f2]

    return features

######################################################################################

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

def TransposeDataMatrix(matrix):
    resultsMatrix = []

    for k in range(0,len(matrix[0]),1):
        resultsMatrix.append([])

    for i in matrix:
        for k in range(0,len(matrix[0]),1):
            resultsMatrix[k].append(i[k])
    return resultsMatrix

######################################################################################

def ProbsClass(classX,matrix):
    # assumes class is in last column
    foundCounter = 0
    total = len(matrix)
    for i in matrix:
        if i[-1] == classX:
            foundCounter += 1
    result = foundCounter / total
    return result

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

######################################################################################

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

######################################################################################
######################################################################################
######################################################################################
######################################################################################

#Must Unpickle

with open('dataMatrixAtt.pkl','rb') as dataPickle:
    dataMatrixAtt = pickle.load(dataPickle)

# Read in test.csv

fileTest = "test.csv"

def FileToList(filepath):
    infile = open(filepath)
    list = []
    for line in infile:
        input = line.split(",")
        list.append(input)
    infile.close

    # Check to remove \ns
    for i in range(0,len(list),1):
        if list[i][-1] == "\n":
            list[i].pop(-1)

    #Convert to Floats
    resultList = []
    for i in list:
        singleRow = []
        for j in i:
            singleRow.append(float(j))
        resultList.append(singleRow)

    return resultList

testCGM = FileToList(fileTest)

# Create Test Matrix with extracted features
testMatrix = []
for i in testCGM:
    testMatrix.append(FeatureExtractor(i))

# Partition Test Matrix
testMatrixAtt = partitionAttributes(testMatrix,5)

# Classify Test Matrix and make a list of outputs
classifiedList = ClassifySplitMulti(testMatrixAtt,dataMatrixAtt)

#Create Result File
fileResult = "Result.csv"

outfile = open(fileResult, 'w')
for i in classifiedList:
    outfile.write(str(i) + '\n')
outfile.close()

