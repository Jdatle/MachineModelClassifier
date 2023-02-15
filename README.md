# MachineModelClassifier
## Machine Model Classifier using data from glucose sensors
```
This is a 3 part individual class project for Data Mining at ASU
```
## SKILLS DEMONSTRATED
```
PART 1: (Branch Part-1)
   - Extract & Synchronize Feature data from two sensors.
   - Compute and report overall statistical measures from data.
PART 2: (Branch Part-2)
   - Develop code to train a machine model to classify given sample data.
   - Assess the accuracy of a machine model.
PART 3: (Branch Part-3)
   - Develop code that performs DBSCAN & KMeans clustering.
   - Test and analyze the results of the clustering code.
   - Assess the accuracy of the clustering using SSE and supervised clustering validity metrics.
```
## INCLUDED FILES:
#### Part 1:
```
CGMData.csv
InsulinData.csv
main.py
Results.csv
```
#### Part 2:
```
CGMData.csv
CGM_patient2.csv
InsulinData.csv
Insulin_patient2.csv
train.py
dataMatrixAtt.pkl
test.py
Result.csv
```
#### Part 3:
```
CGMData.csv
InsulinData.csv
main.py
Results.csv
```

### SUMMARY OF REQUIREMENTS
#### PART 1: 
##### Given two datasets with the following info in a specific column:
```
Continuous Glucose Sensor (CGMData.csv)
	-Data time stamp (Column B & C)
	-5 Minute CGM Reading in mg/DL (Column AE)
```
```
Insulin Pump (InsulinData.csv)
	-Data Time Stamp
	-Meal intake amount in terms of grams of carbohydrates
	-Auto Mode (If auto mode is on or off)
```
##### Extract the following data in 3 different time intervals (Day, Night, All Day)
```
-Percentage time CGM > 180 mg/dL			(Hyperglycemia)
-Percentage time CGM > 250 mg/dL			(Hyperglycemia Critical)
-Percentage time 70 mg/dL =< CGM <= 180 mg/dL		
-Percentage time 70 mg/dL =< CGM <= 150 mg/dL
-Percentage time CGM < 70 mg/dL				(Hypoglycemia Level 1)
-Percentage time CGM < 54 mg/dL 			(Hypoglycemia Level 2)
```
```
Day (6AM to Midnight), Night (Midnight to 6AM), All Day (12AM to 12AM)
```
```
Extract these metrics for both Manual mode and Auto mode
```
#### Output
```
Output Results.csv (2 x 18 matrix) [Manual Mode + Auto Mode] x [6x Day + 6x Night + 6x All Day]
```

#### PART 2:
##### Given Meal Data & No Meal data of 2 people & Ground Truth labels of Meal and No Meal for 2 people
```
CGMData.csv
CGM_patient2.csv
InsulinData.csv
Insulin_patient2.csv
```
```
Train a machine model to predict wheather a given sample represents a person who has eaten a meal or not eaten a meal.
```
##### Following Tasks to be performed
###### Extract Features from Meal and No Meal Training Data Set (train.py)
```
Achieved by modifying part 1 to identify potential two hour meal windows and extract the following features:
	- Max CGM
	- Delta CGM
	- Normalized CGM Difference
	- Peak & Freq of Fast Fourier Transforms of the 1st derivative average from mealSlot to Max
	- Peak & Freq of Fast Fourier Transforms of the 2nd derivative average from mealSlot to Max
```
###### Train a machine to recognize Meal or No Meal Data & "Pickle" the data attribute matrix (train.py)
```
This is achieved by batching each feature into 5 normalized partitions. From this a decision tree style classification matrix was built.
```
###### Use K fold cross validation on training data to evaluate your machine (test.py)
```
Develop function that takes single test sample as input & outputs 1 (for predicted meal sample) or 0 (for predicted no meal sample)
Output results in Results.csv
```

### PART 3:
#### Extract Features from Meal data & Cluster Meal data based on the amount of carbs in each meal
```
Data Files:
	- CGMData.csv
	- InsulinData.csv
```
##### Extract Ground Truth
```
Derive Max and Min value of meal intake from the insulin data. Split into 20 bins.
```
##### Use feature extractor from Part 2 and cluster meal data using DBSCAN & KMeans.
```
Report accuracry of clustering based on SSE, Entropy, & Purity for KMeans and DBSCAN
Output results in Results.csv.
```




