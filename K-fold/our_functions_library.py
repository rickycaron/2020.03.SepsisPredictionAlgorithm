# -*- coding: utf-8 -*-
#  # Function Library
#  This file, FunctionLibrary contains most of the functions created by us, Jeroen & Rui
#  Notes: if you want to see the changes you made here in your imported file, you need to save the changes here and restart the running kernel of that file, then you can see the changes.
#  So it is not recommended to write code here, but only paste the finished code here.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.impute import KNNImputer

# defining the columnbs that are used for the features (X) and for the SepsisLabels (y)
X_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2','BaseExcess', 'HCO3', 'FiO2', 'pH', 
             'PaCO2', 'SaO2', 'AST', 'BUN','Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
             'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium','Bilirubin_total', 'TroponinI', 'Hct', 
             'Hgb', 'PTT', 'WBC','Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2','HospAdmTime',
             'ICULOS', 'Patient_id', 'time']
y_columns = ['Patient_id', 'SepsisLabel']


#Function for writing the different idSEts to a file
#input
    ##Isets: array with 10 index array[10] which contains the 10-fold splitting patients
    ##KforKFold: var which indicate in how many parts the "big patientsID's" is splitted
# Last modified JV - 18-3-21 10:45

def writePatientsIDtoFile(idSets, KforKFold):
    Twrite = time.time()
    open("test.txt", "w").close() # clear contents of existing file
    for i in range(KforKFold):
        splitted = " ".join( repr(e) for e in idSets[i])
        file1 = open("test.txt","a")
        file1.write("\n\n")
        file1.write(str("[" +splitted+"]"))
        file1.write("\n\n")
        file1.close()
    print("Time for writing id to file:", round(time.time()- Twrite,3))

## Funcions that are used for data splitting and inseide the KFold function

# +
#this function will clear everything in the training and test dataset
# Rui: I think this funcion definition should be insinde the file which calls it, not here, otherwise
# the function doesn't know what these variables are, or maybe try the new definition?
# Last modified RZ - 11-3-21 11:12

def clearAllDatasets(X_train, X_test, y_train, y_test):
    if not X_train.empty: 
        X_train = X_train[0:0]
    if not X_test.empty:
        X_test = X_test[0:0]
    if not y_train.empty:
        y_train = y_train[0:0]
    if not y_test.empty:
        y_test = y_test[0:0]
    return X_train, X_test, y_train, y_test
        
# def clearAllDatasets():
#     global X_train, X_test, y_train, y_test
#     if not X_train.empty: 
#         X_train = X_train[0:0]
#     if not X_test.empty:
#         X_test = X_test[0:0]
#     if not y_train.empty:
#         y_train = y_train[0:0]
#     if not y_test.empty:
#         y_test = y_test[0:0]


# -

# Function for printing metrics of the used Dataset
# Last modified RZ - 11-3-21 11:25
def printDataset(X_train, X_test, y_train, y_test):
    print('X train shape:',X_train.shape)
    print(X_train)
    print( 'Y train shape:',y_train.shape)
    print(y_train)
    print('X test shape:',X_test.shape)
    print(X_test)
    print( 'Y test shape:',y_test.shape)
    print(y_test)

# Function for datatype for the different splitted datasets(train and test for X&y)
# Last modified RZ - 11-3-21 11:25
def printDatasetType(X_train, X_test, y_train, y_test):
    print('X_train type:',type(X_train),'y_train type:',type(y_train) )
    print('X test shape:',type(X_test),'Y test shape:',type(y_test) )
    print('X_train data type:',(X_train.dtypes),'y_train data type:',(y_train.dtypes) )
    print('X_test data type:',(X_test.dtypes),'Y_test data type:',(y_test.dtypes) )

#Function for resetting the array's that are holding the results of the function 'findBestKforKNN'
# array's which holds mean and std of the executed Kfold operation with a model
# Last modified JV - 10-3-21 19:25
def KNN_reset():
    #KNN with different K size, 
    KNN_UtilityScore_mean.clear()
    KNN_UtilityScore_std.clear()
    KNN_F1Score_mean.clear()
    KNN_F1Score_std.clear()
    KNN_auroc_mean.clear()
    KNN_auprc_mean.clear()
    KNN_accuracy_mean.clear()
    KNN_accuracy_std.clear()
    KNN_positiveprediction_mean.clear()
    KNN_baseline_mean.clear()
    KNN_total_Time.clear()

# Function to generate the Training dataset
# input: 
    # test_patienIds: The  test patient_id which should not be included/be used for creating the training dataset
    # X_train, y_train: this are the array's which are used to use the generate train features(X) and labels (y)
# output: 
    ##X_train: all features of patient_id equal to patientsIds, 
    ###y_train: the sepsislabels of patients corresponding to patientIds
# Last modified JV - 18-3-21 10:30
def generateTrainDataSet(test_patienIds, X_train, y_train):
    # selecting data from patients with who's patientId is not in test_patienIds
    train_data= originalData[~(originalData.Patient_id.isin(test_patienIds))] 
    X_train = train_data[X_columns]
    y_train = train_data[y_columns]
    return X_train, y_train


#Function to generate the test dataset
# input: 
    ##PatientIds: The patient_id's which are going to be used for creating test dataset (data of patients where patient_id = patienIds)
    ## X_test, y_test: this are the array's which are used to use the generate test features(X) and labels (y)
# out:
    ##X_test: all features of patient_id equal to patientsIds, 
    ##y_test: the sepsislabels of patients corresponding to patientIds
# Last modified JV - 18-3-21 10:30
def generateTestDataSet(patienIds, X_test, y_test):
    #print("test_patienIds: \n", patienIds)
    # selecting data from patients with who's patientId is represent in test_patienIds
    test_data = originalData[originalData['Patient_id'].isin(patienIds)]
    X_test = test_data[X_columns]
    y_test = test_data[y_columns]
    return X_test, y_test


# Functions for Variable Manipulation

#Function where calculating the mean of a given input dataset
# Last modified JV - 10-3-21 19:30
def CalcMean_Std(Data):
    print(Data)
    mean = np.mean(Data, axis=0)
    np.set_printoptions(precision=3, suppress=True)
    print('numpy mean\n', mean)
    Data_std = Data.std(axis=0)
    print('Data_mean\n', Data_std.round(3))


# # Functions for missing Data imputation

# This function is for filling the NaN value of the data linearly, only for one column at a time
# inputs
    ## patientColumn: the column that is going to filled to eliminated the missing values in the column
    ## patientIndexSet:
    ## forwardFilling: 
        #True: perform forward filling for the mediam missing part,just copy the value of the same patient before NaN value
        #False: perform linear imputation filling for the missing part
#output
    ## patientColumn: return the linearly filled dataColumn
# Last modified JV - 10-3-21 19:40

def fillNaNValueColumnPatient(patientColumn, patientIndexSet, forwardFilling=False):
    rowNum = patientColumn.shape[0]
    beginIndex = patientIndexSet[0]
    endIndex = patientIndexSet[-1]
    # generate the true-false table for a column
    columnIsNull = patientColumn.isnull()
    # print(columnIsNull)
    # the first task is to fill all the beginging NaN
    i = 0
    firstIndex = beginIndex
    found = False
    # print("Before the filling")
    # print(patientColumn)
    # found the first non-nan value
    while (i < rowNum and (not found)):
        if not (columnIsNull[i + beginIndex]):
            found = True
            firstIndex = i + beginIndex
            break
        i += 1
    # print(idx,firstIndex)
    # After the search
    if found == False:
        # the whole column is NaN, just leave it untouched, we will fill all these kind of missing values togther later on
        # print(idx,"This whole column is NaN, fill it with 0 or global average value")
        return patientColumn  # because it's all NaN, no need to proceed in this function
    else:
        # fill all the first nan value with the first not-nan value you find
        patientColumn.loc[beginIndex:firstIndex] = patientColumn[firstIndex]

        # the second task is to fill all the ending NaN
    i = rowNum - 1
    lastIndex = endIndex
    found = False
    # print(columnIsNull)
    # found the last non-nan value
    while (i >= 0 and (not found)):
        if not (columnIsNull[beginIndex + i]):
            found = True
            lastIndex = beginIndex + i
            break
        i -= 1
    # print(idx,lastIndex)
    # After the search
    if found == True:
        # fill all the first nan value with the first not-nan value you find
        patientColumn.loc[lastIndex: endIndex] = patientColumn[lastIndex]

        # third task is to fill all the middle missing data
    # we must generate the isnullcolumn again here
    columnIsNull = patientColumn.isnull()
    # print(columnIsNull)
    inSearchForANaN = True
    NaNbeginIndex = beginIndex  # is NaN
    NaNendIndex = endIndex  # is not NaN
    i = 0
    # get into the loop
    while (i < rowNum):
        if (inSearchForANaN and columnIsNull[beginIndex + i]):
            inSearchForANaN = False
            NaNbeginIndex = beginIndex + i
            # print(beginIndex)
        elif (not inSearchForANaN and not columnIsNull[beginIndex + i]):
            inSearchForANaN = True
            NaNendIndex = beginIndex + i
            # print(endIndex)
            if forwardFilling == True:
                patientColumn.loc[NaNbeginIndex: NaNendIndex - 1] = patientColumn[NaNbeginIndex - 1]
            elif forwardFilling == False:
                sliceNum = NaNendIndex - NaNbeginIndex + 2
                newValues = np.linspace(patientColumn[NaNbeginIndex - 1], patientColumn[NaNendIndex], sliceNum).round(2)
                # print(newValues)
                # print(patientColumn [ NaNbeginIndex-1 : NaNendIndex+1])
                patientColumn.loc[NaNbeginIndex - 1: NaNendIndex] = newValues
        i += 1
    return patientColumn

# Functions below are filling the missing data
# This function perform the KNN imputation for missing data filling
# in this function, the missing values of the train/test dataset are calculated by means of KNN imputer 
# Last modified JV - 10-3-21 19:30

def KNNfilling(trainData,testData,K= 5, fillmethod=""):
    imputer = KNNImputer(n_neighbors = K)
    #imputer = FaissKNeighbors(k=K)
    # fit imputer on training data
    imputer.fit(trainData)
    #transfer
    # Impute the missing data of the train/test data using the imputer "model"
#     x_train_impute=imputer.transform(trainData).round(3)
#     x_test_impute=imputer.transform(testData).round(3)
    x_train_impute=imputer.transform(trainData)
    x_test_impute=imputer.transform(testData)
    fillmethod= "KnnFill"
    return x_train_impute, x_test_impute, fillmethod   #This may cause error when the data is very large in size


# -

#Last Modified: Rui 10/3 3:35pm, can fill the missing value both by overall and train mean value  
#fill the missing data by the overall mean value 
def MeanFilling(trainData,testData, fillmethod, overall = True):
    if(overall):
        data_concat = pd.concat((trainData, testData))
        fillingmean = data_concat.mean().round(2)  
        print("fill with overall data mean")
    else:
        fillingmean = trainData.mean().round(2)
        print("fill with training data mean")
    train_mean_filled = trainData.fillna(fillingmean)
    test_mean_filled = testData.fillna(fillingmean)
    fillmethod= "MeanFill"
    #print(train_mean_filled) 
    #print(test_mean_filled)
    return train_mean_filled, test_mean_filled, fillmethod


# This methods fills the whole datarame with linear imputation, here we assume that the data for each patient linearly over time
#Inputs
    ## oridata: the "original" data (with missing values) that needs to be filled (linearly)
    ## fillingChoice: The way of linearly filling the data (there are three options:0,1,2)
        # 0: leave the remaining NaN as NaN 
        # 1: fill remaining NaN values with 0 
        # 2: fill remaining NaN values with overall mean value
    ## forwardFilling: option to fill the data either forwardly or lineartly
# Last modified JV - 10-3-21 20:00

def linearFillingAll(oridata, fillingChoice=0, forwardFilling=False):
    start = time.time()
    filleddata = oridata
    if (forwardFilling == False):
        fillmethod = 'Linear_'
        print("Filling using Linear interpolation")
    elif (forwardFilling == True):
        fillmethod = 'Forw_'
        print("filling using Forward filling")
    Uniq_ID = np.unique(oridata['Patient_id'])
    for Pid in Uniq_ID:
        patientData = oridata.loc[oridata['Patient_id'] == Pid]
        patientIndex = patientData.index
        copy = patientData
        # for every column
        for idx in copy:
            copy[idx] = fillNaNValueColumnPatient(copy[idx], patientIndex, forwardFilling)
        filleddata.loc[filleddata['Patient_id'] == Pid] = copy
    print("The filling choice is of number:", fillingChoice)
    if fillingChoice == 1:
        fillmethod += str('0')
        filleddata = filleddata.fillna(0)
    elif fillingChoice == 2:
        fillmethod += str('mean')
        filleddata = filleddata.fillna(filleddata.mean().round(2))
    print("total time:", round(time.time() - start, 2), "sec")
    return filleddata, fillmethod


# +
#this function generates at each time randomly different X_train,y_train,X_test,y_test,group by the patient ID, from the file
# def generateDateset(randomState=2):
#     global X_train, X_test, y_train, y_test
#     #clean the exsiting dataset
#     clearAllDatasets()
#     #from UniqID numpy array, randomly generates train set and data set of patient ID
#     random.shuffle(Uniq_ID)
#     train_id, test_id = sklearn.model_selection.train_test_split(Uniq_ID, test_size=0.1, random_state=randomState, shuffle=True)
#     print(train_id)
#     print(test_id)
#     #append the data according to the generated array to the train and test dataset
#     generateTrainDataSet(patient_id_training_dataset)
#     generateTestDataSet(patient_id_test_dataset)
#     return X_train, X_test, y_train, y_test
# -

#Function for writing the test data (X + y) to a fill
# inputs:
    ## X_test_impute,y_test: data that contains the features and sepsisLabel of test data, that is going to be written away
    ## NFold: parameter for indication in which fold the data orginated
    ## fillmethod: paramater that indicates with which filling method the test data is filled
# Last modified JV - 10-3-21 20:20

def WriteToFiles(X_test_impute,y_test, NFold, method, fillmethod):
    x= pd.DataFrame(data=X_test_impute, columns=X_columns)
    y= pd.DataFrame(data=y_test, columns=y_columns)
    # tot= x.join(y)
    x.merge(y, how='inner', on='Patient_id')
    patient_size=np.unique(x['Patient_id'])
#     print(patient_size)
#     print(int(patient_size[2]))
    dirName=r'C:\Users\r0631\Documents\K-Fold\filled_data/'+str(fillmethod)+'/Fold'+str(NFold+1)
    if os.path.exists(dirName):
        shutil.rmtree(dirName)  # remove existing directory
    os.makedirs(dirName)
    for ind in range(len(patient_size)):
        #filename=r'output/Fold'+str(NFold+1)+'/p'+str(ind)+'.psv'
        id =int(patient_size[ind])
        filename=dirName+'/p'+str(id)+'.psv'
        patient = x.loc[x['Patient_id'] == id]
        patient.to_csv (filename, index = False, header=True, sep='|')
    print('Fold',NFold,'written away')



# # Functions for dataset evaluation

# Code for K-fold algorithm, Before this step the missing data has been filled,
#inputs
    ## model: Type of classifier,
    ## KforKFold: number of folds,
    ## KforKNN: no of neighbours to be use for KN
# Last modified JV - 10-3-21 20:20

def KFold_patient(model, KforKFold=10, KforKNN=5, fillmethod="",Uniq_ID=[]):
    start = time.time()  # time indicator for how long the Kfold func takes
    global X_train, X_test, y_train, y_test
    print("K Fold of ", KforKFold, " folds with KNN =", KforKNN)
    # initialisation of the array for storing the different intermediate results
    accuracy_model = []
    F1Score_model = []
    baseline_model = []
    auroc_model = []
    auprc_model = []
    physio_accuracy_model = []
    f_measure_model = []
    utility_score_model = []
    mean_train = 0
    positivepredictions = []
    # The unique id sets are created and shuffled in a fix manner, you can just use it here and no more any other manipulation
    idSets = np.array_split(Uniq_ID, KforKFold)  # divide the ids into K groups
   
    # This for loop is for Kfold, calculating the results for K times
    for i in range(KforKFold):
        start1 = time.time()
        clearAllDatasets()  # first clear all the datasets
        print("for the", i + 1, "th iteration", idSets[i])
        X_test, y_test = generateTestDataSet(idSets[i])
        for j in range(KforKFold):
            if j != i:
                X_train, y_train = generateTrainDataSet(idSets[j])
        # Now the train and test dataset is generated
        # we can begin to train the model wit the training set and evaaulate the performance with the test sett
        X_train = X_train.astype('float64')
        X_test = X_test.astype('float64')
        YTest_copy = y_test  # variable of joining the filled data (X) and Y (Train_output)
        patientID_ytest = y_test['Patient_id']
        y_train = y_train.drop('Patient_id', 1)
        y_test = y_test.drop('Patient_id', 1)
        y_train = y_train.astype('float64')
        y_test = y_test.astype('float64')
        # print('YTest',YTest_copy.head())

        # fill the missing data
        # if there is missing values, the data will be filled with one of the filling methods (KNN, mean, Linear, Forwards)
        if X_train.isnull().values.any() or X_test.isnull().values.any():
            print("X_train or X_test contains NaN values, KNN/mean is performed.")
            X_train_impute, X_test_impute, fillmethod = KNNfilling(X_train, X_test, KforKNN, fillmethod)
            # X_train_impute, X_test_impute,fillmethod = MeanFilling(X_train,X_test, fillmethod)
        else:
            print("X_train or X_test have all been filled ")
            X_train_impute = X_train
            X_test_impute = X_test

        # Scale the data
        #         scaler = preprocessing.StandardScaler()
        #         scaler.fit(X_train_impute)
        #         X_train_impute = scaler.transform(X_train_impute)
        #         X_test_impute = scaler.transform(X_train_impute)

        # fit the model and predict
        model.fit(X_train_impute, y_train)
        y_predicted = model.predict(X_test_impute)
        y_predicted_probobility = model.predict_proba(X_test_impute)

        # transfer the output and evalute it
        y_labels = y_test.astype(int).to_numpy()
        y_predicted = y_predicted.astype(int)
        y_predicted_probobility = y_predicted_probobility[:, 1].round(4)

        auroc, auprc, physio_accuracy, f_measure, utility_score = evaluate_performance(y_labels, y_predicted,y_predicted_probobility, patientID_ytest)
        #         print("\nauroc",round(auroc,4),"auprc",round(auprc,4),"util_accuracy",round(accuracy1,4))
        #         print("f_measure",round(f_measure,4),"utility_score",round(utility_score,4))
        auroc_model.append(round(auroc, 4))
        auprc_model.append(round(auprc, 4))
        physio_accuracy_model.append(round(physio_accuracy, 4))
        f_measure_model.append(round(f_measure, 4))
        utility_score_model.append(round(utility_score, 4))
        positivepredictions.append(np.sum(y_predicted))

        # WriteToFiles(X_train_impute,YTest_copy,i, fillmethod)

        # take down the results
        accuracy_model.append((accuracy_score(y_test, y_predicted, normalize=True) * 100).round(2))
        F1Score_model.append((f1_score(y_test, y_predicted) * 100).round(2))
        baseline_model.append(round((1 - float(y_test.mean())) * 100, 2))
        print("\ny_test size:", y_test.shape, '1´s in y_test', y_test.sum())
        # baseline_model.append(((1-y_test.mean())*100).round(2))
        print("The number of 1 (SepsisLabel) in this prediction: ", np.sum(y_predicted))
        #     print(accuracy_model)
        #     print(F1Score_model)
        end1 = time.time()
        print("Time spent in this KFold iteration", round(end1 - start1, 2), "sec.\n")
        print("******************************************************************")
    print('Accuracy model:', accuracy_model)
    print('F1_score model:', F1Score_model)
    print('Baseline model:', baseline_model)

    print("\nEvaluation parameters of the utiltiy evaluation function:")
    print('auroc of model:', auroc_model)
    print('auprc of model:', auprc_model)
    print('Utility accuracy of model:', accuracy_model)
    print('utility F1 of model:', f_measure_model)
    print('Utility score of model:', utility_score_model)

    KNN_UtilityScore_mean.append(np.mean(utility_score_model))
    KNN_UtilityScore_std.append(np.std(utility_score_model))

    KNN_F1Score_mean.append(np.mean(f_measure_model))
    KNN_F1Score_std.append(np.std(f_measure_model))

    KNN_auroc_mean.append(np.mean(auroc_model))
    KNN_auprc_mean.append(np.mean(auprc_model))

    KNN_accuracy_mean.append(np.mean(physio_accuracy_model))
    KNN_accuracy_std.append(np.std(physio_accuracy_model))

    KNN_positiveprediction_mean.append(np.mean(positivepredictions))
    KNN_baseline_mean.append(np.mean(baseline_model))
    totalTime = round(time.time() - start, 2)
    KNN_total_Time.append(totalTime)

    print("\nTotal Time spent in  KFold function", totalTime, "sec.\n")


# # Functions for Model Training

# # Functions for Results Evaluation + printing
# Fucntion for printing the results of the Kfold_patient & FindbestKforKNN Function
# Last modified JV - 10-3-21 20:20

def Print_DATA():
    print("KNN_UtilityScore_mean", KNN_UtilityScore_mean)
    print("KNN_UtilityScore_std", KNN_UtilityScore_std)
    print("KNN_F1Score_mean", KNN_F1Score_mean)
    print("KNN_F1Score_std", KNN_F1Score_std)
    print("KNN_auroc_mean", KNN_auroc_mean)
    print("KNN_auprc_mean", KNN_auprc_mean)
    print("KNN_accuracy_mean", KNN_accuracy_mean)
    print("KNN_accuracy_std", KNN_accuracy_std)
    print("KNN_positiveprediction_mean", KNN_positiveprediction_mean)
    print("KNN_baseline_mean", KNN_baseline_mean)

    print(len(KNN_accuracy_mean))
    print(len(KNN_accuracy_std))
    print(len(KNN_F1Score_mean))
    print(len(KNN_F1Score_std))
    print(len(KNN_positiveprediction_mean))

# Function for displaying the current results
# Last modified JV - 10-3-21 20:20
def displayCurrentResult(KforKNNstart, KforKNNend):
    print((KNN_accuracy_mean))
    print((KNN_accuracy_std))
    print((KNN_F1Score_mean))
    print((KNN_F1Score_std))
    print((KNN_positiveprediction_mean))
    print(len(KNN_accuracy_mean))
    print(len(KNN_accuracy_std))
    print(len(KNN_F1Score_mean))
    print(len(KNN_F1Score_std))
    print(len(KNN_positiveprediction_mean))
    plotKNNResultFigure(KforKNN, KNN_accuracy_mean, "Mean Accuracy vs K", xlabel='K', ylabel="Mean Accuracy")
    plotKNNResultFigure(KforKNN, KNN_accuracy_std, "Std of Accuracy vs K", xlabel='K', ylabel="Std of Accuracy")
    plotKNNResultFigure(KforKNN, KNN_F1Score_mean, "Mean F1 score vs K", xlabel='K', ylabel="Mean F1 score")
    plotKNNResultFigure(KforKNN, KNN_F1Score_std, "Std F1 score vs K", xlabel='K', ylabel="Std F1 score")
    plotKNNResultFigure(KforKNN, KNN_positiveprediction_mean, "Mean positive prediction vs K", xlabel='K',
                        ylabel="Mean positive prediction")



# Function for finding the best K "neighbours" which gives the best result
# Last modified JV - 10-3-21 20:45
def findBestKforKNN(model, KforKFold=10, KforKNNstart=1, KforKNNend=10, stepsize=5):
    if (KforKNNend <= 1):
        print("K must be a interger larger than 1")
        return
    KNN_reset()
    step = stepsize
    for i in range(KforKNNstart, KforKNNend + 1, step):
        print("KNN of K = ", i)
        KFold_patient(model, 10, i)
    print("Now all the training is finished.")
    plotKNNResultFigure(KforKNNstart, KforKNNend, KNN_UtilityScore_mean, "Mean Utility Score vs K", xlabel='K',
                        ylabel="mean Utility Score")
    plotKNNResultFigure(KforKNNstart, KforKNNend, KNN_UtilityScore_std, "std Utility Score vs K", xlabel='K',
                        ylabel="std Utility Score")
    plotKNNResultFigure(KforKNNstart, KforKNNend, KNN_F1Score_mean, "Mean F1 score vs K", xlabel='K',
                        ylabel="Mean F1 score")
    plotKNNResultFigure(KforKNNstart, KforKNNend, KNN_F1Score_std, "Std F1 score vs K", xlabel='K',
                        ylabel="Std F1 score")
    plotKNNResultFigure(KforKNNstart, KforKNNend, KNN_accuracy_mean, "Mean Accuracy vs K", xlabel='K',
                        ylabel="Mean Accuracy")

# # Function for getting the sepsis_score deviated from example code Phyionet
# Last modified JV - 10-3-21 20:45

def get_sepsis_score(X_test_impute):
    np.set_printoptions(precision=4)

    # X_mean, X_Std && c_mean, c_std are computed from full data A+B combined, for each column, attributes
    x_mean = np.array([
        84.581, 97.194, 36.977, 123.75, 82.4,
        63.831, 18.726, 32.958, -0.69, 24.075,
        0.555, 7.379, 41.022, 92.654, 260.223,
        23.915, 102.484, 7.558, 105.828, 1.511,
        1.836, 136.932, 2.647, 2.051, 3.544,
        4.136, 2.114, 8.29, 30.794, 10.431,
        41.231, 11.446, 287.386, 196.014])
    x_std = np.array([
        17.325, 2.937, 0.77, 23.232, 16.342,
        13.956, 5.098, 7.952, 4.294, 4.377,
        11.123, 0.075, 9.267, 10.893, 855.747,
        19.994, 120.123, 2.433, 5.88, 1.806,
        3.694, 51.311, 2.526, 0.398, 1.423, 0.642,
        4.311, 24.806, 5.492, 1.969, 26.218, 7.731,
        153.003, 103.635])

    c_mean = np.array([62.009, 0.559, 0.497, 0.503, -56.125, 26.995])
    c_std = np.array([16.386, 0.496, 0.5, 0.5, 162.257, 29.005])

    beta = np.array([
        0.1806, 0.0249, 0.2120, -0.0495, 0.0084,
        -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
        0.7476, 0.0323, 0.0305, -0.0251, 0.0330,
        0.1424, 0.0324, -0.1450, -0.0594, 0.0085,
        -0.0501, 0.0265, 0.0794, -0.0107, 0.0225,
        0.0040, 0.0799, -0.0287, 0.0531, -0.0728,
        0.0243, 0.1017, 0.0662, -0.0074, 0.0281,
        0.0078, 0.0593, -0.2046, -0.0167, 0.1239])
    rho = 7.8521
    nu = 1.0389

    num_rows = X_test_impute.shape[0]
    print('num_rows', num_rows)
    som = 0
    labelsom = 0
    row = num_rows
    scores = (np.empty([row]).astype('float64'))
    labels = np.empty([row])  # label parameter:output from get_sepsis_score
    for i in range(row):
        x = X_test_impute[i, 0:34]
        # print('x',x)
        c = X_test_impute[i, 34:40]
        # print('c',c)
        x_norm = np.nan_to_num((x - x_mean) / x_std)
        c_norm = np.nan_to_num((c - c_mean) / c_std)

        xstar = np.concatenate((x_norm, c_norm))
        exp_bx = np.exp(np.dot(xstar, beta))  # get a value from inner product of xstar and beta
        l_exp_bx = pow(4 / rho, nu) * exp_bx

        score = 1 - np.exp(-l_exp_bx)
        label = score > 0.45
        # print('score',score)
        scores[i] = score
        labels[i] = label
        som += score
        labelsom += label
    np.set_printoptions(precision=4)
    # print('scores:',scores,'\nLabels:', labels)
    print('Sum of scores', som)
    print('Sum of Labels', labelsom)
    return scores, labels

# Function for plotting the BestKforKNN results against k neighbours
#input
    # KValuestart: K value where X-axis starts from, KvalueEnd :K value untill where X-axis ranges
    # yValues: the values for the y-axis
    # title, xlabel, ylabel: values for the title of the graph and labels for x & y axis
# Last modified JV - 10-3-21 20:45

def plotKNNResultFigure(KValuestart,KvalueEnd,yValues,title,xlabel,ylabel):
    plt.figure(figsize=(10,6))
    plt.plot(range(KValuestart,KvalueEnd+1,5),yValues,color = 'blue',linestyle='dashed',
             marker='o',markerfacecolor='red', markersize=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print("Maximum ",ylabel,":-",max(yValues),"at K =", 1 + yValues.index(max(yValues)))

# # Functions for Results Evaluation
