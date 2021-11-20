'''Implementation of KNN classifier for predicting the label for incoming datapoints.'''

import pandas as pd
import math


def load_scale_data(type_of_file):
    '''
    This function will convert a .csv file into a 
    Pandas Dataframe and Standarize the data.
    '''
    # Converting the .csv to pandas.DataFrame()
    dataset_df = pd.read_csv(type_of_file)

    # For each column we have to Scale the data,
    # (i.e. calculate the mean, std_dev of the column and then 
    # update each value of the column with (each_value-mean_col)/std_dev.  
    # ) :
    # From first to the second_last column: (Since, The last column would be 'target')
    for column in range(0, len(dataset_df.columns)-1):
        std_dev_col = dataset_df.iloc[0:, column].std(ddof=0)
        mean_col = dataset_df.iloc[0:, column].mean()
        # From first to the last row:
        for row in range(0, len(dataset_df)):
            dataset_df.iloc[row, column] = (dataset_df.iloc[row, column] - mean_col)/std_dev_col

    return dataset_df





def split_train_valid_test(incoming_dataframe):
    '''
    This function splits the dataframe into three parts for
    training, validation and testing purpose(6:2:2 -> ratio). It returns them 
    in the tuple as a return value in-order(i.e. training, validation, testing).
    '''
    
    # Lengths or the Number of rows for each of the training, validation
    # and testing dataset.
    training_dataset_length = int(0.6 * len(incoming_dataframe))

    validation_dataset_length = int(0.2 * len(incoming_dataframe))
    
    testing_dataset_length = int(0.2 * len(incoming_dataframe))

    # Bisecting the Original dataframe into three parts and eventually 
    # returning them.

    training_dataset_df = incoming_dataframe.iloc[: training_dataset_length ,:]

    validation_dataset_df = incoming_dataframe.iloc[training_dataset_length: training_dataset_length + \
        validation_dataset_length,:]

    testing_dataset_df = incoming_dataframe.iloc[training_dataset_length + validation_dataset_length :,:]


    return training_dataset_df, validation_dataset_df, testing_dataset_df





def df_to_list(df):
    '''
    Function to convert a dataframe into a list of lists.
    '''
    return df.values.tolist()



def cross_validation(list_k, validation_dataset_df, training_dataset_df):
    '''

    list_k = A list of values of 'k' to predict the classes for validation_dataset_df
             and finding out the best value of 'k' for prediction.

    validation_dataset_df = Dataset to calculate the classes of prediction for.

    training_dataset_df = Dataset used for predicting the classes for datapoints of 
             validation_dataset_df.

    This function returns the best_value of 'k' out of the ones mentioned in the list passed,
    as per the highest value of f1_score corresponding to a value of 'k'.
    '''

    # validation_dataset_list is a list of lists.
    validation_dataset_list = df_to_list(validation_dataset_df.iloc[:, :-1])

    # training_dataset_list is a list of lists.
    training_dataset_list = df_to_list(training_dataset_df.iloc[:, :-1])

    # Forming a list of actual_class values corresponding to each datapoint
    # in validation_dataset_df, helpful for further calculations :
    actual_class_list = list(validation_dataset_df.iloc[:, -1])

    # To store the mapping of each 'k' and their corresponding f1_score 
    # in the cross validation process.
    f1_score_dict = dict()

    for k in list_k:
        
        predicted_list = list()
        correct_prediction = 0
        wrong_prediction = 0

        # For each row in validation_dataset_list, for each row
        # we need to calculate its distance with all the rows(i.e. datapoints)
        # of training_dataset_list.

        for each_row in validation_dataset_list:
            
            # Dictionary to map the distances with index_training_dataset_row.
            d = dict()
            index_training_dataset_row = 0
            validate_dataset_target_index = 0

            for each_row_ in training_dataset_list:

                distance = math.sqrt( (each_row[0]-each_row_[0])**2 + \
                    (each_row[1]-each_row_[1])**2 + (each_row[2]-each_row_[2])**2 +\
                    (each_row[3]-each_row_[3])**2 + (each_row[4]-each_row_[4])**2 +\
                    (each_row[5]-each_row_[5])**2 + (each_row[6]-each_row_[6])**2 +\
                    (each_row[7]-each_row_[7])**2 + (each_row[8]-each_row_[8])**2 +\
                    (each_row[9]-each_row_[9])**2 + (each_row[10]-each_row_[10])**2 +\
                    (each_row[11]-each_row_[11])**2 + (each_row[12]-each_row_[12])**2)

                d[index_training_dataset_row] = distance
                index_training_dataset_row += 1
            
            # Sorting the dictionary 'd' by its values: In
            d = dict(sorted(d.items(), key = lambda item : item[1]))
            
            # Listing the 'k' nearest neighbor's(i.e. datapoints) index in the 
            # training_dataset_df : 
            nearest_k_neighbors_index = [key for key, value in d.items()][:k]
            
            # The list of classes for the 'k' nearest neighbors:
            list_of_target = [ i for i in training_dataset_df.iloc[nearest_k_neighbors_index, -1]]
            
            # Predicting the class for a single datapoint of validation_dataset_df:
            predicted_class = 1 if list_of_target.count(1) > list_of_target.count(0) else 0
            
            predicted_list.append(predicted_class)
            
            # The actual class to which the datapoint of validation_dataset_df belongs:
            actual_class = actual_class_list[validate_dataset_target_index]

            validate_dataset_target_index += 1

            if predicted_class == actual_class:
                correct_prediction += 1
            else:
                wrong_prediction += 1
        
        

        # For each value of 'k', finding out TP, FP, TN, FN 
        # in order to calculate precision and recall's value,
        # for f1_score:
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        
        
        for i in range(0,len(actual_class_list)):
            if actual_class_list[i] == predicted_list[i] and predicted_list[i] == 1:
                true_positive += 1

        for i in range(0,len(actual_class_list)):
            if actual_class_list[i] == predicted_list[i] and actual_class_list[i] == 0:
                true_negative += 1
        
        for i in range(0,len(actual_class_list)):
            if actual_class_list[i] == 0 and predicted_list[i] == 1:
                false_positive += 1
        
        for i in range(0,len(actual_class_list)):
            if actual_class_list[i] == 1 and predicted_list[i] == 0:
                false_negative += 1

        
        # Precision :
        precision = true_positive / (true_positive + false_positive)
        
        # Recall :
        recall = true_positive / (true_positive + false_negative)
        
        # f1_score :
        f1_score =  2*(precision * recall)/(precision + recall)
        
        # Appending the f1_score values for different 'k' 
        # in the below list :
        f1_score_dict[k] = f1_score
        
        # Pulling out the 'k' with maximum value of f1_score from
        # f1_score_dict : In
        best_k = max(zip(f1_score_dict.values(), f1_score_dict.keys()))[1]

    return best_k



def testing(best_k, testing_dataset_df, training_dataset_df):
    '''
    best_k : value obtained from cross_validation()

    testing_dataset_df : testing_dataframe for prediction the classes of 
        datapoints of.

    training_dataset_df : training_dataframe used for predicting the class
       for testing_dataframe's datapoints.
    
    This function returns the 'f1_score' and 'confusion_matrix', for depicting the 
    performance of the KNN algorithm.
    '''

    # testing_dataset_list is a list of lists.
    testing_dataset_list = df_to_list(testing_dataset_df.iloc[:, :-1])

    # training_dataset_list is a list of lists.
    training_dataset_list = df_to_list(training_dataset_df.iloc[:, :-1])

    # Forming a list of actual_class values corresponding to each datapoint
    # in testing_dataset_df, helpful for further calculations :
    actual_class_list = list(testing_dataset_df.iloc[:, -1])


    
    predicted_list = list()
    correct_prediction = 0
    wrong_prediction = 0

    # For each row in validation_dataset_list, for each row
    # we need to calculate its distance with all the rows(i.e. datapoints)
    # of training_dataset_list.

    for each_row in testing_dataset_list:
        # In the below code, For each datapoint in testing_dataset_df(or 
        # testing_dataset_list), we are predicting its class. 
        # By the end of the for loop, we would have had a List of prediction,
        # (i.e. prediction_list), Number of Wrong predictions(i.e. wrong_predictions),
        # and Number of Correct predictions (i.e correct_predictions).
        
        # Dictionary to map the distances of a datapoint with all the datapoints of 
        # training_dataset :
        # The {key, value} pairs would be = 
        # {Index corresponding to the datapoint in training_dataset_df : Distance of the testing_datapoint 
        # with 'that' datapoint of the training_dataset.}
        d = dict()
        index_training_dataset_row = 0

        # This will further be used for computing the actual_class value
        # for a datapoint in the testing_dataset_df.
        testing_dataset_target_index = 0

        for each_row_ in training_dataset_list:

            distance = math.sqrt( (each_row[0]-each_row_[0])**2 + \
                (each_row[1]-each_row_[1])**2 + (each_row[2]-each_row_[2])**2 +\
                (each_row[3]-each_row_[3])**2 + (each_row[4]-each_row_[4])**2 +\
                (each_row[5]-each_row_[5])**2 + (each_row[6]-each_row_[6])**2 +\
                (each_row[7]-each_row_[7])**2 + (each_row[8]-each_row_[8])**2 +\
                (each_row[9]-each_row_[9])**2 + (each_row[10]-each_row_[10])**2 +\
                (each_row[11]-each_row_[11])**2 + (each_row[12]-each_row_[12])**2)

            d[index_training_dataset_row] = distance
            # The below incrementing is done to map the distance to the 
            # correct value of index of training_dataset_df. 
            # Or, else all the distance values would be mapped to the row with Index '0'
            # of the training_dataset.
            index_training_dataset_row += 1
        
        # Sorting the dictionary 'd' by its values: In
        d = dict(sorted(d.items(), key = lambda item : item[1]))
        
        # Listing the 'best_k' nearest neighbor's(i.e. datapoints) index in the 
        # training_dataset_df : 
        nearest_k_neighbors_index = [key for key, value in d.items()][:best_k]
        
        # The list of classes for the 'best_k'(i.e. k) nearest neighbors:
        list_of_target = [ i for i in training_dataset_df.iloc[nearest_k_neighbors_index, -1]]
        
        # Predicting the class for a single datapoint of testing_dataset_df:
        predicted_class = 1 if list_of_target.count(1) > list_of_target.count(0) else 0
        
        # Appending the predicted class to predicted_list.
        predicted_list.append(predicted_class)
        
        # The actual class to which the datapoint of testing_dataset_df belongs:
        actual_class = actual_class_list[testing_dataset_target_index]
        
        # INCrementing the variable on each iteration to find out the actual_class of 
        # a testing_dataset's datapoint. 
        # It is used as an index.
        testing_dataset_target_index += 1
        
        # INCrementing either correct_prediction or wrong_prediction, 
        # depending upon a correct or wrong prediction of a datapoint.
        if predicted_class == actual_class:
            correct_prediction += 1
        else:
            wrong_prediction += 1
    
    

    # For the value of 'best_k', finding out TP, FP, TN, FN 
    # in order to calculate precision and recall's value,
    # for f1_score:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    
    
    for i in range(0,len(actual_class_list)):
        if actual_class_list[i] == predicted_list[i] and predicted_list[i] == 1:
            true_positive += 1

    for i in range(0,len(actual_class_list)):
        if actual_class_list[i] == predicted_list[i] and actual_class_list[i] == 0:
            true_negative += 1
    
    for i in range(0,len(actual_class_list)):
        if actual_class_list[i] == 0 and predicted_list[i] == 1:
            false_positive += 1
    
    for i in range(0,len(actual_class_list)):
        if actual_class_list[i] == 1 and predicted_list[i] == 0:
            false_negative += 1

    
    # Precision :
    precision = true_positive / (true_positive + false_positive)
    
    # Recall :
    recall = true_positive / (true_positive + false_negative)
    
    # f1_score :
    f1_score =  2*(precision * recall)/(precision + recall)
    
    dict_conf = {'Actual_Output' : actual_class_list, 'Predicted_Output': predicted_list}
    df = pd.DataFrame(dict_conf, columns=['Actual_Output','Predicted_Output'])
    confusion_matrix = pd.crosstab(df['Actual_Output'], df['Predicted_Output'], rownames=['Actual_Output'], colnames=['Predicted_Output'])

    return f1_score, confusion_matrix




# IMPlementing the above functions :-

# Loading the dataset, post standarizing it :-
dataset_dataframe = load_scale_data('heart.csv')


# Splitting the dataset into three parts :
# Splitting the dataset using the function 'split_train_valid_test()' :
training_dataset_df, validation_dataset_df, testing_dataset_df= split_train_valid_test(dataset_dataframe)


# Determining the best value of 'k' for predicting classes for validation_dataset :
best_value_k = cross_validation([3,7,13,17,27], validation_dataset_df, training_dataset_df)

# Using the best_value_k's value in the testing() function, and obtaining the
# f1_score and confusion_matrix :
f1_score, confusion_matrix = testing(best_value_k, testing_dataset_df, training_dataset_df)


# Printing the values of f1_score and confusion_matrix.
print('f1_score in percentage form :- ', f1_score * 100)
print(confusion_matrix)
