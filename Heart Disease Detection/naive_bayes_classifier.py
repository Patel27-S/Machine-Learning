'''Implementation of Naive Bayes Classifier for a Heart Disease Classifying Problem.'''

# Step 1 :- Loading the data :-

# Step 2 :- Scaling the data after removing target.

# Step 3 :- Separating Training dataset and Testing dataset.

# Step 4 :- In training dataset, separating dataset corresponding to one 
# class with the dataset corresponding to the other class.

# Step 5 :- For each dataset of Step 4 (i.e. of different classes in training dataset),
        #       For a Single dataset, for each column, calculate the densities/probabilities for discrete data 
        #       and Gradient Distribution function for continous data.

# Step 6 :- The Algorithm is done and ready to be tested!!

# Step 7 :- You can form a function for calculating the class for future incoming data point's prediction. 
# i.e. The function which can calculate its probability for 1, probability for 0. Then, compares the probability
# and assigns the higher probability category to the datapoint. (i.e. If the Probability for it to be 1 > Probability for it to be 0, then
# classify it as/ label it as/ predict it as of the '1' Class, vice-versa otherwise.)


# Note :- The .csv file downloaded from Kaggle, had to be shuffled due to rows arranged corresponding to 
# the classes 1 and 0 on top and bottom respectively.

import pandas as pd
import numpy as np
import math


# Step 1 : Below

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
        # From second to the last row:
        for row in range(0, len(dataset_df)):
            dataset_df.iloc[row, column] = (dataset_df.iloc[row, column] - mean_col)/std_dev_col

    
    # Now, we will write the code below to randomly arrange the rows, so that,
    # If, it is already arranged in such a way that the rows corresponding to different 
    # classes, are bunched separately, then the problem can be solved.
    # We can easily thereafter divide the data into training-dataset and testing-dataset.
    # sample() is a method in Dataframe class, that helps randomly shuffling the rows.
    # 'frac=1' implies all the rows will be shuffled and returned(i.e. 100% fraction).
    dataset_df = dataset_df.sample(frac=1)

    return dataset_df



dataset_dataframe = load_scale_data("heart.csv")


# Defining a function for splitting the data into training and testing dataset :

def split_train_test(incoming_dataframe):
    '''
    This function will split the incoming dataset into 8:2 ratio,
    and return a tuple of dataframes for training_dataset and 
    testing_dataset correspondingly.
    '''
    # Number of rows in the training_dataset's dataframe,
    # 80% of the total number of rows.
    # Note :- We are taking the floor value to get an integral number.
    training_dataset_length = math.floor(0.8* len(incoming_dataframe))

    # Number of rows in the testing dataset's dataframe,
    # 20% of the total number of rows.
    testing_dataset_length = 1 - training_dataset_length

    # Below, is the training_test's dataframe:
    # It will have rows starting from the first row of the original dataframe till the row 
    # corresponding to training_dataset_length index.
    training_dataset_df = incoming_dataframe.iloc[0:training_dataset_length, : ]

    # Below, is the testing_test's dataframe:
    # It will have rows starting from the (training_dataset_length + 1)th index  
    # till the last row of the original dataframe.
    testing_dataset_df = incoming_dataframe.iloc[training_dataset_length+1 : len(incoming_dataframe)-1 , : ]

    return training_dataset_df, testing_dataset_df



training_dataset_df, testing_dataset_df = split_train_test(dataset_dataframe)



# The below function is customized just for two classes : Class0 and Class1.
# It is not a generalized one.
def split_training_dataset(training_dataset_df):
    '''
    This function splits the training dataset's dataframe by class value and returns them as a tuple.
    '''

    training_dataset_class0 = training_dataset_df[training_dataset_df['target']==0]
    training_dataset_class1 = training_dataset_df[training_dataset_df['target']==1]

    return training_dataset_class0, training_dataset_class1


# Training the Model :-

# Calculating the Prior Probabilities : (i.e. P(Class = 0) and P(Class = 1))
# For, an incoming 'to be predicted' datapoint, the prior probabilty will be 
# calculated with the below lines.
total_rows = float(training_dataset_df.shape[0])
class1_count = training_dataset_df[training_dataset_df['target'] == 1].shape[0]
class0_count = training_dataset_df[training_dataset_df['target'] == 0].shape[0]
prior_class1_prob = class1_count / total_rows
prior_class0_prob = class0_count / total_rows


# Calculating the Continous variable probability : 
# The value of 'x' below would be known from the datapoint to be predicted(If it belongs to 
# Class0 or Clas1) for testing the model. 
def calculate_continous_variable_prob(mean, std, x):
    '''
    This function returns a guassian distribution function(i.e. Probability) 
    value for a given 'x', for a continous attribute of the dataset.
    '''
    exp = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exp



# Calculating the Discrete variable probability :
def calculate_discrete_variable_prob(attr, val):
    y_count = dataset_dataframe[(dataset_dataframe['target'] == 1) & (dataset_dataframe[attr] == val)].shape[0]
    n_count = dataset_dataframe[(dataset_dataframe['target'] == 0) & (dataset_dataframe[attr] == val)].shape[0]
    return y_count / class1_count, n_count / class0_count




# Below, are the dictionaries with attribute:(std,mean) as key-value pairs for continous 
# features in training_dataset_class1 and training_dataset_class0 respectively.
continuous_attr_class1_dict = dict()
continuous_attr_class0_dict = dict()
# Below are the features with Continous values in the dataset.
continuous_value_attributes = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for attribute in continuous_value_attributes:
    # For each continous attribute, differentiating the class0 and class1 part,
    # and thereafter computing mean and std_dev for each one.
    # Storing the values in dictionaries.
    np_arr_class1 = np.array(dataset_dataframe[dataset_dataframe['target'] == 1][attribute])
    np_arr_class0 = np.array(dataset_dataframe[dataset_dataframe['target'] == 0][attribute])
    continuous_attr_class1_dict[attribute] = (np.mean(np_arr_class1), np.std(np_arr_class1))
    continuous_attr_class0_dict[attribute] = (np.mean(np_arr_class0), np.std(np_arr_class0))



# Storing the last column of testing_df, which is 'target'.
class_values = testing_dataset_df['target']
testing_dataset_df = testing_dataset_df.iloc[:, :-1]


# Testing the Naive Bayes Classifier Model :-

correct_predictions = 0
wrong_predictions = 0
# Below, statement is the number of rows in the testing_dataset_df.
test_records = testing_dataset_df.shape[0]

# Creating a list below to store/track the output obtained by prediction
# with the help of the Model for all the rows of testing_df.
predicted_output = list()

# Predicting the Class(i.e. 0 or 1) for each record/row in the 
# testing_dataset_df :-
for i in range(test_records):

    # Below, statement denotes a row of the dataframe. 
    # (i.e. testing_dataset_df in our case.)
    row = testing_dataset_df.iloc[i]
     
    class1_prob = 1
    class0_prob = 1

    for attribute, value in row.iteritems():
        # For a single row of testing_dataset_df, it is getting
        # checked if an attribute is continous valued or discrete valued.
        # Thereafter, the probabilties are getting calculated accordingly,
        # for class0 and class1.
        if attribute in continuous_attr_class1_dict:
            class1_mean = continuous_attr_class1_dict.get(attribute)[0]
            class1_std = continuous_attr_class1_dict.get(attribute)[1]
            class1_prob *= calculate_continous_variable_prob(class1_mean, class1_std, value)
            class0_mean = continuous_attr_class0_dict.get(attribute)[0]
            class0_std = continuous_attr_class0_dict.get(attribute)[1]
            class0_prob *= calculate_continous_variable_prob(class0_mean, class0_std, value)
        else:
            discrete_probabilities = calculate_discrete_variable_prob(attribute, value)
            class1_prob *= discrete_probabilities[0]
            class0_prob *= discrete_probabilities[1]
    # Multiplying by prior probability
    class1_prob *= prior_class1_prob
    class0_prob *= prior_class0_prob
    
    # Assigninging the Class to the datapoint :-
    if class1_prob > class0_prob:
        predicted_class = 1
    else:
        predicted_class = 0
    
    # Capturing the predicted class, predicted by the Model :
    predicted_output.append(predicted_class)

    

    # Checking for the correctness of prediction :

    actual_class = class_values.values[i]

    if predicted_class == actual_class:
        correct_predictions += 1
    else:
        wrong_predictions += 1


# F1-Score, Confusion Matrix :-

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0


for i in range(0,len(list(class_values))):
    if list(class_values)[i] == predicted_output[i] and list(class_values)[i] == 1:
       true_positive += 1


for i in range(0,len(list(class_values))):
    if list(class_values)[i] == predicted_output[i] and list(class_values)[i] == 0:
       true_negative += 1


for i in range(0,len(list(class_values))):
    if list(class_values)[i] == 0 and predicted_output[i] == 1:
       false_positive += 1


for i in range(0,len(list(class_values))):
    if list(class_values)[i] == 1 and predicted_output[i] == 0:
       false_negative += 1


accuracy = (true_positive + true_negative)/ (true_positive + true_negative + false_positive + false_negative) 

recall_score = true_positive / (true_positive + false_negative)

precision = true_positive/(true_positive+false_positive)

f1_score =  2*(precision * recall_score)/(precision + recall_score)

dict_conf = {'Actual_Output' : list(class_values), 'Predicted_Output': predicted_output}
df = pd.DataFrame(dict_conf, columns=['Actual_Output','Predicted_Output'])
confusion_matrix = pd.crosstab(df['Actual_Output'], df['Predicted_Output'], rownames=['Actual_Output'], colnames=['Predicted_Output'])

print(confusion_matrix)

print('F1_score :-', f1_score)