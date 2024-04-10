import numpy as np
import pandas as pd
import graphviz
import time
import uuid

# Using graphviz to visualize my decision tree
dot = graphviz.Digraph(comment="Example", format='png', engine='dot')

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2): 
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    def fit(self,X):
        # grow tree
        self.root = self.grow_tree(X)
    def grow_tree(self,X,depth=0,edge_label='root',edge_side='left'):
        #n_samples is row count
        #n_features is column count
        #n_labels is good and bad count if it is 1 it means there is a leaf node
        n_samples, n_features = X.shape
        n_labels = len(np.unique(X['class']))

        measure_of_goodness = self.calculate_measure_of_goodness(X)
        # grow_tree function is recursive
        # I want to make sure I have unique id
        # Therefore I used uuid and time
        unique_id = str(uuid.uuid4()) 
        node_name = edge_label + unique_id + str(int(time.time()))

        if(depth >= self.max_depth) or n_labels == 1 or n_samples <= self.min_samples_split or n_features<=1:
            leaf_value = self.most_common_label(X)
            node_label = leaf_value
            dot.node(node_name,node_label,{'color': 'aquamarine', 'shape' : 'box'})
            if edge_side =='left':
                dot.edge(edge_label,node_name,label='YES')
            else:
                dot.edge(edge_label,node_name,label='NO')
            return Node(value=leaf_value)
        else:
            node_label = measure_of_goodness['feature_index'] + "==" + measure_of_goodness['threshold']
            dot.node(node_name,node_label,{'color': 'aquamarine'})
            if(edge_label!='root'):
                if(edge_side=='left'):
                    dot.edge(edge_label,node_name,label='YES')
                else:
                    dot.edge(edge_label,node_name,label='NO')

            left_subtree = self.grow_tree(measure_of_goodness['dataset_left'],depth+1,edge_label=node_name,edge_side='left')
            right_subtree = self.grow_tree(measure_of_goodness['dataset_right'],depth+1,edge_label=node_name,edge_side='right')
            return Node(measure_of_goodness['feature_index'],measure_of_goodness['threshold'],left_subtree,right_subtree)
    def calculate_measure_of_goodness(self,data):
        max_goodness={}
        data_label = data.drop(data.columns[-1],axis=1)
        max_goodness_value = 0
        #FIND EACH COLUMN's UNIQUE VALUES
        # EX: unique_values_dict['A8'] will return ['f','t',..]
        unique_values_dict = {}
        for column in data_label.columns:
            unique_values_dict[column] = data_label[column].unique()
        # key will be class name 
        # value will be unique elements in the class    
        for key, value in unique_values_dict.items():
            # EX: value=['f','t'] ==> value will go through value[0] to value[last]
            for unique_value in value:
                #Initialize the variables to 0 in each iteration
                Pleft = 0
                Pright = 0
                Pleftgood = 0
                Pleftbad = 0
                Prightgood = 0
                Prightbad = 0
                Restdatapoint = 0

                number_of_unique_value_dict = {}
                
                #DEFINE ROW COUNT TO USE PROBABILITY CALCULATION
                row_count = data.shape[0]
                #DEFINE TOTAL GOOD COUNT TO USE PROBABILITY CALCULATION
                total_good_count = len(data[data['class'] == 'good'])
                #Number of unique_value in a selected column which is key
                number_of_unique_value_dict[unique_value] = ((data[key] == unique_value).sum())
                dict_value = number_of_unique_value_dict[unique_value]
                if dict_value == row_count:
                    current_measure_of_goodness = 0
                else:
                    Restdatapoint = row_count - dict_value
                    # Pleft = value count in dataset/total number of data
                    # Pright = 1 - Pleft
                    Pleft = (dict_value/row_count)
                    Pright = 1-Pleft

                    # Given attributes will be used in epsilon calculation part
                    # Good and bad counts can be done with filtering
                    
                    left_good_count = len(data[(data[key] == unique_value) & (data['class']== 'good')])
                    left_bad_count = dict_value - left_good_count
                    right_good_count = total_good_count - left_good_count
                    right_bad_count = Restdatapoint - right_good_count

                    Pleftgood = (left_good_count / dict_value)
                    Pleftbad =  (left_bad_count/ dict_value)
                    Prightgood = (right_good_count/Restdatapoint)
                    Prightbad = (right_bad_count/Restdatapoint)
                    current_measure_of_goodness = (2 * Pleft * Pright) * (abs(Pleftgood-Prightgood) + abs(Pleftbad-Prightbad))
                if(current_measure_of_goodness>=max_goodness_value):
                    max_goodness['feature_index'] = key
                    max_goodness['threshold'] = unique_value
                    #left_subtree will have all rows which key column has unique_value and then drop the key column
                    max_goodness['dataset_left'] = data[data[key] == unique_value].drop(key, axis=1)
                    #delete 'unique_value' in a selected column for right_subtree
                    if len(np.unique(data[key]))==1:
                        max_goodness['dataset_right'] = data[data[key] != unique_value].drop(key,axis=1)
                    else:    
                        max_goodness['dataset_right'] = data[data[key] != unique_value]
                    max_goodness_value = current_measure_of_goodness
        return max_goodness      
    def most_common_label(self,X):
        # Handle the case where the dataset is empty
        # It is a common scenario
        if X.empty:
            return "good"
        # If dataset is not empty
        # It will return most common label
        else:
            return X['class'].value_counts().idxmax()  
    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root

        if node.value is not None:
            print(indent + "Leaf Node: " + str(node.value))
        else:
            print(indent + "Decision Node:")
            print(indent + "  Feature Index:", node.feature_index)
            print(indent + "  Threshold:", node.threshold)
            print(indent + "  Left Subtree:")
            self.print_tree(node.left, indent + "    ")
            print(indent + "  Right Subtree:")
            self.print_tree(node.right, indent + "    ")
    def make_prediction(self,X):
        #X will have 10 columns and 200 rows
        #Take each row and sen it to recursive_prediction func
        X = X.drop(X.columns[-1],axis=1)
        predictions=[]
        for index,row in X.iterrows():
            prediction = self.recursive_prediction(row, self.root)
            predictions.append(prediction)
        return predictions
    def recursive_prediction(self, X, tree):

        if tree.value is not None: 
            return tree.value
    
        feature_ind = tree.feature_index
        threshold_val = tree.threshold
        if X[feature_ind] == threshold_val:
            return self.recursive_prediction(X,tree.left)
        else:
            return self.recursive_prediction(X,tree.right) 


def equal_width_partition(datalabel):
    #there will be 3 binns
    #ALGORITHM ==> (MAX-MIN)/numberOfBins
    #FUNCTION RETURN BINNED COLUMN
    numberOfBins=3
    group_names= ['a','b','c']
    minValue = datalabel.min()
    maxValue = datalabel.max()

    width = (maxValue-minValue)/numberOfBins
    bin_edges = [minValue + i * width for i in range(numberOfBins + 1)]
    datalabel = pd.cut(datalabel, bins=bin_edges, labels=group_names, include_lowest=True)
    return datalabel
def dataBinning(data):
    #A2 and A7 columns are not categorical so we need to make them categorical    
    data['A2'] = equal_width_partition(data['A2'])
    data['A7'] = equal_width_partition(data['A7'])
    return data
def performance_scores(test_class,prediction_class):
    performance_score = {}
    #accuracy = (TP+TN)/#of instances
    #TPrate = TP/(TP+FN)
    #TNrate = TN/(TN+FP)
    #Precision = TP/(TP+FP)
    #FScore = 2 *(Precision * TPrate) / (Precision + TPrate)
    predicted_positive = prediction_class.count('good')
    predicted_negative = prediction_class.count('bad')
    TotalNumberofTP = np.sum((test_class == prediction_class) & (test_class == 'good'))
    TotalNumberofTN = np.sum((test_class == prediction_class) & (test_class == 'bad'))
    TotalNumberOfFP = predicted_positive - TotalNumberofTP
    TotalNumberOfFN = predicted_negative - TotalNumberofTN
    performance_score['accuracy_score'] = np.sum(test_class == prediction_class) / len(test_class)
    performance_score['TPrate'] = TotalNumberofTP / (TotalNumberofTP+TotalNumberOfFN)
    performance_score['TNRate'] = TotalNumberofTN / (TotalNumberofTN+TotalNumberOfFP)
    performance_score['Precision'] = TotalNumberofTP / (TotalNumberofTP+TotalNumberOfFP)
    
    if performance_score['Precision']==0:
        performance_score['FScore'] = 0
    else:
        performance_score['FScore'] = 2 * performance_score['Precision'] * performance_score['TPrate'] / (performance_score['Precision']+performance_score['TPrate'])
    
    performance_score['TotalNumberOfTP'] = TotalNumberofTP
    performance_score['TotalNumberOfTN'] = TotalNumberofTN
    return performance_score


#DEFINE COLUMN NAMES AND READ CSV FILE
col_names = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','class']
data = pd.read_csv("CART/dataset/trainSet.csv",skiprows=1,header=None,names=col_names)

#MAKE BINNING FOR CONTINUES VALUES
dataBinning(data)

#CREATE A DECISION TREE AND FIT THE DATA
classifier = DecisionTreeClassifier(min_samples_split=2,max_depth=15)
classifier.fit(data)
classifier.print_tree()

# Render the graph to a file
dot.render('CART/output/graph_of_tree')

test_data = pd.read_csv("CART/dataset/testSet.csv",skiprows=1,header=None,names=col_names)

#MAKE BINNING FOR CONTINUES VALUES
dataBinning(test_data)

#Test decision tree with train data and test data
train_test_class = data['class']
test_class = test_data['class']

train_data_predictions = classifier.make_prediction(data)
train_performance_score=performance_scores(train_test_class,train_data_predictions)
print("Train Set Results:")
for key in train_performance_score:
    print(key,"==>",train_performance_score[key])

test_predictions = classifier.make_prediction(test_data)
test_performance_score=performance_scores(test_class,test_predictions)
print("\n\n\nTest Set Results:")
for key in test_performance_score:
    print(key,"==>",test_performance_score[key])

#           RANDOM FOREST              #

# We have same data
# I will create random subspaces of this data
# Each subspace will have 6 features as default subspacing method
# Then each subspace will fit a DecisionTreeClassifier class
# Then classifier will be tested with testSet data
# Lastly, performance score will be calculated with each classifier's performance score

import itertools

# Select 6 columns without selecting 'class' column
# Total 84 different combination
num_columns = 6
data_to_select_columns = data.drop(data.columns[-1],axis=1)
all_column_combinations = list(itertools.combinations(data_to_select_columns.columns, num_columns))

# Creating a classifier
random_forest_classifier = DecisionTreeClassifier(min_samples_split=2,max_depth=8)

# Prediction list for calculating mean performance scores
predictions = []

# Creating a new DataFrame with randomly selected columns
for column_combination in all_column_combinations:
    # Concatenate 'class' to column combination 
    column_combination_w_class = column_combination + ('class',)
    # Train data dataframe with selected columns
    data_train= data[list(column_combination_w_class)]
    # Fit dataframe to the classifier
    random_forest_classifier.fit(data_train)
    # Prepare test data
    test_data_w_columns = test_data[list(column_combination_w_class)]
    # Test data values dataframe
    test_data_class = test_data_w_columns['class']

    test_prediction = random_forest_classifier.make_prediction(test_data_w_columns)
    performance_score = performance_scores(test_data_class,test_prediction)
    predictions.append(performance_score)   


total_accuracy = 0
total_tprate = 0
total_tnrate = 0
total_precision =0
total_fscore =0
total_numberof_tp =0
total_numberof_tn =0


# Calculate total performance scores
for prediction_dict in predictions:
    total_accuracy += prediction_dict['accuracy_score'] 
    total_tprate += prediction_dict['TPrate']
    total_tnrate += prediction_dict['TNRate']
    total_precision += prediction_dict['Precision']
    total_fscore += prediction_dict['FScore']
    total_numberof_tp += prediction_dict['TotalNumberOfTP']
    total_numberof_tn += prediction_dict['TotalNumberOfTN']
    

# print average performance scores
print("\n\n\nRandom Forest Applied Results:")
print("Accuary ==>",total_accuracy/len(predictions))
print("TPRate ==>",total_tprate/len(predictions))
print("TNRate ==>",total_tnrate/len(predictions))
print("Precision ==>",total_precision/len(predictions))
print("FScore ==>",total_fscore/len(predictions))
print("TotalNumberOfTP ==>",total_numberof_tp/len(predictions))
print("TotalNumberOfTN ==>",total_numberof_tn/len(predictions))


