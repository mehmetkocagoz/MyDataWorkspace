import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Multi-Layer Perceptron must have at least 1 or more hidden layer
col_names = ['theta1','theta2','theta3','theta4','theta5','theta6','thetad1','thetad2','thetad3','thetad4','thetad5','thetad6','tau1','tau2','tau3','tau4','tau5','dm1','dm2','dm3','dm4','dm5','da1','da2','da3','da4','da5','db1','db2','db3','db4','db5','ANGLE-ACC-ARM']

train_data = pd.read_csv("MLP/dataset/arm_angle_TRAIN.csv",skiprows=1,header=None,names=col_names)

X_test_data = train_data['ANGLE-ACC-ARM']
X_train_data = train_data.drop(train_data.columns[-1],axis=1)


# set the random_state to the same value to make the training process more reproducible(taking same results in each run)
# hidden_layer_sizes =(x,y,z), x,y,z means there is 3 hidden layers and number of neurons are x,y,z respectively
# In default hidden_layer_sizes(100), 1 hidden layer and 100 nuerons, I changed it to (80,80,80) and it improved the metrics slightly(approximately 0.25)
# early_stopping improved the metrics
# I tried different learning_rates (adaptive,constant..), it did not change any metrics - learning_rate: schedule for weight updates
# In default alpha=0.0001, I changed it to 0.0003 and it improved the metrics - alpha: adds a penalty term to the weights to prevent overfitting.
# In default learning_rate_init=0.001, I changed it to 0.003 and it improved the metrics - learning_rate_init: initial learning rate used by the optimizer during training.
# In default momentum = 0.9, I changed it and fit the model with different momentum values, but it did not change any metrics
# In default solver(gradient_method) = adam, sgd method requires more iteration for my specific Regressor and I achieved best results with adam
# In default batch_size=200, I tried different batch sizes, higher batch size has decreased R^2 and icreased other metrics which is not good, 
# lower batch size has decreased the R^2 but it also decreased other metrics in test results which is good, therefore I choose batch_size=30
# In default activation = relu, I tried other activation functions such as logistic, identity but it did not improved metrics
# In default valdation_fraction=0.1, changing it made metrics worse
# In default tol=0.0001, I changed it to 0.0005 and it improved metrics slightly(approximately 0.005)

regressor = MLPRegressor(random_state=2,hidden_layer_sizes=(80,80,80),max_iter=1000,early_stopping=True,alpha=0.0003,learning_rate_init=0.003,batch_size=30,tol=0.0005).fit(X_train_data,X_test_data)

predicted_values = regressor.predict(X_train_data)

score = regressor.score(X_train_data,X_test_data)
mse = mean_squared_error(X_test_data, predicted_values)
mae = mean_absolute_error(X_test_data, predicted_values)
rmse = np.sqrt(mse)

print("Train Results")
print("MAE: ",mae)
print("MSE: ",mse)
print("RMSE: ",rmse)
print("R^2 (coefficient of determination): ",score)


test_data = pd.read_csv("MLP/dataset/arm_angle_TEST.csv",skiprows=1,header=None,names=col_names)

y_test_data = test_data['ANGLE-ACC-ARM']
y_train_data = test_data.drop(test_data.columns[-1],axis=1)

predicted_values_for_test_data = regressor.predict(y_train_data)

score_for_test_data = regressor.score(y_train_data,y_test_data)
mse_for_test_data = mean_squared_error(y_test_data,predicted_values_for_test_data)
mae_for_test_data = mean_absolute_error(y_test_data,predicted_values_for_test_data)
rmse_for_test_data = np.sqrt(mse_for_test_data)

print("Test Results:")
print("MAE: ",mae_for_test_data)
print("MSE: ",mse_for_test_data)
print("RMSE: ",rmse_for_test_data)
print("R^2 (coefficient of determination): ",score_for_test_data)


# weights
# parameters that connect the nodes
for i, coef in enumerate(regressor.coefs_):
    print(f"Layer {i+1}: {coef.shape} weights")
    print(coef)
    print()

# biases
# parameters associated with each node
for i, bias in enumerate(regressor.intercepts_):
    print(f"Layer {i + 1}: {bias.shape} biases")
    print(bias)
    print()
