import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


df_train = pd.read_csv("train.csv")
#df_test = pd.read_csv("test.csv")
data_train = df_train.as_matrix().astype(np.float32)
#data_test = df_test.as_matrix().astype(np.float32)
np.random.shuffle(data_train)

X_train = data_train[:,1:]
Y_train = data_train[:,0].astype(np.int32)

def normalized_data(data):
	mu = data.mean(axis = 0)
	std = data.std(axis = 0)
	std_zero = std == 0
	std_zero = std_zero.astype(np.int32)
	std = std + std_zero
	t = (data - mu) / std
	#t = (data - mu) 
	return t 

def y2indicator(Y_data):
	N = len(Y_data)
	K = 10
	T = np.zeros((N, K))
	for i in range(N):
		T[i,Y_data[i]] = 1
	return T



#normalized_train = normalized_data(X_train)
normalized_train = X_train / 255
ind = y2indicator(Y_train)

def get_data():
	return normalized_train, ind, Y_train
#print(normalized_train.shape)
#print(normalized_train)
#print(ind)
# print(ind.shape)
# for i in range(10):
# 	print(Y_train[i])
# for i in range(10):
# 	print(ind[i])
# print(normalized_train[0])
# print(normalized_train[1])
# print(normalized_train[2])
# print(normalized_train[3])