import numpy as np
import pandas as pd
import math

def kFold(fold_no,df,fold_length):
	train=[]
	test=[]

	for j in range(df.shape[0]):
		if j%fold_length==fold_no:
			test.append(df[j])
		else:
			train.append(df[j])

	train = np.array(train)
	test = np.array(test)
	return train,test

def activation(output):
	#print(output.shape)
	return 1/(1 + np.exp(-(output)))

def activation_der(output):
	return output*(1-output)

df = pd.read_csv('IRIS.csv')
df = df.sample(frac=1.0)
df = np.array(df)

learning_rate=0.9
no_of_folds = 10

class1 = "Iris-setosa"
class2 = "Iris-versicolor"

accuracy = []

precision1 = []
precision2 = []

recall1 = []
recall2 = []

n = df.shape[1]-1
#print(n)
hidden_wts = np.full((5,n),1/(n*5)).astype('float32')
output_wts = np.full((1,5),1/5).astype('float32')
hidden_biases = np.full((5,1),1/6).astype('float32')
output_biases = np.full((1,1),1/6).astype('float32')

#--------
for k in range(no_of_folds):
	train, test = kFold(k,df,no_of_folds)
	print(train.shape,test.shape)
	train_data = train[:,:df.shape[1]-1]
	train_labels = train[:,df.shape[1]-1:]
	test_data = test[:,:df.shape[1]-1]
	test_labels = test[:,df.shape[1]-1:]

	tp=0
	tn=0
	fp=0
	fn=0

	for itr in range(5):
		# hidden_wts = np.full((5,n),1/(n*5)).astype('float32')
		# output_wts = np.full((1,5),1/5).astype('float32')
		# hidden_biases = np.full((5,1),1/6).astype('float32')
		# output_biases = np.full((1,1),1/6).astype('float32')

		for i in range(train_data.shape[0]):
			
			train_sample = np.array(train_data[i].reshape((4,1)))

			hidden_output = np.dot(hidden_wts,train_data[i]).astype('float32')
			hidden_output = hidden_output.reshape((5,1))
			hidden_output = hidden_output + hidden_biases
			hidden_output = activation(hidden_output)

			final_output = np.dot(output_wts,hidden_output).astype('float32')
			final_output = final_output.reshape((1,1))
			final_output = final_output + output_biases
			final_output = activation(final_output)

			if (final_output[0,0]>0.5 and train_labels[i,0]==class1) or (final_output[0,0]<0.5 and train_labels[i,0]==class2):

				error_op = final_output[0,0] * (1-final_output[0,0]) * (0.5-final_output)
				error_hidden = hidden_output * (1-hidden_output) * np.dot(output_wts.transpose(),error_op)

				output_wts = output_wts + learning_rate * np.dot(error_op,hidden_output.transpose())
				#print('opwt = ',output_wts)
				output_biases = output_biases + learning_rate*error_op
				hidden_wts = hidden_wts + learning_rate * np.dot(error_hidden,train_sample.transpose())
				#print('hidden_wts=',hidden_wts)
				hidden_biases = hidden_biases + learning_rate*error_hidden
			#break
		#break
	for i in range(test_data.shape[0]):

		hidden_output = np.dot(hidden_wts,test_data[i]).astype('float32')
		hidden_output = hidden_output.reshape((5,1))
		hidden_output = hidden_output + hidden_biases
		hidden_output = activation(hidden_output)

		final_output = np.dot(output_wts,hidden_output).astype('float32')
		final_output = final_output.reshape((1,1))
		final_output = final_output + output_biases
		final_output = activation(final_output)

		if final_output[0,0]<=0.5:
			assigned_class=class1
		elif final_output[0,0]>0.5:
			assigned_class=class2

		actual_class = test_labels[i,0]

		if assigned_class==class1 and actual_class==class1:
			tp+=1
		elif assigned_class==class1 and actual_class!=class1:
			fp+=1
		elif assigned_class==class2 and actual_class==class2:
			tn+=1
		elif assigned_class==class2 and actual_class!=class2:
			fn+=1

	accuracy.append((tp+tn)/(tp+tn+fp+fn))

	precision1.append(tp/(tp+fp) if tp+fp!=0 else 0)
	precision2.append(tn/(tn+fn) if tn+fn!=0 else 0)

	recall1.append(tp/(tp+fn) if tp+fn!=0 else 0)
	recall2.append(tn/(tn+fp) if tn+fp!=0 else 0)

	print('Fold number = {}, Accuracy = {}, Precision = {}, Recall = {}'.format(k+1,accuracy[len(accuracy)-1],precision1[len(precision1)-1],recall1[len(recall1)-1]))

final_accuracy = sum(accuracy)/len(accuracy)
final_precision1 = sum(precision1)/len(precision1)
final_precision2 = sum(precision2)/len(precision2)
final_recall1 = sum(recall1)/len(recall1)
final_recall2 = sum(recall2)/len(recall2)

print('Final Average Values:')
print('Accuracy = {}, Precision = {}, Recall = {}'.format(final_accuracy,final_precision1,final_recall1))