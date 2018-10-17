import numpy as np
import pandas as pd

def kFold(fold_no,df,fold_length):
	train=[]
	test=[]
	for j in range(df.shape[0]):
		#print(i,fold_no)
		if j%fold_length==i:
			test.append(df[j])
		else:
			train.append(df[j])

	train = np.array(train)
	test = np.array(test)
	return train,test


df = pd.read_csv('SPECT.csv')
df = df.sample(frac=1.0)
#print(df.head())
df = np.array(df)

no_of_folds=10

class1 = "Yes"
class2 = "No"

accuracy = []

precision1 = []
precision2 = []

recall1 = []
recall2 = []

for i in range(no_of_folds):
	train,test = kFold(i,df,no_of_folds)
	print(test.shape,train.shape)
	train_data = train[:,1:]
	train_labels = train[:,0:1]
	test_data = test[:,1:]
	test_labels = test[:,0:1]

	tp=0
	tn=0
	fp=0
	fn=0

	labels = list(train_labels)
	yess = labels.count(class1)
	nos = labels.count(class2)
	print(yess,nos)

	attribute_ones = []
	attribute_zeros = []
	for j in range(train_data.shape[1]):
		oneyes=0
		oneno=0
		zeroyes=0
		zerono=0
		for k in range(train_data.shape[0]):
			if train_data[k,j]==1 and labels[k]==class1:
				oneyes+=1
			elif train_data[k,j]==1 and labels[k]==class2:
				oneno+=1
			elif train_data[k,j]==0 and labels[k]==class1:
				zeroyes+=1
			elif train_data[k,j]==0 and labels[k]==class2:
				zerono+=1

		attribute_ones.append((oneyes,oneno))
		attribute_zeros.append((zeroyes,zerono))

	for j in range(test_data.shape[0]):
		pyes=1
		pno=1
		for k in range(test_data.shape[1]):
			if test_data[j,k]==1:
				pyes*=(attribute_ones[k][0]/yess)
				pno*=(attribute_ones[k][1]/nos)
			elif test_data[j,k]==0:
				pyes*=(attribute_zeros[k][0]/yess)
				pno*=(attribute_zeros[k][1]/nos)
		#print(yess+nos,train_data.shape[0])
		pyes*=(yess/(yess+nos))
		pno*=(nos/(nos+yess))
		
		if pyes>pno:
			assigned_class=class1
		else:
			assigned_class=class2

		actual_class=test_labels[j,0]

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

	print('Fold number = {}, Accuracy = {}, Precision = {}, Recall = {}'.format(i+1,accuracy[len(accuracy)-1],precision1[len(precision1)-1],recall1[len(recall1)-1]))

final_accuracy = sum(accuracy)/len(accuracy)
final_precision1 = sum(precision1)/len(precision1)
final_precision2 = sum(precision2)/len(precision2)
final_recall1 = sum(recall1)/len(recall1)
final_recall2 = sum(recall2)/len(recall2)

print('Final Average Values:')
print('Accuracy = {}, Precision = {}, Recall = {}'.format(final_accuracy,final_precision1,final_recall1))