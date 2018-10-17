import numpy as np
import pandas as pd

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


df = pd.read_csv('IRIS.csv')
df = df.sample(frac=1.0)
#print(df.head())
df = np.array(df)

no_of_folds=10
k=10

class1 = "Iris-setosa"
class2 = "Iris-versicolor"

accuracy = []

precision1 = []
precision2 = []

recall1 = []
recall2 = []

for i in range(no_of_folds):
	train,test = kFold(i,df,no_of_folds)
	print(test.shape,train.shape)
	train_data = train[:,:df.shape[1]-1]
	train_labels = train[:,df.shape[1]-1:]
	test_data = test[:,:df.shape[1]-1]
	test_labels = test[:,df.shape[1]-1:]

	tp=0
	tn=0
	fp=0
	fn=0

	for t in range(test_data.shape[0]):
		distances=[]
		for tr in range(train_data.shape[0]):
			d = np.linalg.norm(train_data[tr]-test_data[t])
			distances.append((d,train_labels[tr,0]))
			#print(train_labels[tr,0])
		distances.sort()

		c1=0
		c2=0
		for j in range(k):
			if distances[j][1]==class1:
				c1+=1
			elif distances[j][1]==class2:
				c2+=1
		assigned_class=''
		if c1>c2:
			assigned_class=class1
		else:
			assigned_class=class2

		#print(c1,c2)

		actual_class=test_labels[t,0]

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