import numpy as np
import pandas as pd
import random

df = pd.read_csv('SPECT.csv')
#df = df.sample(frac=1.0)
df = np.array(df)

data = df[:,1:]
labels = df[:,:1]
print(data.shape,labels.shape)
no_of_samples = df.shape[0]
k = 2

class1 = 'Yes'
class2 = 'No'

centers = []
centers_index = random.sample(range(no_of_samples),k)
for i in range(len(centers_index)):
	centers.append(data[centers_index[i]])
#print(centers)
for itr in range(10):
	clusters = []
	clabels = []
	for i in range(k):
		clusters.append([])
		clabels.append([])

	for i in range(data.shape[0]):
		distances = []
		for j in range(len(centers)):
			d = np.linalg.norm(centers[j]-data[i])
			distances.append((d,j))
		distances.sort()
		#print(distances)
		leastindex = distances[0][1]
		clusters[leastindex].append(data[i])
		clabels[leastindex].append(labels[i][0])
		#print(clusters)

	#Update centers
	for i in range(len(centers)):
		centers[i] = sum(clusters[i])/len(clusters[i])

	#print(centers)

#Evaluation
c1yes = clabels[0].count(class1)
c1no = clabels[0].count(class2)

c2yes = clabels[1].count(class1) 
c2no = clabels[1].count(class2)

print(c1yes,c1no)
print(c2yes,c2no)

if c1yes>c2yes:
	print('C1 - Yes , C2 - No')
	print('C1 accuracy = ', c1yes/len(clabels[0]))
	print('C2 accuracy = ', c2no/len(clabels[1]))
else:
	print('C1 - No , C2 - Yes')
	print('C1 accuracy = ', c1no/len(clabels[0]))
	print('C2 accuracy = ', c2yes/len(clabels[1]))