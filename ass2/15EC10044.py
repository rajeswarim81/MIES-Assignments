import numpy as np 
import matplotlib.pyplot as plt

#the sigma_pos, and sigma_neg have to be changed for competitor
mu_pos,sigma_pos=37,1
mu_neg,sigma_neg=32,4
y_pos=np.zeros(310000)
y_neg=np.zeros(310000)
X=np.zeros(310000)
for x in range(200000,510000): 
	y_pos[x-200000]=1/(sigma_pos * np.sqrt(2 * np.pi)) * np.exp( - (x/10000 - mu_pos)**2 / (2 * sigma_pos**2))#malignant distribution
	y_neg[x-200000]=1/(sigma_neg * np.sqrt(2 * np.pi)) * np.exp( - (x/10000 - mu_neg)**2 / (2 * sigma_neg**2))#benign distribution
	X[x-200000]=x/10000

plt.figure(num="1")
plt.plot(X,y_pos)
plt.plot(X,y_neg)
#may or may not be plotted

tot_pos=np.zeros(310000)
tot_pos[0]= y_pos[0]
for i in range(1,y_pos.shape[0]):
	tot_pos[i]=tot_pos[i-1]+y_pos[i]
pos=tot_pos[309999]	

tot_neg=np.zeros(310000)
tot_neg[0]=y_neg[0]
for i in range(1,y_neg.shape[0]):
	tot_neg[i]=tot_neg[i-1]+y_neg[i]
neg=tot_neg[309999]	

#true positive rate= true positives/total positives
#flase positive rate=false positives/total negatives

tpr=np.zeros(310000)
fpr=np.zeros(310000)

for thres in range(0,X.shape[0]):
	tpr[thres]=1-tot_pos[thres]/pos
	fpr[thres]=1-tot_neg[thres]/neg

plt.figure(num="2")
plt.plot(fpr,tpr)
plt.title('Third Eye Technologies')
#plt.title('competitor')
plt.xlabel('false positive ratio')
plt.ylabel('true positive ratio')
plt.show()

auc=0
for i in range(0,tpr.shape[0]):
	auc=auc+tpr[i]
print auc*0.0001