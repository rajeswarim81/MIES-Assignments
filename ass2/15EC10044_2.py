import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.integrate import trapz
import numpy as np

def gaussian_dist(mean,std):
    variance = np.square(std)
    x = np.arange(mean-5*std,mean+5*std,.0001)
    f= np.exp(-np.square(x-mean)/(2*variance))/(np.sqrt(2*np.pi*variance))
    #plt.plot(x,f)
    return f,x

def specificity(f,x,th):
    if len(x[x[:]<=th])!=0:
        I1=simps(f[x[:]<=th],x[x[:]<=th])
    else:
        I1=0
    if len(x[x[:]>th])!=0:
        I2=simps(f[x[:]>th],x[x[:]>th])
    else:
        I2=0
    return I1/(I1+I2)
def sensitivity(f,x,th):
    if len(x[x[:]>=th])!=0:
        I1=simps(f[x[:]>=th],x[x[:]>=th])
    else:
        I1=0
    if len(x[x[:]<th])!=0:
        I2=simps(f[x[:]<th],x[x[:]<th])
    else:
        I2=0
    return I1/(I1+I2)
def ROC(mean1,std1,mean2,std2):
    f1,x1=gaussian_dist(mean1,std1)
    f2,x2=gaussian_dist(mean2,std2)
    #plt.savefig("dist_1.png")
    plt.show()
    list1=[]
    list2=[]
    for threshold in range(15,50):
      tpf=sensitivity(f1,x1,threshold)
      tnf=specificity(f2,x2,threshold)
      fpf=1-tnf
      list1.append(tpf)
      list2.append(fpf)
    plt.plot(list2,list1)
    plt.show()
    #print(list1,list2)
    AUC=-trapz(np.array(list1),np.array(list2))
    return f1,f2,AUC

f11,f12,AUC1=ROC(37,1,32,4)
f21,f22,AUC2=ROC(37,2,32,3)
print(AUC1,AUC2)
#plt.show()
