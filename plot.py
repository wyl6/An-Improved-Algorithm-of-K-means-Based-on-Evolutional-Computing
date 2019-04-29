import matplotlib.pyplot as plt  
import numpy as np  


T = [100,200,300,400,500]

Time = [6.503,10.286,21.835,40.774,68.224]
Acc = [0.866,
0.882,
0.890,
0.891,
0.892]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Running time on CPU')
p1 = ax1.plot(T, Time, label='Time/s')
ax2 = ax1.twinx()  # this is the important function
ax2.set_ylabel('Acc')
ax1.set_xlabel('T')
p2, = ax2.plot(T, Acc, c='m', label='Acc')
fig.legend(bbox_to_anchor=(0.35,0.75))
fig.savefig('T_Time_Acc.png')
fig.show()
#
# In[2]: double y
#
T = [0.5, 0.6, 0.7, 0.8, 0.9]
Time = [5.435,
5.763,
5.702,
6.187,
6.439]
Acc = [0.827,
0.836,
0.840,
0.842,
0.873]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Running time on CPU')
p1 = ax1.plot(T, Time, label='Time/s')
ax2 = ax1.twinx()  # this is the important function
ax2.set_ylabel('Acc')
ax1.set_xlabel('alpha')
p2, = ax2.plot(T, Acc, c='m', label='Acc')
#fig.legend([p1, p2], ['Time','Acc'],loc = 'best', bbox_to_anchor=(0.3,0.75))
fig.legend(bbox_to_anchor=(0.35,0.75))
fig.savefig('Alpha_Time_Acc.png')
fig.show()


# In[3]: double y 

#Methods = ['K', 'KSJ', 'KSAJ', 'KFJ', 'KFAJ']
#Acc = [0.667,
#0.810,
#0.862,
#0.750,
#0.885]
#SSE = [142.754,
#101.902,
#86.451,
#121.576,
#80.161]
#
#
#colums_x = ['aa','bc','ad','bd','de'] 
#colums_y = [12,14,10,15,8] 
#plt.xticks(range(len(Methods)),Methods) 
#plt.bar(range(len(Acc)),Acc,0.3, label='Acc') 
#plt.legend(bbox_to_anchor=(0.41,0.86))
#
#plt1 = plt.twinx()
#plt1.plot(range(len(SSE)), SSE, c='m', label='SSE')
#plt1.legend(bbox_to_anchor=(0.8,0.99))
#plt.savefig('Methods.png')
#
#plt.show()

