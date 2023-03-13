import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

_data_n3 = pd.read_excel('C:/Users/user/Desktop/n3.xlsx', header=None)
data_n3 = np.array(_data_n3)

_data_n4 = pd.read_excel('C:/Users/user/Desktop/n4.xlsx', header=None)
data_n4 = np.array(_data_n4)

_data_n5 = pd.read_excel('C:/Users/user/Desktop/n5.xlsx', header=None)
data_n5 = np.array(_data_n5)

iterations3 = []
confidence3 = []
for i in range(1, 11):
    # if i % 10 == 0:
    iterations3.append(data_n3[i][0])
    confidence3.append(data_n3[i][10])

iterations4 = []
confidence4 = []
for i in range(1, 11):
    # if i % 10 == 0:
    iterations4.append(data_n4[i][0])
    confidence4.append(data_n4[i][10])

iterations5 = []
confidence5 = []
for i in range(1, 11):
    # if i % 10 == 0:
    iterations5.append(data_n5[i][0])
    confidence5.append(data_n5[i][10])

# print(iterations)
# print(confidence)
iterations3 = [str(i) for i in iterations3]
plt.bar(iterations3, confidence3, label='n=3', linewidth=2, color='dodgerblue')
# plt.plot(iterations4, confidence4, 'o-', label='n=4', linewidth=2, color='blue')
# plt.bar(iterations5, confidence5,  label='n=5', linewidth=2, color='dodgerblue')
plt.xlabel('Executor serial number', fontsize=14)
plt.ylabel('Average online time', fontsize=14)
plt.legend()
plt.show()
