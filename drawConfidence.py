import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

_data_n3 = pd.read_excel('n3_confidence.xlsx', header=None)
data_n3 = np.array(_data_n3)

_data_n4 = pd.read_excel('n4_confidence.xlsx', header=None)
data_n4 = np.array(_data_n4)

_data_n5 = pd.read_excel('n5_confidence.xlsx', header=None)
data_n5 = np.array(_data_n5)

iterations3 = []
confidence3 = []
for i in range(0, len(data_n3)):
    if i % 10 == 0:
        iterations3.append(data_n3[i][0])
        confidence3.append(data_n3[i][1])

iterations4 = []
confidence4 = []
for i in range(0, len(data_n4)):
    if i % 10 == 0:
        iterations4.append(data_n4[i][0])
        confidence4.append(data_n4[i][1])

iterations5 = []
confidence5 = []
for i in range(0, len(data_n5)):
    if i % 10 == 0:
        iterations5.append(data_n5[i][0])
        confidence5.append(data_n5[i][1])

# print(iterations)
# print(confidence)

plt.plot(iterations3, confidence3, 'o-', label='n=3', linewidth=2, color='red')
plt.plot(iterations4, confidence4, 'o-', label='n=4', linewidth=2, color='blue')
plt.plot(iterations5, confidence5, 'o-', label='n=5', linewidth=2, color='gold')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Confidence', fontsize=14)
plt.legend()
plt.show()
