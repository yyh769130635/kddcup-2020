# Author:peter young
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.axisartist as axisartist
from mpl_toolkits.mplot3d import Axes3D #画三维图不可少

dataset=pd.read_csv('new.csv',sep=',',header=None)
a=np.array(dataset)
# writer = pd.ExcelWriter('hhh.xlsx')
# dataset.to_excel(writer,'page_1',float_format='%.5f')  #关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
# writer.save()  #关键4
##二维表格

Y = np.arange(np.shape(a)[0], 0,-1)
X = np.arange(np.shape(a)[1],0,  -1)
X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, a, cmap=cm.gist_rainbow)
plt.show()

