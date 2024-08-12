import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei','Songti SC','STFangsong']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



data_frame_1 = pd.read_excel('out_prediction/predict_Covit_SFD2.xls', sheet_name='out')

print(data_frame_1.head())
# 若需要取列名为”UserName“的一列数据：
user_names = data_frame_1["prediction"]
predt_label=user_names.tolist()
true_label = data_frame_1["true_label"]
true_label=true_label.tolist()

for i in range(len(predt_label)):
    predt_label[i]="c%s"%predt_label[i]

# for i in range(len(true_label)):
#     true_label[i]=true_label[i][1:]



model_name="CVIT"
classes=["Safe Driving", "Text Right", "Phone Right",
"Text Left", "Phone Left", "Adjust Radio", "Drinking", "Reaching Behind", "Hair or Makeup", "Talking to Passenger"]



# cm = confusion_matrix(true_label,predt_label, labels=[ "c0","c1", "c2","c3", "c4", "c5","c6", "c7", "c8","c9"])

accuracy_score_out=accuracy_score(true_label,predt_label)
precision_score_out=precision_score(true_label,predt_label, average='macro')
recall_score_out=recall_score(true_label,predt_label, average='macro')
f1_score_out=f1_score(true_label,predt_label, average='macro')

print("accuracy_score:",accuracy_score_out )
print("precision_score；", precision_score_out)
print("recall_score：", recall_score_out)
print("f1_score：", f1_score_out)


# acc_list=["accuracy_score_out",accuracy_score_out,
#           "precision_score_out",precision_score_out,
#           "recall_score_out",recall_score_out,
#           "f1_score_out",f1_score_out]
# dataFrame2 = pd.DataFrame(acc_list)  # 说明行和列的索引名
# with pd.ExcelWriter("out_confusion_precise_test/%s_precise.xlsx"%model_name) as writer2:  # 一个excel写入多页数据
#     dataFrame2.T.to_excel(writer2, sheet_name='page1', float_format='%.6f')
#
#
# def plot_confusion_matrix(cm, savename, title='混淆矩阵'):
#     plt.figure(figsize=(12, 8), dpi=100)
#     np.set_printoptions(precision=2)
#
#     # 在混淆矩阵中每格的概率值
#     ind_array = np.arange(len(classes))
#     x, y = np.meshgrid(ind_array, ind_array)
#     for x_val, y_val in zip(x.flatten(), y.flatten()):
#         c = cm[x_val][y_val]#每一个方格的值
#         total=np.sum(cm[:,x_val])#每一行总值
#         plt.text(y_val, x_val, c, color='black', size=10, weight='bold', va='center', ha='center')
#         # if c > 0.001:
#         #     plt.text(y_val,x_val,  "%0.2f%s" % (c/total*100,"%"), color='black', size=10,weight='bold' ,va='center', ha='center')
#         # else:
#         #     plt.text(y_val, x_val, "0.00%" , color='black', size=10, weight='bold',va='center', ha='center')
#
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix", size=17)
#     plt.colorbar()
#     xlocations = np.array(range(len(classes)))
#     # 修改坐标轴字体及大小
#     plt.xticks(xlocations, classes, size=15,rotation=45)
#     plt.yticks(xlocations, classes, size=15)
#
#
#     plt.ylabel('Real Label',  size=15)
#     plt.xlabel('Predicted label',  size=15)
#
#     # offset the tick
#     tick_marks = np.array(range(len(classes))) + 0.5
#     plt.gca().set_xticks(tick_marks, minor=True)
#     plt.gca().set_yticks(tick_marks, minor=True)
#     plt.gca().xaxis.set_ticks_position('none')
#     plt.gca().yaxis.set_ticks_position('none')
#     plt.grid(True, which='minor', linestyle='-')
#     plt.gcf().subplots_adjust(bottom=0.15)
#
#     # show confusion matrix
#     plt.savefig('out_confusion_precise_test/%s_confusion.png'%model_name,bbox_inches ="tight", dpi=500, pad_inches=0.0,format="png")
#     plt.show()
#
# plot_confusion_matrix(cm, "savename", title='Confusion Matrix')