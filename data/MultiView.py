import sys
from tqdm import tqdm
import util
import  os
import numpy as np
from nilearn import connectome
# 脑区时间序列数据读取
def GetTimeSeries(DataPath):
    # 变量路径下的文件名
    FileName = os.listdir(DataPath)
    TimeSeriesList = []

    Max = 0
    print("Load TimeSeries:")
    for Name in tqdm(FileName):
        # 得到文件的绝对路径

        FilePath = os.path.join(DataPath, Name)
        TimeSeries = np.loadtxt(FilePath, skiprows=0)
        if TimeSeries.shape[0] > Max:
            Max = TimeSeries.shape[0]
        TimeSeriesList.append(TimeSeries)
    return TimeSeriesList, Max

# TimeSeriesList, Max = GetTimeSeries('data/AAL871')
# fc_list = []
# for subject_Series in TimeSeriesList:
#     subject_Series = np.array(subject_Series)
#     conn_measure = connectome.ConnectivityMeasure(kind='correlation')
#     fc = conn_measure.fit_transform([subject_Series])[0]
#     fc_list.append(fc)
# fc_list = np.array(fc_list)
# np.save("data/FC871.npy", fc_list)
FC871 = np.load("data/FC871.npy")
MultiViewFC_list = []
for subjectFC in FC871:
    first = np.where(subjectFC < 0.25, subjectFC, 0)

    second = np.where(subjectFC < 0.5, subjectFC, 0)
    second = np.where(second > 0.25, second, 0)

    third = np.where(subjectFC < 0.75, subjectFC, 0)
    third = np.where(third > 0.5, third, 0)

    foured = np.where(subjectFC > 0.75, subjectFC, 0)
    MultiViewFC = np.concatenate((first, second, third, foured), axis=1)
    MultiViewFC = np.swapaxes(MultiViewFC.reshape((116,4,116)), 1, 2)
    MultiViewFC_list.append(MultiViewFC)
print("finish")
MultiViewFC_list = np.array(MultiViewFC_list)
np.save("data/MultiViewFC.npy",MultiViewFC_list)
# x1 = np.array(range(9)).reshape((3,3))
# x2 = np.array(range(9)).reshape((3,3))
# x3 = np.array(range(9)).reshape((3,3))
# x4 = np.array(range(9)).reshape((3,3))
# MultiViewFC = np.concatenate((x1, x2, x3, x4), axis=1)
# print(MultiViewFC)
# print(MultiViewFC.reshape((4,4,4)))
# MultiViewFC = np.swapaxes(MultiViewFC.reshape((4,4,4)), 1, 2)
# print(MultiViewFC)
