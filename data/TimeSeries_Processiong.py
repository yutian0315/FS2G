import numpy as np
from sklearn import preprocessing
import torch
import os
from tqdm import tqdm
from random import randrange
from nilearn import connectome

def Anchor_Sample_Time(TimeSeries, TargetSeriesLength):
    # TimeSeries 所有样本的时间序列
    AllSubject = [] # 初始化存放处理过后的时间序列的列表
    print("AnchorSample Process TimeSeries:")
    for subject_series in tqdm(TimeSeries):
        RawSeriesLength = subject_series.shape[0] # 原始时间序列的长度
        AnchorInterVal = RawSeriesLength / TargetSeriesLength # 锚点间隔
        SlideNode = AnchorInterVal
        AnchorInterVal = int(AnchorInterVal)
        if AnchorInterVal % 2 == 0:
            AnchorSampleLength = AnchorInterVal + 1
        else:
            AnchorSampleLength = AnchorInterVal + 2

        pre_timeseries = []
        for i in range(60):
            AnchorNode = int(SlideNode)
            AnchorSampleIndex_down = int(AnchorNode - ((AnchorSampleLength - 1) / 2))
            AnchorSampleIndex_up = int(AnchorNode + ((AnchorSampleLength - 1) / 2))
            AnchorSampleData = subject_series[AnchorSampleIndex_down: AnchorSampleIndex_up + 1, :]
            MeanTimeSeries = np.mean(AnchorSampleData, axis=0)
            pre_timeseries.append(MeanTimeSeries)
            SlideNode += AnchorInterVal
        AllSubject.append(pre_timeseries)
    AllSubject = np.array(AllSubject)
    return AllSubject

def DynamicFC(Timeseries, NumOfGraph, triu=True):
    np.seterr(divide='ignore', invalid='ignore')
    # timeseries 所有样本的时间序列信息 【batch * 时间点 *脑区数】
    # 需要生成多少个动态图
    print("生成动态图：")
    all_subject_fc = []
    for subject_series in tqdm(Timeseries):
        RawSeriesLength = subject_series.shape[0] # 原始时间序列的长度
        AnchorInterVal = RawSeriesLength / (NumOfGraph+1) # 动态图间隔
        SlideNode = AnchorInterVal
        TempAnchorInterVal = int(AnchorInterVal)
        if TempAnchorInterVal % 2 == 0:
            AnchorSampleLength = TempAnchorInterVal + 1
        else:
            AnchorSampleLength = TempAnchorInterVal + 2

        # 准备动态图时间段
        FC_TimeSeries = [] # NumOfGraph * 间隔长度 * 脑区数
        for i in range(NumOfGraph):
            AnchorNode = int(SlideNode)
            AnchorSampleIndex_down = int(AnchorNode - ((AnchorSampleLength - 1) / 2))
            AnchorSampleIndex_up = int(AnchorNode + ((AnchorSampleLength - 1) / 2))
            AnchorSampleData = subject_series[AnchorSampleIndex_down: AnchorSampleIndex_up + 1, :]
            FC_TimeSeries.append(AnchorSampleData)
            SlideNode += AnchorInterVal
        FC_TimeSeries = np.stack(FC_TimeSeries, axis=0)

        # 3个间隔一组计算FC图
        subject_fc_list = []
        for i in range(NumOfGraph):
            if i == 0:
                Group_i_fc = np.concatenate(FC_TimeSeries[i:i+2, :, :], axis=0)
            elif i == NumOfGraph -1:
                Group_i_fc = np.concatenate(FC_TimeSeries[i-1:i+1, :, :], axis=0)
            else:
                Group_i_fc = np.concatenate(FC_TimeSeries[i-1:i+2, :, :], axis=0)
            conn_measure = connectome.ConnectivityMeasure(kind='correlation')
            fc = conn_measure.fit_transform([Group_i_fc])[0]
            if triu == True:
                fc = fc[np.triu_indices(fc.shape[0], k=1)]
            subject_fc_list.append(fc)
        all_subject_fc.append(subject_fc_list)
    return np.array(all_subject_fc)

def ShortestTimeSeires(Timeseries):
    # 根据最短时间序列的长度，把所有样本的时间长度统一

    # 找出最短时间序列长度
    print("Intercept the shortest TimeSeries:")
    ShortLength = 9999
    for subject_timeseries in tqdm(Timeseries):
        subject_length = len(subject_timeseries)
        if subject_length < ShortLength:
            ShortLength = subject_length
    Timeseries = np.stack(([d[:ShortLength, :] for d in Timeseries]), axis=0)
    return Timeseries

def FillZeroTimeSeires(Timeseries):
    # 以最长的时间序列长度为基准，短的时间填充0

    # 找出最长时间序列长度
    print("Fill Zero TimeSeries:")
    LongLength = 0
    for subject_timeseries in Timeseries:
        subject_length = len(subject_timeseries)
        if subject_length > LongLength:
            LongLength = subject_length
    FillTimeSeries = []
    for subject_timeseries in tqdm(Timeseries):
        FillLength = LongLength - len(subject_timeseries)
        subject_timeseries = np.pad(
            subject_timeseries,
            (
                (0, FillLength),
                (0, 0)
            ),
            'constant',  # 一样的值填充
            constant_values = (
                0,  # 左边填充的数值
                0  # 右边填充的数值
            ),
        )
        FillTimeSeries.append(subject_timeseries)
    return np.array(FillTimeSeries)