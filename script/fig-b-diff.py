#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diff(auc_gen_tii, auc_vul_tii, auc_gen_tse,auc_vul_tse):
    auc_tse_vul_list = (auc_vul_tse.values.tolist())
    auc_tii_vul_list = (auc_vul_tii.values.tolist())
    diffListTwo= [auc_tse_vul_list[i][1] - auc_tii_vul_list[i][1] for i in range(len(auc_tse_vul_list))]
    auc_tse_gen_list = (auc_gen_tse.values.tolist())
    auc_tii_gen_list = (auc_gen_tii.values.tolist())
    diffListGemini = [auc_tse_gen_list[i][1] - auc_tii_gen_list[i][1] for i in range(len(auc_tse_gen_list))]

    x_two = [v for v in range(1, 101)]

    plt.plot(x_two, diffListTwo, label='IoTSeeker:TII&BinSeeker-:TSE',color='darkorange')
    plt.plot(x_two, diffListGemini, label='Gemini:TII&Gemini:TSE',color='blue')
    plt.xlim(0, 100)
    plt.ylim(-0.015, 0.015)
    # plt.yscale('log')
    # plt.axis(yscale='log')
    plt.legend(loc='best')
    # 设置title和x，y轴的label
    plt.title("(b) AUC value for different training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Difference")
    # 保存图片到指定路径
    plt.savefig("auc-diff.pdf")

def main(file_tii, file_tse):
    data_tii=pd.DataFrame(pd.read_csv(file_tii))
    data_tse=pd.DataFrame(pd.read_csv(file_tse))
    # 如果有空值 就将这一行删除
    data_tii.dropna()
    data_tse.dropna()

    auc_gen_tii = data_tii[['Epoch1', 'AUC1']]
    auc_vul_tii = data_tii[['Epoch2', 'AUC2']]
    auc_gen_tse = data_tse[['Epoch1', 'AUC1']]
    auc_vul_tse = data_tse[['Epoch2', 'AUC2']]
    diff(auc_gen_tii, auc_vul_tii, auc_gen_tse,auc_vul_tse)

if __name__ == '__main__':
    main("../auc-csv-tii.csv", "../auc-csv-tse.csv")