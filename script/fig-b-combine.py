#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    plt.plot(auc_vul_tii['Epoch2'], auc_vul_tii['AUC2'],label='IoTSeeker:TII',color='darkorange',linestyle="--")
    plt.plot(auc_gen_tii['Epoch1'], auc_gen_tii['AUC1'], label='Gemini:TII', color='green',linestyle="--")
    plt.plot(auc_vul_tse['Epoch2'], auc_vul_tse['AUC2'],label='BinSeeker-:TSE',color='red',linestyle=":")
    plt.plot(auc_gen_tse['Epoch1'], auc_gen_tse['AUC1'], label='Gemini:TSE', color='gold',linestyle=":")

    # x，y轴取值范围设置
    plt.xlim(0, 100)
    plt.ylim(0.6, 1.0)
    # plt.yscale('log')
    # plt.axis(yscale='log')
    plt.legend(loc='best')
    # 设置title和x，y轴的label
    plt.title("(b) AUC value for different training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    # 保存图片到指定路径
    plt.savefig("auc-combine.pdf")
    # 展示图片 *必加


if __name__ == '__main__':
    main("../auc-csv-tii.csv", "../auc-csv-tse.csv")