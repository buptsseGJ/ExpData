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

    acc_gen_tii = data_tii[['Epoch1', 'Loss1']]
    acc_vul_tii = data_tii[['Epoch2', 'Loss2']]
    acc_gen_tse = data_tse[['Epoch1', 'Loss1']]
    acc_vul_tse = data_tse[['Epoch2', 'Loss2']]

    plt.plot(acc_vul_tii['Epoch2'], acc_vul_tii['Loss2'],label='IoTSeeker:TII',color='darkorange',linestyle="--")
    plt.plot(acc_gen_tii['Epoch1'], acc_gen_tii['Loss1'], label='Gemini:TII', color='green',linestyle="--")
    plt.plot(acc_vul_tse['Epoch2'], acc_vul_tse['Loss2'],label='BinSeeker-:TSE',color='red',linestyle=":")
    plt.plot(acc_gen_tse['Epoch1'], acc_gen_tse['Loss1'], label='Gemini:TSE', color='gold',linestyle=":")

    # x，y轴取值范围设置
    plt.xlim(0, 100)
    plt.ylim(0.6, 1.3)
    # plt.yscale('log')
    # plt.axis(yscale='log')
    plt.legend(loc='best')
    # 设置title和x，y轴的label
    plt.title("(a) Loss value for diffrent training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # 保存图片到指定路径
    plt.savefig("loss-combine.pdf")
    # 展示图片 *必加


if __name__ == '__main__':
    main("../loss-csv-tii.csv", "../loss-csv-tse.csv")