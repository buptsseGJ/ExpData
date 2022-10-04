#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diff(acc_gen_tii, acc_vul_tii, acc_gen_tse,acc_vul_tse):
    acc_tse_vul_list = (acc_vul_tse.values.tolist())
    acc_tii_vul_list = (acc_vul_tii.values.tolist())
    diffListTwo= [acc_tse_vul_list[i][1] - acc_tii_vul_list[i][1] for i in range(len(acc_tse_vul_list))]
    acc_tse_gen_list = (acc_gen_tse.values.tolist())
    acc_tii_gen_list = (acc_gen_tii.values.tolist())
    diffListGemini = [acc_tse_gen_list[i][1] - acc_tii_gen_list[i][1] for i in range(len(acc_tse_gen_list))]

    x_two = [v for v in range(1, 101)]

    plt.plot(x_two, diffListTwo, label='IoTSeeker:TII&BinSeeker-:TSE',color='darkorange')
    plt.plot(x_two, diffListGemini, label='Gemini:TII&Gemini:TSE',color='blue')
    plt.xlim(0, 100)
    plt.ylim(-0.015, 0.015)
    # plt.yscale('log')
    # plt.axis(yscale='log')
    plt.legend(loc='best')
    # 设置title和x，y轴的label
    plt.title("(a) Loss value for diffrent training epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Difference")
    # 保存图片到指定路径
    plt.savefig("loss-diff.pdf")

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
    diff(acc_gen_tii, acc_vul_tii, acc_gen_tse,acc_vul_tse)

if __name__ == '__main__':
    main("../loss-csv-tii.csv", "../loss-csv-tse.csv")