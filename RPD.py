#RPDクラスを定義する
import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RPD:

    #RPDクラスの定義
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.rpd = self.calc_rpd()
    
    #RPDの計算
    def calc_rpd(self):
        #行列の列ごとに標準偏差で割る
        matrix_divided_by_std1 = self.divide_matrix_by_std(self.data1)
        matrix_divided_by_std2 = self.divide_matrix_by_std(self.data2)

        #それぞれの行列積を計算
        matrix_product1 = np.dot(matrix_divided_by_std1, matrix_divided_by_std1.T)
        matrix_product2 = np.dot(matrix_divided_by_std2, matrix_divided_by_std2.T) 

        #分子を計算(行列積の差のフロベニウスノルム)
        numerator = self.calc_frobenius_norm(matrix_product1 - matrix_product2)
    
        #分母を計算(行列積のフロベニウスノルムの積)
        denominator = self.calc_frobenius_norm(matrix_product1)*self.calc_frobenius_norm(matrix_product2)
    
        #RPDを計算
        rpd = numerator / denominator

        return rpd
    
    #引数で与えられた行列を列ごとに標準偏差で割って返す
    def divide_matrix_by_std(self, data):
        return data / np.std(data, axis=0)

    #行列のフロベニウスノルムを計算する
    def calc_frobenius_norm(self, data):
        return np.linalg.norm(data, ord='fro')
    
    #インスタンス変数rpdのゲッター
    def get_rpd(self):
        return self.rpd



#RPDを計算し戻り値として返す関数
def rpd(data1, data2):
    #RPDを計算する関数
    rpd = RPD(data1, data2)
    return rpd.get_rpd()


#複数のデータをまとめてRPDを計算して行列で返す関数
def rpd_matrix(data_list):
    rpd_matrix = np.zeros((len(data_list), len(data_list)))
    for i in range(len(data_list)):
        for j in range(len(data_list)):
            rpd_matrix[i][j] = rpd(data_list[i], data_list[j])
    return rpd_matrix


    
if __name__ == '__main__':

    data1 = [[1, 1],
             [100, 920]]
    data2 = [[890, 1000], 
             [1000, 920]]
    data3 = [[1, 1],
             [100, 920]]
    li = [data1, data2, data3]

    li  = np.array(li)

    data1 = np.array(data1)
    data2 = np.array(data2)
    print(rpd(data1, data2))
    print(" ")
    print(rpd_matrix(li))
    