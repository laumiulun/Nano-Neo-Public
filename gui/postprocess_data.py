#!/usr/bin/env python
# coding: utf-8

import numpy as np
# get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import os

const = {
    "C0":24.5,
    "C1":4763.099673,
    "C2":-251703.898944,
    "C3":2496563.762688,
    "C4":-6260430.021381,
    "C5":4045456.815465
}

cwd = os.getcwd()
dir = 'FS'
file = 'Fused Silica_20210412_1545_00008.txt'
title = 'Fused Silica'
# dir = 'POCO-P11'
# file = 'POCO-P11-10.txt'
# title = 'POCO ZXF-5Q'

def OliverPharr(x,A,h_f,m):
    return A*(x-h_f)**m

def read_indent_files(file):
    Af = 11.51
    hf = 94.8
    m = 1.315

    # POCO-P11-10 (Andy's fit)
    # Af = 0.00046
    # hf = 1095
    # m = 2.651

    pathRaw = os.path.join(cwd, 'raw', dir, file)
    pathProc = os.path.join(cwd, 'proc', dir, 'p' + file)

    raw_data = np.genfromtxt(pathRaw, skip_header=4, delimiter='\t')
    raw_x = raw_data[1:, 0]
    raw_y = raw_data[1:, 1]
    max_xloc = np.argmax(raw_x)
    max_x = raw_x[max_xloc]
    max_yloc = np.argmax(raw_y)
    max_y = raw_y[max_yloc]

    plt.plot(raw_x, raw_y, label='Indent data', color='cornflowerblue')
    plt.xlabel("Depth: h (nm)")
    plt.ylabel("Load: P (μN)")
    plt.title(title)
    plt.ylim([max_y*.01, max_y*1.05])

    proc_data = np.genfromtxt(pathProc, delimiter=',', skip_header=1)
    process_x = proc_data[:, 0]
    process_y = proc_data[:, 1]

    # print(process_x)

    # process_x = raw_data[max_xloc::,0]
    # process_y = raw_data[max_xloc::,1]

    plt.plot(process_x, process_y, label='Fit data', color='darkorange')

    # new_index = np.argwhere((process_x > 1400) & (process_x < 1652))
    # slice_x = process_x[new_index].flatten()
    # slice_y = process_y[new_index].flatten()

    # plt.plot(raw_x,raw_y)
    # plt.plot(process_x,process_y)
    # plt.plot(slice_x,slice_y)

    # max_x_fit = (raw_y[max_xloc]/Af)**(1/m)+hf
    max_x_fit = (max_y/Af)**(1/m)+hf
    # print(max_x_fit)
    x_fit = np.arange(0, max_x_fit)
    fit = np.zeros(len(x_fit))
    for i in range(len(x_fit)):
        fit[i] = OliverPharr(x_fit[i], Af, hf, m)

    plt.plot(x_fit, fit, label='Model', linestyle='--', color='tab:green')
#     with open('test_sample2.txt','w') as f:
#         f.write('# X,Y\n')
#         for i in range(len(slice_x)):
#             f.write(str(slice_x[i]) + ',' + str(slice_y[i]) + '\n')
    # len_slice = len(x_power)

    dx = x_fit[-1] - x_fit[-2]
    dy = fit[-1] - fit[-2]
    dydx = dy/dx
    print("Stiffness S = dP/dh (uN/nm):", np.round(dydx, 3))
    b = fit[-1] - dydx * x_fit[-1]
    linear_response = []
    x_linear = np.arange(0, max_x_fit)
    for i in range(len(x_linear)):
        linear_response.append(dydx*x_linear[i] + b)

    plt.plot(x_linear, linear_response, label='Stiffness (S = dP/dh)', linestyle='--', color='tab:purple')
    plt.legend()

    h_max = max_x
    # h_max = max_x_fit
    P = max_y
    # P = raw_y[max_xloc]
    S = dydx

    hc = h_max - 0.75 * P / S
    # A = const['C0'] * hc ** 2 + const['C1'] * hc ** 1 + const['C2'] * hc ** (1 / 2) + const['C3'] * hc ** (1 / 4) + \
    #     const['C4'] * hc ** (1 / 8) + const['C5'] * hc ** (1 / 16)
    A = const['C0']*hc**2 + const['C1']*hc**1 + const['C2']*hc**(0.5) + const['C3']*hc**(0.25) + const['C4']*hc**(0.125) + const['C5']*hc**(0.0625)
    # A = const['C0'] * hc ** 2 + const['C1']*hc**1 + const['C2']*hc**(0.5)

    H = (P / A) * 10 ** 6
    # E = np.pi ** (1 / 2) * S / (2 * A ** (1 / 2)) * 10 ** 3
    E = np.pi ** (0.5) * S / (2 * A ** (0.5)) * 10 ** 3

    print('Maximum Depth (nm):' + str(h_max))
    print('Maximum Load (uN):' + str(P))
    print("Area: " + str(np.round(A, 5)) + ' nm^2')
    print("H: " + str(np.round(H, 5)) + ' MPa')
    print("E: " + str(np.round(E, 5)) + ' GPa')

# # Hardness
# H = P/A(hc)
# P = Maximum Load point
# A = Area
# h = Maximum depth
# S = slope at maximum depth

read_indent_files(file)
plt.show()
