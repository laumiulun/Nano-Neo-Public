import numpy as np
import matplotlib.pyplot as plt
import os

def OliverPharr(x,A,h_f,m):
    return A*(x-h_f)**m

def read_files(raw_data,limits=(None,None)):
    # Raw Data
    # raw_data = np.genfromtxt(file, skip_header=4, delimiter='\t')
    # raw_data = np.delete(raw_data, 0, axis=0)

    raw_x = raw_data[:, 0]
    raw_y = raw_data[:, 1]

    # Process the data
    max_xloc = np.argmax(raw_x)

    process_x = raw_data[max_xloc::,0]
    process_y = raw_data[max_xloc::,1]

    y_max = np.nanmax(raw_y)
    low_y = limits[0]*y_max
    upper_y = limits[1]*y_max
    new_index = np.argwhere((process_y > low_y) & (process_y < upper_y))
    slice_x = process_x[new_index].flatten()
    slice_y = process_y[new_index].flatten()


    return (slice_x,slice_y)


if __name__ == "__main__":
    limits = (0.2, 0.95)

    # cwd = os.getcwd()
    # dir = 'POCO-P11'
    # path = os.path.join(cwd,'raw', dir)
    #
    # for file in os.listdir(path):
    #     if file.endswith(".txt"):
    #         fpath = os.path.join(path, file)
    #         print(fpath)
    #         read_files(fpath, limits, output_file_name='p' + file)
    #         continue
    #     else:
    #         continue
