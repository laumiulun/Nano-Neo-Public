import numpy as np


class NanoIndent_Data:

    def __init__(self,file):
        self._file = file
        self._processed = False
        self._readed = False


    def read_data(self):
        self._rawdata = np.genfromtxt(self._file,delimiter='\t',skip_header=3)

        self._readed = True

    def pre_processing(self,limits=(10,10),plot=False):

        if self._readed == False:
            self.read_data()

        self._data_x = self._rawdata[:,0]
        self._data_y = self._rawdata[:,1]
        # parameters
        self.max_xloc = np.argmax(self._rawdata[:,0])
        self.max_yloc = np.argmax(self._rawdata[:,1])
        self.max_x = self._rawdata[:,0][self.max_xloc]
        self.max_y = self._rawdata[:,1][self.max_yloc]

        process_x = self._rawdata[self.max_xloc::,0]
        process_y = self._rawdata[self.max_xloc::,1]

        lower_lim = limits[0]*np.max(process_y)
        upper_lim = limits[1]*np.max(process_y)

        new_index = np.argwhere((process_y > lower_lim) & (process_y < upper_lim))

        self._slice_x = process_x[new_index]
        self._slice_y = process_y[new_index]

        self._slice_data = np.concatenate((self._slice_x,self._slice_y),axis=1)

        if plot:
            self.plot_data()


    def get_length(self):
        return len(self._slice_x)

    def get_raw_data(self):
        return self._rawdata

    def get_slice_data(self):
        return self._slice_data

    def plot_data(self):
        plt.plot(self.combined_data[:,0],self.combined_data[:,1],'--')
