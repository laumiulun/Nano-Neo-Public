import numpy as np
import matplotlib.pyplot as plt

class NanoIndent_Data:

    def __init__(self,file):
        self._file = file
        self._ptype = ['HYS','IMICRO','NAN']
        self._ftype = 'NAN'

        self.check_file_type()
        self._processed = False
        self._readed = False


    def check_line(self,line_number,string,accepted_type):
        if line_number == string:
            return True
        else:
            return False

    def check_file_type(self):
        confirm_ftype = False
        if self.check_Hysitron_file() == True:
            confirm_ftype == True
            return
        elif self.check_iMicro_file() == True:
            confirm_ftype == True
            return
        else:
            confirm_ftype == False
            return

    def check_Hysitron_file(self):
        checklist = []
        self.label = None
        with open(self._file,'r') as f:
            # Check the label for Hysitron
            lines = f.readlines()
            for i,line in enumerate(lines):
                if line.split(' ')[0] == 'Depth':
                    self.label = i
                    break
            if self.label == None:
                return False
            # Additional Check
            checklist.append(self.check_line(lines[self.label].split()[0],'Depth',0))
            checklist.append(self.check_line(lines[self.label].split()[1],'(nm)',0))
            if all(ele == True for ele in checklist):
                self._ftype = 'HYS'
                return True
            else:
                return False

    def check_iMicro_file(self):
        # Checking with formats
        with open(self._file,'r') as f:
            line = f.readlines()
            checklist = []
            checklist.append(self.check_line(line[0].split(',')[0],'Markers',1))
            checklist.append(self.check_line(line[1].split(',')[0],'',1))
            if all(ele == True for ele in checklist):
                self._ftype = 'IMICRO'
                return True
            else:
                return False

    def read_data(self):
        # Read the actual data
        if self._ftype == 'HYS':
            self._rawdata = np.genfromtxt(self._file,delimiter='\t',skip_header=self.label+1)

        elif self._ftype == 'IMICRO':
            with open(self._file,'r') as f:
                lines = f.readlines()
                # subtract two lines for iMicro
                n_lines = len(lines) - 2
                self._rawdata = np.zeros((n_lines,3))
                std_dis = float(lines[2].split(',')[1])
                std_force = float(lines[2].split(',')[2])
                for i,line in enumerate(lines):
                    if i > 1:
                        raw_line = line.split(',')
                        self._rawdata[i-2,:] = [float(raw_line[1])-std_dis,float(raw_line[2])-std_force,float(raw_line[3])]
        else:
            # Assumes it is a CSV data set with no header
            self._rawdata = np.genfromtxt(self._file,delimiter=',',skip_header=0)

        self._readed = True

    def pre_processing(self,limits=(0.1,0.9),plot=False):

        if self._readed == False:
            self.read_data()

        self._data_x = self._rawdata[:,0]
        self._data_y = self._rawdata[:,1]
        # parameters
        self.max_xloc = np.nanargmax(self._rawdata[:,0])
        self.max_yloc = np.nanargmax(self._rawdata[:,1])

        self.max_x = self._rawdata[:,0][self.max_xloc]
        self.max_y = self._rawdata[:,1][self.max_yloc]

        process_x = self._rawdata[self.max_xloc::,0]
        process_y = self._rawdata[self.max_xloc::,1]

        delta = abs(process_y[0]-process_y[-1])
        lower_lim =  process_y[-1] + limits[0]*delta
        upper_lim = process_y[0] - (1-limits[1])*delta
#         print(lower_lim,upper_lim)

        new_index = np.argwhere((process_y > lower_lim) & (process_y < upper_lim))
#         print(len(new_index))
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
        plt.plot(self.get_raw_data()[:,0],self.get_raw_data()[:,1],'b--')
        plt.plot(self.get_slice_data()[:,0],self.get_slice_data()[:,1],'r--')
