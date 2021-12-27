import sys,glob,re,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from sklearn.preprocessing import MinMaxScaler
import re
import fnmatch
import copy,csv,itertools
from nano_neo_data import NanoIndent_Data

def sort_fold_list(dirs):
    fold_list = list_dirs(dirs)
    fold_list.sort(key=natural_keys)
    return fold_list
## Human Sort
def list_dirs(path):
    return [os.path.basename(x) for x in filter(
        os.path.isdir, glob.glob(os.path.join(path, '*')))]


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def OliverPharr(x,A,hf,m):
    return A*(x-hf)**m

class NanoIndent_Analysis:
    def __init__(self,dirs,npaths,params):
        self.dirs = dirs
        self.npaths = npaths
        # need to remove this
        self.paths = np.arange(1,self.npaths+1)
        self.base = params['base']
        self.params = params
        # print(self.base)
        # print(os.path.join(self.base,params['file']))
        file = os.path.join(self.base,params['file'])

        self.data_obj = NanoIndent_Data(file)
        self.data_obj.pre_processing(params['data_cutoff'])

        self.x_raw = self.data_obj.get_raw_data()[:,0]
        self.y_raw = self.data_obj.get_raw_data()[:,1]
        self.x_slice = self.data_obj.get_slice_data()[:,0]
        self.y_slice = self.data_obj.get_slice_data()[:,1]


    def normalizer(X,min,max):
        X_std = (X-X.min)

    def extract_data(self,plot_err=False):
        """
        Extract data value using array data
        """
        full_mat = self.read_result_files(self.dirs)
        # print(full_mat)
        bestFit,err = self.construct_bestfit_err_mat(full_mat,self.npaths,plot_err)
        best_Fit = np.mean(full_mat,axis=0).reshape(-1,3).round(6)
        print("Score:")
        print(best_Fit)
        self.bestFit = bestFit
        self.err = err
        self.err_full = self.err.reshape((-1,3))

        self.bestFit_mat = best_Fit

        # print("Not Normalized:")
        self.best_Fit_n = copy.deepcopy(best_Fit)
        # self.best_Fit_n[:,0] -= self.scaler.min_
        # self.best_Fit_n[:,0] /= self.scaler.scale_
        # print(self.best_Fit_n)
        # self.x_model = np.arange
        self.y_model = self.fit_2_arr(self.best_Fit_n)

    def score(self,verbose=False):
        loss =self.fitness(self.bestFit_mat)
        print('Fitness Score (Chi2):',loss)
        # print('Fitness Score (ChiR2):', loss/(len(self.x_raw)-4*self.npaths))
        # self.fwhm(verbose=verbose)
        # self.cal_area(verbose=verbose)
        # print(self.err_full)

    def calculate_parameters(self,verbose=True):
        # print(self.params[])
        try:
            self.params['calibrations']
        except NameError:
            raise NameError("Missing Calibrations parameters")

        dx = self.x_model[-1] - self.x_model[-2]
        dy = self.y_model[-1] - self.y_model[-2]
        dydx = dy/dx

        b = self.y_model[-1] -dydx * self.x_model[-1]

        # Calculate linear
        x_linear_min = self.x_slice[-1]
        x_linear_max = (self.data_obj.max_y - b)/dydx
        self.x_linear = np.arange(x_linear_min,x_linear_max)
        self.y_linear = np.zeros(len(self.x_linear))
        for i in range(len(self.x_linear)):
            self.y_linear[i] = (dydx*self.x_linear[i] + b)

        for i in range(len(self.y_linear)):
            k = 0
            if self.y_linear[i] > 0:
                k = i
                break

        self.x_linear = self.x_linear[k::]
        self.y_linear = self.y_linear[k::]

        #
        h_max = self.data_obj.max_x
        P = self.data_obj.max_y
        S = dydx

        hc = h_max - 0.75 * P/S
        A = self.calculate_A(hc)
        H = (P/A) *10**6
        E = np.pi **(0.5) *S / (2*A**(0.5))*10**3
        calculated_result = {
            'Stiffness': np.round(dydx,3),
            'Max Depth': h_max,
            'Max Load': P,
            'Area': A,
            'H': H,
            'E' :E
        }
        self.params['result'] = calculated_result
        if verbose:
            print("Stiffness S = dP/dh: ", np.round(dydx, 3))
            print('Maximum Depth (nm):' + str(h_max))
            print('Maximum Load (uN):' + str(P))
            print("Area: " + str(np.round(A, 5)) + ' nm^2')
            print("H: " + str(np.round(H, 5)) + ' MPa')
            print("E: " + str(np.round(E, 5)) + ' GPa')
            
    def calculate_A(self,hc):
        const = 2
        A = 0
        for i in range(len(self.params['calibrations'])):
            A += self.params['calibrations']['C' + str(i)] *hc **const
            const = const/2

        return A



    def fitness(self,bestFit):
        """
            Fitness Calculation


        To do: Needto normalized first, since it is not right
        """
        loss = 0

        self.area = np.zeros(self.npaths)
        self.individual_export_arr= np.zeros((self.npaths,len(self.x_slice)))
        yTotal = np.zeros(len(self.x_slice))
        # print(self.bestFit)
        for i in range(len(bestFit)):

            # y = voigt_shape.voigt_fuc(self.x_raw, bestFit[i,0], bestFit[i,1], bestFit[i,2], bestFit[i,3])
            y = OliverPharr(self.x_slice,bestFit[i,0],bestFit[i,1],bestFit[i,2])
            self.individual_export_arr[i,:] = y
            # self.individual_export_arr[i,:] -= self.scaler.min_
            # self.individual_export_arr[i,:] /= self.scaler.scale_
            self.area[i] = np.trapz(y,self.x_slice)

            yTotal += y

        for j in range(len(self.x_slice)):
            loss = loss + (yTotal[j]*self.x_slice[j]**2 - self.y_slice[j]* self.x_slice[j]**2 )**2
        return loss

    def read_result_files(self,folder):
        r"""
        read result files (helper function)

        Inputs:
            folder (str): str pointing the files locations
        """
        num_path = self.npaths
        full_mat = []
        files = []
        folder = self.dirs
        files_opt = []
        files_opt_data = []
        print(folder)
        for r, d, f in os.walk(folder):
            f.sort(key = natural_keys)
            for file in f:
                if fnmatch.fnmatch(file,'*_data.csv'):
                    files.append(os.path.join(r, file))
        files.sort(key=natural_keys)

        for i in range(len(files)):
    #         file = os.path.join(folder,'test_' + str(i) + '_data.csv')
            file = files[i]
    #         print(file)
            try:
                os.path.exists(file)
                gen_csv = np.genfromtxt(file,delimiter=',')[-1]

    #             print(gen_csv.shape)
                # gen_csv_unflatten = gen_csv.reshape((-1,4*num_path))
                gen_csv_unflatten = gen_csv.reshape((-1,3*num_path))
                if i == 0:
                    full_mat = gen_csv_unflatten
                full_mat = np.vstack((full_mat,gen_csv_unflatten))
            except OSError:
                print(" " + str(i) + " Missing")
                pass

        return full_mat

    def generate_labels(self):
        label=[]
        amp_label = []
        center_label = []
        sigma_label = []
        gamma_label = []

        for i in range(1,self.npaths+1):
            label.append('amp_' + str(i))
            amp_label.append('amp_' + str(i))

            label.append('center_' + str(i))
            center_label.append('center_' + str(i))

            label.append('sigma_' + str(i))
            sigma_label.append('sigma_' + str(i))

            label.append('gamma_' + str(i))
            gamma_label.append('gamma_' + str(i))

        return label,amp_label,center_label,sigma_label,gamma_label

    def construct_bestfit_err_mat(self,full_mat,npaths,plot=False):
        r"""
        Construct the average best fit matrix using the sum of the files, and
        generate the corresponding labels using the paths provided.

        (Helper function)

        """
        full_mat_var_cov = np.cov(full_mat.T)
        full_mat_diag = np.diag(full_mat_var_cov)#/self.scaler.scale_
        err = np.sqrt(full_mat_diag)

        labels = self.generate_labels()
        bestFit = np.mean(full_mat,axis=0)
        if plot:
            plt.figure(figsize=(7,5))
            plt.xticks(np.arange(len(full_mat_diag)),labels[0],rotation=70);
            plt.bar(np.arange(len(full_mat_diag)),np.sqrt(full_mat_diag))

        return bestFit,err

    def convert_to_str(self,val,prec):
        prec_val = '{:.' + str(prec) + 'f}'
        return prec_val.format(round(val,prec))

    def convert_label(self,select_bestfit_r):
        temp_label=''
        for j in range(len(select_bestfit_r)):
            temp_label+=select_bestfit_r[j][0]
            temp_label+='-'
        temp_label=temp_label[:-1]
        return temp_label

    def cal_err_prec(self,val_arr):
        prec_arr = []
        for i in range(len(val_arr)):
            dist = abs(int(np.log10(abs(val_arr[i])))-2)
            prec_arr.append(dist)
        return prec_arr

    def fit_2_arr(self,bestFit):
        loss = 0
        # print(np.min(self.x_slice),np.max(self.x_slice))
        self.x_model = np.arange(np.min(self.x_slice),np.max(self.x_slice))
        yTotal = np.zeros(len(self.x_model))

        for i in range(len(bestFit)):

            y = OliverPharr(self.x_model,bestFit[i,0],bestFit[i,1],bestFit[i,2])

            yTotal += y
        return yTotal

    def stack_export_data(self,exp_x,exp_y,fit_x,fit_y):
            exp_data = np.vstack((exp_x,exp_y)).T
            fit_data = np.vstack((fit_x,fit_y)).T

            return exp_data,fit_data

    def write_dat_csv(self,writer,data):
        for i in range(len(data)):
            writer.writerow((data[i,:]))

    def export_bestFit(self,exp_x,exp_y,fit_x,fit_y,name='bestFit.csv',header_base='Sample'):

        with open(name, mode='w', newline='', encoding='utf-8') as write_file:

            exp_data,fit_data = self.stack_export_data(exp_x,exp_y,fit_x,fit_y)

            writer = csv.writer(write_file, delimiter=',')

            writer.writerow(['data_' + header_base + '_x','data_' + header_base+'_y'])
            self.write_dat_csv(writer,exp_data)
            writer.writerow('')
            writer.writerow(['fit_' + header_base + ')_x','fit_' + header_base+'_y'])
            self.write_dat_csv(writer,fit_data)

    def export_individual(self,exp_x,exp_y,fit_x,fit_y,export_path,name='Individual.csv',header_base='Sample'):
        with open(name, mode='w', newline='', encoding='utf-8') as write_file:

            exp_data,fit_data = self.stack_export_data(exp_x,exp_y,fit_x,fit_y)
            bg_data = np.vstack((self.x_raw,self.background-np.nanmax(self.background))).T

            writer = csv.writer(write_file, delimiter=',')
            writer.writerow(['data_' + header_base + '_x','data_' + header_base+'_y'])
            self.write_dat_csv(writer,exp_data)
            writer.writerow('')
            writer.writerow(['fit_' + header_base + '_x','fit_' + header_base+'_y'])
            self.write_dat_csv(writer,fit_data)

            writer.writerow('')
            writer.writerow(['bg_' + header_base + '_x','bg_' + header_base+'_y'])
            self.write_dat_csv(writer,bg_data)

            for i in range(self.npaths):
                path_header = ['path_'+ str(i)+ '_' + header_base +'_x','path_'+str(i)+'_' + header_base+'_y']
                writer.writerow(path_header)
                arr = np.stack((self.x_raw,export_path[i,:]))
                # print(arr.shape)
                self.write_dat_csv(writer,arr.T)
                writer.writerow('')

    def export_files(self,header='test',dirs='',igor=False):
        self.header = header
        file_name_k = dirs  + 'bestFit_' + header + '.csv'
        file_name_best = dirs  + header + '_bestFit_err.csv'
        file_name_ind = dirs + 'Individual_Fit_' + header + '.csv'

        # print(self.fit_2_arr(self.best_Fit_n))
        y_fit = self.fit_2_arr(self.best_Fit_n)
        self.export_bestFit(self.x_raw,self.y_bg,self.x_raw,y_fit,name=file_name_k,header_base=header)
        # exit()
        self.export_individual(self.x_raw,self.y_bg,self.x_raw,y_fit,self.individual_export_arr,name=file_name_ind,header_base=header)
        if igor:
            self.export_igor_individual()

    def plot_data(self,title='Test'):
        plt.rc('xtick', labelsize='12')
        plt.rc('ytick', labelsize='12')
        # plt.rc('font',size=30)
        plt.rc('figure',autolayout=True)
        plt.rc('axes',titlesize=12,labelsize=12)
        # plt.rc('figure',figsize=(7.2, 4.45))
        plt.rc('axes',linewidth=1)

        plt.rcParams["font.family"] = "Times New Roman"
        fig,ax = plt.subplots(1,1,figsize=(6,4.5))
        # ax.plot(self.x_raw,self.y_raw,'ko-',linewidth=1,label='Data')
        ax.plot(self.x_raw,self.y_raw,'b--',linewidth=1,label='Data')
        # ax.plot(self.x_slice,self.y_slice,'r--',linewidth=1.2,label='Slice Data')
        ax.plot(self.x_model,self.y_model,'r',linewidth=1,label='Fit')
        ax.plot(self.x_linear,self.y_linear,'--',color='tab:purple')
        ax.legend()


    def export_igor_individual(self,file_paths='export_ind.ipf'):
        r"""
        export files in igor plotting for individuals, must be ran after
        the indiviudal methods

        Input:
            file_paths (str): locations for the igor plot script
        """
        # Displace all data
        f = open(file_paths,"w")
        f.write('•Display data_' + self.header + '_y vs data_' + \
            self.header + '_x;')
        f.write('\n')
        f.write('•AppendToGraph fit_' + self.header + '_y vs fit_' +\
            self.header + '_x;' )
        f.write('\n')
        f.write('•AppendToGraph bg_' + self.header + '_y vs bg_' +\
            self.header + '_x;' )
        f.write('\n')

        for i in range(self.npaths):
            f.write('•AppendToGraph path_' + str(i) + '_'+ self.header+ \
                '_y vs path_' + str(i) +'_'+ self.header + '_x;' )
            f.write('\n')

        # f.write('•SetAxis bottom *,11');f.write('\n')

        ## Offset
        # offset first two is designated number

        # f.write('•ModifyGraph offset(path_' + str(0)+ '_'+self.header + '_y)={0,5}');f.write('\n')
        # f.write('•ModifyGraph offset(path_' + str(1) + '_' + self.header + '_y)={0,10}');f.write('\n')
        # f.write('•ModifyGraph offset(path_' + str(2) + '_' + self.header + '_y)={0,12.5}');f.write('\n')

        # offset the rest
        # for i in range(self.npaths):
        #     curr_paths = str(i)
        #     f.write('•ModifyGraph offset(path_' + curr_paths + '_' +self.header + '_y)={0,' + str(15 + i) + '}' )
        #     f.write('\n')
        # f.write(r'•Label left "k\\S2\\M χ(k) (Å\\S-2\\M)";DelayUpdate');f.write('\n')
        f.write(r'•Label bottom "Binding Energy (eV)"');f.write('\n')
        f.write('•ModifyGraph lsize(fit_' + self.header  + '_y)=2');f.write('\n')
        f.write('•ModifyGraph lsize(bg_' + self.header  + '_y)=2');f.write('\n')

        for i in range(self.npaths):
            f.write('•ModifyGraph lsize(path_' + str(i) + '_' + self.header + '_y)=2')
            f.write('\n')
        ## Legend
        # \r  - new line
        self.adjust_color(f)
        # self.create_legend(f)

        f.write('•ModifyGraph mode(data_' + self.header  + '_y)=3');f.write('\n')

    def create_legend(self,f):
        legend_header = R"""•Legend/C/N=text0/J "Test\rk\\S2\\Mχ(k)\rTest_Detail\r\r\\s("""
        legend_1 = 'data_' + self.header + r"_chi2) Data\r\\s(fit_" + self.header + r'_chi2) Fit";'
        legend = legend_header + legend_1
        f.write(legend); f.write('\n')
        for i in range(self.npaths):
            if int(self.nleg_arr[i]) > 2:
                addition = ' MS '
            else:
                addition = ' '
            paths_arr = str(i) + '_' + self.header + '_y) Path ' + str(self.paths[i]) + addition + self.label_arr[i] + r'";DelayUpdate'
            f.write('•AppendText/N=text0 "\\s(path_'+ paths_arr)
            f.write('\n')
    # Adjust the color using jet reverse color bar
    def adjust_color(self,f,color_map=plt.cm.jet_r):
        r"""
        adjust color for legend

        Input:
            f (str): files output name and locations
            color_map (cmp): matplotlib.cm.cmap objects, default to
                plt.cm.jet_r
        """

        x = np.linspace(0,1,self.npaths)
        color = [color_map(i) for i in x]
        test = 65535;
        for i in range(len(color)):
            color[i] = (int(test*color[i][0]),int(test*color[i][1]),int(test*color[i][2]))
        # Change to X and Y for data
        f.write('•ModifyGraph rgb(fit_' + self.header + '_y)=(0,0,0)');f.write('\n')
        for i in range(len(color)):
            f.write('•ModifyGraph rgb(path_' + str(i) + '_' + self.header + '_y)=' + str(color[i]))
            f.write('\n')

    def jet_r(self,x):

        return plt.cm.jet_r(x)


    def construct_latex_table(self):
        R"""
        Construct simple latex table

        Todo:
        Change this from printout to files instead.
        """
        # err_full = self.construct_full_err(self.err)
        # self.err_full = self.err.reshape((-1,4))
        # print(self.paths)
        # print(self.bestFit_mat)
        self.latex_table(self.best_Fit_n,self.err_full)


        # self.nleg_arr = nleg_arr
        # self.label_arr = label_arr

    # def cal_err_prec(self,val_arr):
    #     prec_arr = []
    #     for i in range(len(val_arr)):
    #         dist = abs(int(np.log10(abs(val_arr[i])))-1)
    #         prec_arr.append(dist)
    #     return prec_arr




    def latex_table(self,best_Fit,err_full):
        nleg_arr =[]
        label_arr = []
        # for i in range(self.npaths):
        #     label_arr.append(self.convert_label(self.bestFit_mat[i,6]))
        #     nleg_arr.append(str(int(best_Fit_r[i,5])))
        paths = self.paths
        best_Fit_r = np.array(best_Fit)
        print(R"""\floatsetup[table]{capposition=top}
            \begin{table}[]
                \centering
                    \footnotesize
                        \caption{test}
                            \begin{tabular}{cccccc}
                                \hline
                            \vspace{0.05in}
                                Path \# & $f_{G}$ & $f_{L}$ & $f_{V}$ & center & total area\\
                                \hline""")

        for i in range(len(paths)):
            err_arr = np.array([self.err_full[i,2],self.err_full[i,3],self.fv_err[i],self.err_full[i,1]])
            # print(err_arr)
            prec_arr = self.cal_err_prec(err_arr)
            # print(prec_arr)
            print("                        " + str(paths[i]) + " & " + \
                  self.convert_to_str(self.fg[i],prec_arr[0]) + r"$\pm$" + self.convert_to_str(err_full[i,2],prec_arr[0]) + " & " +
                  self.convert_to_str(self.fl[i],prec_arr[1]) + r"$\pm$" + self.convert_to_str(err_full[i,3],prec_arr[1]) + " & " +
                  self.convert_to_str(self.fv[i],prec_arr[2]) + r"$\pm$" + self.convert_to_str(self.fv_err[i],prec_arr[2]) + " & " +
                  self.convert_to_str(best_Fit_r[i,1],prec_arr[3]) + r"$\pm$" + self.convert_to_str(err_full[i,1],prec_arr[3]) + " & " +
                  self.convert_to_str(self.area[i],prec_arr[3]) + r"$\pm$" + self.convert_to_str(err_full[i,3],prec_arr[3]) + r"\\")

        print(r"""                        \hline
                            \end{tabular}
                            \label{Label}
            \end{table}""")
            # return
        # return

    def cal_dv_dL2(self,fL,fG):
        return 0.285797+0.750649*fL**2 /(fL**2 + 0.1*fG**2) + 0.926355*fL/(np.sqrt(fL**2+0.1*fG**2))

    def cal_dv_dG2(self,fL,fG):
        return 16*fL**2/(0.2166*fL**2+fG**2)

    def cal_err_fv(self,fL,fG,fV,err_fL,err_fG):
        return np.sqrt(self.cal_dv_dG2(fL,fG)*err_fG**2 + self.cal_dv_dL2(fL,fG)*err_fL**2)


    def fwhm(self,verbose=False):
        if verbose:
            print("N,Fg,Fl,Fv")
        self.fg = []
        self.fl = []
        self.fv = []
        self.fv_err = []
        for i in range(self.npaths):
            fg_temp= 2* self.best_Fit_n[i,2]*np.sqrt(2*np.log(2))
            fl_temp= 2* self.best_Fit_n[i,3]
            fv_temp= 0.5346 *fl_temp + np.sqrt(0.2166*fl_temp**2 + fg_temp**2)
            self.fg.append(fg_temp); self.fl.append(fl_temp); self.fv.append(fv_temp)
            fv_err = self.cal_err_fv(fl_temp,fg_temp,fv_temp,self.err_full[i,2],self.err_full[i,3])
            self.fv_err.append(fv_err)
            if verbose:
                print(i,fg_temp,fl_temp,fv_temp)
                print(fv_err)
    def cal_area(self,verbose=False):
        total_area = np.sum(self.area)
        area_per = self.area/total_area

        if verbose:
            print("Total Area:")
            print(self.area)
            print("Area Percent:")
            print(area_per)

    def cal_area_err(self,verbose=False):
        # Calculate the area given of a

        # amptiude_list = np.arange()
        area_average=[]
        self.area_std = []
        for i in range(self.npaths):

            # print(self.best_Fit_n[i,0]-self.err_full[i,0],self.best_Fit_n[i,0]+self.err_full[i,0])
            # print(self.bestFit_mat[i,0]-self.err_full[i,0],self.bestFit_mat[i,0]+self.err_full[i,0])
            # print(self.best_Fit_n[i,2]-self.err_full[i,2],self.best_Fit_n[i,2]+self.err_full[i,2])
            # print(self.best_Fit_n[i,3]-self.err_full[i,3],self.best_Fit_n[i,3]+self.err_full[i,3])

            amp_list = np.linspace(self.bestFit_mat[i,0]-self.err_full[i,0],self.bestFit_mat[i,0]+self.err_full[i,0],10)
            sigma_list = np.linspace(self.best_Fit_n[i,2]-self.err_full[i,2],self.best_Fit_n[i,2]+self.err_full[i,2],10)
            gamma_list = np.linspace(self.best_Fit_n[i,3]-self.err_full[i,3],self.best_Fit_n[i,3]+self.err_full[i,3],10)


            comb_list = np.vstack((amp_list,sigma_list,gamma_list))
            combination = list(itertools.product(*comb_list))
            # print(combination[:10])
            area_list = np.zeros(len(combination))
            for j in range(len(combination)):
                func = voigt_shape.voigt_fuc(self.x_raw, combination[j][0], self.bestFit_mat[i,1] , combination[j][1], combination[j][2])
                # print(len(self.scaler.inverse_transform(func.reshape(-1,1))))
                area = np.trapz(self.scaler.inverse_transform(func.reshape(1,-1)),self.x_raw)
                area_list[j] = area

            self.area_std.append(np.std(area_list))
            print("----")

    def verbose_result(self):
        self.fwhm(verbose=True)
        self.cal_area(verbose=True)
        # return (fg,fl,fv)
