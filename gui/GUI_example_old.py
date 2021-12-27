"""
Authors    Matthew Adas, Andy Lau, Dr Jeffrey Terry
Email      madas@hawk.iit.edu
Version    0.0
Date       May 30 2020

Please start the program within the "gui" directory, or "select file" buttons won't start the user in EXAFS.
"Select Directory" button starts the user in EXAFS/path_files/Cu
"""
from tkinter import ttk,Tk,N,W,E,S,StringVar,IntVar,DoubleVar,BooleanVar,Checkbutton,NORMAL,DISABLED,filedialog,messagebox
from tkinter.font import Font
import matplotlib
import matplotlib.pyplot as plt
import os
# from larch.xafs import autobk, xftf
# import larch
# from larch.io import read_ascii
# from larch import Interpreter
# from larch.xafs import pre_edge
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
# from larch.1 import xafsplots

# Goal: to ensure the gui is running in EXAFS/gui no matter what the current working directory is
"""if os.getcwd() != "gui":
    os.chdir("gui")
else:
    pass
"""

def create_file(*args):
    """
    create the final .ini files
    """
    try:
        multiple_csv_val = bool(multiple_csv.get()) #add this to gui
        csv_val = str(csv_file.get())
        csv_series_val = str(csv_series.get()) # something may be missing
        out_val = str(output_file.get())
        feff_folder_val = str(feff_folder.get())
        population_val = int(population.get())
        num_gen_val = int(num_gen.get())
        best_sample_val = int(best_sample.get())
        lucky_few_val = int(lucky_few.get())
        chance_mutation_val = int(chance_of_mutation.get())
        original_chance_mutation_val = int(original_chance_of_mutation.get())
        chance_mutation_e0_val = int(chance_of_mutation_e0.get())
        mutated_opt_val = int(mutated_options.get())
        individual_path_val = bool(individual_path.get())
        path_range_val = int(path_range.get())
        path_list_val = str(path_list.get()) #make an array, not str? Try to get into this here: https://stackoverflow.com/questions/50023635/python-convert-string-to-an-array
        path_optimize_val = bool(path_optimize.get())
        steady_state_exit_val = bool(steady_state_exit.get())
        kmin_val = float(k_min.get())
        kmax_val = float(k_max.get())
        kweight_val = float(k_weight.get())
        deltak_val = float(delta_k.get())
        rbkg_val = float(r_bkg.get())
        bkgkw_val = float(bkg_kw.get())
        bkgkmax_val = float(bkg_kmax.get())
        print_graph_val = bool(print_graph.get())
        num_output_paths_val = bool(num_output_paths.get())

        #file contents
        inputs = ("[INPUTS] \nmultiple = {mult} \ncsv file = {csv} \ncsv series = {series} \noutput file = {out} \nfeff file = {feff}"
        .format(mult = str(multiple_csv_val), csv=str(csv_val), series=str(csv_series_val), out=str(out_val), feff=str(feff_folder_val)))
        XAFS_file = open("XAFS.ini", "w+") #creates a file in the working directory
        XAFS_file.write(str(inputs))
        XAFS_file.close()

        populations = ("\n\n[POPULATIONS] \npopulation = {pop} \nnum_gen = {numgen} \nbest_sample = {best} \nlucky_few = {luck}"
        .format(pop=str(population_val), numgen=str(num_gen_val), best=str(best_sample_val), luck=str(lucky_few_val)))
        XAFS_file = open("XAFS.ini", "a+") #appends to the same file just created
        XAFS_file.write(str(populations))

        mutations = ("\n\n[MUTATIONS] \nchance_of_mutation = {chance} \noriginal_chance_of_mutation = {original} \nchance_mutation_e0 = {e0} \nmutated_options = {opt}"
        .format(chance=str(chance_mutation_val), original=str(original_chance_mutation_val), e0=str(chance_mutation_e0_val), opt=str(mutated_opt_val)))
        XAFS_file = open("XAFS.ini", "a+") #appends to the same file just created
        XAFS_file.write(str(mutations))

        paths = ("\n\n[PATHS] \nindividual_path = {tf}  \npath_range = {range} \npath_list = {list} \npath_optimize = {optimize}"
        .format(tf=str(individual_path_val), range=str(path_range_val), list=str(path_list_val), optimize=str(path_optimize_val)))
        XAFS_file.write(str(paths))

        larch_paths = ("\n\n[LARCH PATHS] \nkmin = {min} \nkmax = {max} \nkweight = {weight} \ndeltak = {delk} \nrbkg = {rb} \nbkgkw = {bk} \nbkgkmax = {bmax}"
        .format(min=kmin_val, max=kmax_val, weight=kweight_val, delk=deltak_val, rb=rbkg_val, bk=bkgkw_val, bmax=bkgkmax_val))
        XAFS_file.write(str(larch_paths))

        outputs = ("\n\n[OUTPUTS] \nprint_graph = {pg}  \nnum_output_paths = {outpath} \nsteady_state_exit = {steady}"
        .format(pg=print_graph_val, outpath=num_output_paths_val, steady=steady_state_exit_val))
        XAFS_file.write(str(outputs))

        XAFS_file.close()

        #graph

        if print_graph_val == True:
            x = [1,2,3,4]       #values for x-axis
            y = [1,4,9,16]      #values for y-axis
            plt.plot(x, y)
            plt.ylabel('x^2 integers')
            plt.title('nothing spectacular')
            plt.show()
        else:
            pass

    except ValueError:
        pass

def on_closing():
    """
    on closeing function
    """
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.quit()


##########################################################################################
############################### Plotting functions for background #######################################
##########################################################################################
def plot_bkg_1():
    """
    Plotting functions for general
    """
    try:
        # csv_val = str(csv_file.get())
        csv_val = str(csv_file.get())
    except ValueError:
        pass
    mylarch = Interpreter()
    # print(larch.__version__)
    cu = read_ascii(csv_val)

    try:
        cu.chi
    except AttributeError:
        autobk(cu, rbkg=float(r_bkg.get()), kweight=float(bkg_kw.get()), kmax=float(bkg_kmax.get()), _larch = mylarch)

    fig, ax1 = plt.subplots(1, 1,figsize=(6,4))
    ax1.plot(cu.energy,cu.mu,'b',label='mu')
    ax1.plot(cu.energy,cu.bkg,'k',label='background')
    ax1.set_xlabel('Energy (eV)')
    ax1.set_ylabel('$\mu$(E)')
    ax1.legend()
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=bkg_tab)
    canvas.draw()
    canvas.get_tk_widget().grid(column=3,row=1,rowspan=7,sticky=W+E+N)

def plot_bkg_2():
    try:
        csv_val = str(csv_file.get())
    except ValueError:
        pass
    mylarch = Interpreter()
    print(larch.__version__)
    cu = read_ascii(csv_val)

    try:
        cu.chi
    except AttributeError:
        autobk(cu, rbkg=float(r_bkg.get()), kweight=float(bkg_kw.get()), kmax=float(bkg_kmax.get()), _larch = mylarch)

    xftf(cu.k, cu.chi, kmin=float(k_min.get()), kmax=float(bkg_kmax.get()), dk=4,
    window='hanning',kweight=float(bkg_kw.get()), group=cu, _larch=mylarch)

    fig, ax2 = plt.subplots(1, 1,figsize=(6,4))
    ax2.plot(cu.k, cu.chi*cu.k**2,'b',label='K Space')
    ax2.set_xlabel('$k$ (Å$^{-1}$)')
    ax2.set_ylabel('$k\chi(k)$')
    ax2.legend()
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=bkg_tab)
    canvas.draw()
    canvas.get_tk_widget().grid(column=3,row=1,rowspan=7,sticky=W+E+N)

def plot_bkg_3():
    try:
        csv_val = str(csv_file.get())
    except ValueError:
        pass
    mylarch = Interpreter()
    print(larch.__version__)
    cu = read_ascii(csv_val)

    try:
        cu.chi
    except AttributeError:
        autobk(cu, rbkg=float(r_bkg.get()), kweight=float(bkg_kw.get()), kmax=float(bkg_kmax.get()), _larch = mylarch)

    xftf(cu.k, cu.chi, kmin=float(k_min.get()), kmax=float(bkg_kmax.get()), dk=4,
    window='hanning',kweight=float(bkg_kw.get()), group=cu, _larch=mylarch)

    fig, ax3 = plt.subplots(1, 1,figsize=(6,4))
    ax3.plot(cu.r, cu.chir_mag,'b',label='R Space')
    ax3.set_xlabel('$r$ (Å$^{-1}$)')
    ax3.set_ylabel('$\chi(r)$')
    ax3.legend()
    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=bkg_tab)
    canvas.draw()
    canvas.get_tk_widget().grid(column=3,row=1,rowspan=7,sticky=W+E+N)

#function for individual path checkbutton
def activate_check():
    """
    check function of individual paths
    """
    if individual_path.get() == 1:
        path_list_entry.config(state=NORMAL)
        path_range_entry.config(state=DISABLED)
    elif individual_path.get() == 0:
        path_list_entry.config(state=DISABLED)
        path_range_entry.config(state=NORMAL)

def select_multiple():
    """
    #ATM selecting the Multiple csv/xmu checkbutton does not change what the button does. I thought the button would now do nothing, but my logic may be faulty. Let's make the select_csv_series and see what happens when we use it here
    """
    if multiple_csv.get() == 1:
        csv_series_button.config(state=NORMAL)
        csv_series_entry.config(state=NORMAL)
        csv_file_button.config(state=DISABLED)
        csv_file_entry.config(state=DISABLED)
        bkg_csv_file_button.config(state=DISABLED)
        bkg_csv_file_entry.config(state=DISABLED)
    elif multiple_csv.get() == 0:
        csv_series_button.config(state=DISABLED)
        csv_series_entry.config(state=DISABLED)
        csv_file_button.config(state=NORMAL)
        csv_file_entry.config(state=NORMAL)
        bkg_csv_file_button.config(state=NORMAL)
        bkg_csv_file_entry.config(state=NORMAL)

#functions for file paths
def select_csv_file():
    os.chdir("..") #change the working directory from gui to EXAFS
    file_name =  filedialog.askopenfilename(initialdir = os.getcwd(), title = "Choose xmu/csv", filetypes = (("xmu files", "*.xmu"),("csv files","*.csv"),("all files","*.*")))
    # change line (ANDY)
    csv_file.set(file_name)
    os.chdir("gui")
    return file_name

def select_output_file():
    """
    Select output files
    """
    os.chdir('..')
    file_name = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Choose", filetypes = ((".ini files","*.ini"),("all files","*.*")))
    # change line (ANDY)
    output_file.set(file_name)
    os.chdir("gui")
    return file_name

def select_feff_folder():
    """
    Select feff folder for inputs
    """
    os.chdir('..')
    os.chdir("path_files/Cu")
    folder_name = filedialog.askdirectory(initialdir = os.getcwd(), title = "Select folder")
    #, filetypes = ((".dat files","*.dat"),("all files","*.*"))
    feff_folder.set(folder_name)
    os.chdir('..')
    os.chdir('..')
    os.chdir("gui")
    return folder_name

def select_csv_series():
    """
    Select CSV series file
    # https://stackoverflow.com/questions/16790328/open-multiple-filenames-in-tkinter-and-add-the-filesnames-to-a-list
    """
    os.chdir("..") #change the working directory from gui to EXAFS
    file_name =  filedialog.askopenfilenames(initialdir = os.getcwd(), title = "Choose multiple xmu OR csv", filetypes = (("xmu files", "*.xmu"),("csv files","*.csv"),("all files","*.*")))
    # change line (ANDY)
    csv_series.set(file_name)
    os.chdir("gui")
    return file_name
"""     #import tkFileDialog
        #import re
        #ff = tkFileDialog.askopenfilenames()
        #filez = re.findall('{(.*?)}', ff)
        import Tkinter,tkFileDialog
        root = Tkinter.Tk()
        filez = tkFileDialog.askopenfilenames(parent=root,title='Choose a file')
"""


###################
####    GUI    ####
###################
root = Tk()
root.title("XAFS AI GUI")
root.geometry("950x500")
mainframe = ttk.Notebook(root, padding="5")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
#bkg_tab = ttk.Notebook(mainframe, padding="5")
#bkg_tab.grid(column=2, row=2, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.resizable(False, False)

#specify standard font for OS independence
entryFont = Font(family="TkTextFont", size=11)
labelFont = Font(family="TkMenuFont", size=11)
s = ttk.Style()
s.configure('my.TButton', font=labelFont)

#Inputs (rows 1-3)
multiple_csv = BooleanVar(root, False)
csv_file = StringVar(root, "Please choose a file")
csv_series = StringVar(root, "Please choose files")
output_file = StringVar(root, "Please choose a file")
feff_folder = StringVar(root, "Please choose a directory")

#Populations (column 1)
population = IntVar(root, 2000)
num_gen = IntVar(root, 100)
best_sample = IntVar(root, 400)
lucky_few = IntVar(root, 200)

#Mutations (column 3)
chance_of_mutation = IntVar(root, 20)
original_chance_of_mutation = IntVar(root, 20)
chance_of_mutation_e0 = IntVar(root, 20)
mutated_options = IntVar(root, 0)

#Paths (column 1, row 8-10)
individual_path = BooleanVar(root, False)
path_range = IntVar(root, 20)
path_list = StringVar(root)
path_optimize = BooleanVar(root, True)

#Larch Paths (column 5)
k_min = DoubleVar(root, 2.5)
k_max = DoubleVar(root, 20)
k_weight = DoubleVar(root, 2)
delta_k = DoubleVar(root, 0.05)
r_bkg = DoubleVar(root, 1.0)
bkg_kw = DoubleVar(root, 2.0)
bkg_kmax = DoubleVar(root, 15)

#Outputs
print_graph = BooleanVar(root, False)
num_output_paths = BooleanVar(root, True)
steady_state_exit = BooleanVar(root, True)

#create tabs
input_tab = ttk.Frame(mainframe, style='my.TButton')
population_tab = ttk.Frame(mainframe)
path_tab = ttk.Frame(mainframe)
mutation_tab = ttk.Frame(mainframe)
larch_tab = ttk.Frame(mainframe)
bkg_tab = ttk.Frame(mainframe)
#bkg_tab_2 = ttk.Frame(bkg_tab)
#bkg_tab_3 = ttk.Frame(bkg_tab)
output_tab = ttk.Frame(mainframe)

mainframe.add(input_tab, text="Inputs")
mainframe.add(population_tab, text="Populations")
mainframe.add(path_tab, text="Paths")
mainframe.add(mutation_tab, text="Mutations")
mainframe.add(larch_tab, text="Larch Paths")
mainframe.add(bkg_tab, text="Background Plots")
#bkg_tab.add(bkg_tab_2, text="Background 2")
#bkg_tab.add(bkg_tab_3, text="Background 3")
mainframe.add(output_tab, text="Outputs")

# label widgets
# inputs tab (tab 1)
array_input = ["multiple csv/xmu", "csv/xmu file", "csv/xmu series", "output file", "feff_folder"]
for x in array_input:
    ttk.Label(input_tab, text=x, font=labelFont).grid(column=1, row=array_input.index(x)+2, sticky=W)

# populations tab (tab 2)
array_populations = ["population", "num gen", "best sample", "lucky few"]
for x in array_populations:
    ttk.Label(population_tab, text=x, font=labelFont).grid(column=1, row=array_populations.index(x)+2, sticky=W)

# paths tab (tab 3)
array_paths = ["individual path", "path range", "path list", "path optimize"]
for x in array_paths:
    ttk.Label(path_tab, text=x, font=labelFont).grid(column=1, row=array_paths.index(x)+2, sticky=W)

# mutations tab (tab 4)
array_mutations = ["chance of mutation", "original chance of mutation", "chance of mutation e0", "mutated options"]
for x in array_mutations:
    ttk.Label(mutation_tab, text=x, font=labelFont).grid(column=1, row=array_mutations.index(x)+2, sticky=W)

# larch paths tab (tab 5)
array_larch = ["kmin", "kmax", "kweight", "delta k", "r bkg", "bkg kw", "bkg kmax"]
for x in array_larch:
    ttk.Label(larch_tab, text=x, font=labelFont).grid(column=1, row=array_larch.index(x)+2, sticky=W)

# background plot tab (tab 6)
array_bkg = ["r bkg", "bkg kw", "bkg kmax"]
for x in array_bkg:
    ttk.Label(bkg_tab, text=x, font=labelFont).grid(column=1, row=array_bkg.index(x)+1, sticky=W)

# outputs tab (tab 7)
array_outputs = ["print graph", "num output paths", "steady state exit"]
for x in array_outputs:
    ttk.Label(output_tab, text=x, font=labelFont).grid(column=1, row=array_outputs.index(x)+2, sticky=W)

# entry widgets
# inputs tab
#multiple_csv_checkbutton =
csv_file_entry = ttk.Entry(input_tab, textvariable=csv_file, font=entryFont)
csv_file_entry.grid(column=2, row=3, sticky=(W, E))
csv_series_entry = ttk.Entry(input_tab, textvariable=csv_series, font=entryFont)
csv_series_entry.grid(column=2, row=4, sticky=(W, E))
csv_series_entry.config(state=DISABLED)
output_file_entry = ttk.Entry(input_tab, textvariable=output_file, font=entryFont)
output_file_entry.grid(column=2, row=5, sticky=(W, E))
feff_folder_entry = ttk.Entry(input_tab, textvariable=feff_folder, font=entryFont)
feff_folder_entry.grid(column=2, row=6, sticky=(W, E))

# populations tab
population_entry = ttk.Entry(population_tab, width=7, textvariable=population, font=entryFont)
population_entry.grid(column=2, row=2, sticky=W)
num_gen_entry = ttk.Entry(population_tab, width=7, textvariable=num_gen, font=entryFont)
num_gen_entry.grid(column=2, row=3, sticky=W)
best_sample_entry = ttk.Entry(population_tab, width=7, textvariable=best_sample, font=entryFont)
best_sample_entry.grid(column=2, row=4, sticky=W)
lucky_few_entry = ttk.Entry(population_tab, width=7, textvariable=lucky_few, font=entryFont)
lucky_few_entry.grid(column=2, row=5, sticky=W)

# paths tab --- row 2 for path_tab is a checkbutton, therefore in the "button" section of this code
path_range_entry = ttk.Entry(path_tab,  width=7, textvariable=path_range, font=entryFont)
path_range_entry.grid(column=2, row=3, sticky=W)
path_list_entry = ttk.Entry(path_tab,  textvariable=path_list, font=entryFont) #need to change path list so that it can handle a list
path_list_entry.grid(column=2, row=4, sticky=(W,E))
path_list_entry.config(state=DISABLED)

# mutations tab
mut_list = list(range(101))
chance_of_mutation_entry = ttk.Combobox(mutation_tab, width=7, textvariable=chance_of_mutation, values=mut_list, state="readonly")
chance_of_mutation_entry.grid(column=4, row=2, sticky=W)
original_chance_of_mutation_entry = ttk.Combobox(mutation_tab, width=7, textvariable=original_chance_of_mutation, values=mut_list, state="readonly")
original_chance_of_mutation_entry.grid(column=4, row=3, sticky=W)
chance_of_mutation_e0_entry = ttk.Combobox(mutation_tab, width=7, textvariable=chance_of_mutation_e0, values=mut_list, state="readonly")
chance_of_mutation_e0_entry.grid(column=4, row=4, sticky=W)
mutated_options_drop_list = ttk.Combobox(mutation_tab, width=2, textvariable=mutated_options, values=[1,2,3], state="readonly")
mutated_options_drop_list.grid(column=4, row=5, sticky=W)

# larch paths tab
k_min_entry = ttk.Entry(larch_tab, width=7, textvariable=k_min, font=entryFont)
k_min_entry.grid(column=2, row=2, sticky=W)
k_max_entry = ttk.Entry(larch_tab, width=7, textvariable=k_max, font=entryFont)
k_max_entry.grid(column=2, row=3, sticky=W)
k_weight_entry = ttk.Entry(larch_tab, width=7, textvariable=k_weight, font=entryFont)
k_weight_entry.grid(column=2, row=4, sticky=W)
delta_k_entry = ttk.Entry(larch_tab, width=7, textvariable=delta_k, font=entryFont)
delta_k_entry.grid(column=2, row=5, sticky=W)
r_bkg_entry = ttk.Entry(larch_tab, width=7, textvariable=r_bkg, font=entryFont)
r_bkg_entry.grid(column=2, row=6, sticky=W)
bkg_kw_entry = ttk.Entry(larch_tab, width=7, textvariable=bkg_kw, font=entryFont)
bkg_kw_entry.grid(column=2, row=7, sticky=W)
bkg_kmax_entry = ttk.Entry(larch_tab, width=7, textvariable=bkg_kmax, font=entryFont)
bkg_kmax_entry.grid(column=2, row=8, sticky=W)

# background tab
r_bkg_entry = ttk.Entry(bkg_tab, textvariable=r_bkg, font=entryFont)
r_bkg_entry.grid(column=2, row=1, sticky=W)
bkg_kw_entry = ttk.Entry(bkg_tab, textvariable=bkg_kw, font=entryFont)
bkg_kw_entry.grid(column=2, row=2, sticky=W)
bkg_kmax_entry = ttk.Entry(bkg_tab, textvariable=bkg_kmax, font=entryFont)
bkg_kmax_entry.grid(column=2, row=3, sticky=W)
bkg_csv_file_entry = ttk.Entry(bkg_tab, textvariable=csv_file, font=entryFont) #exactly the same as in the inputs tab, but needs to be a separate widget for the tab it's in
bkg_csv_file_entry.grid(column=2, row=4, sticky=W)
bkg_csv_series_entry = ttk.Entry(bkg_tab, textvariable=csv_series, font=entryFont)
bkg_csv_series_entry.grid(column=2, row=5, sticky=W)
bkg_csv_series_entry.config(state=DISABLED)

##### blank plot
fig, ax1 = plt.subplots(1, 1,figsize=(6,4))
plt.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=bkg_tab)
canvas.draw()
canvas.get_tk_widget().grid(column=3,row=1,rowspan=7,sticky=W+E+N)

# button widgets
# inputs tab
multiple_csv_checkbutton = ttk.Checkbutton(input_tab, var=multiple_csv, command=select_multiple)
multiple_csv_checkbutton.grid(column=2, row=2, sticky=W)
csv_file_button = ttk.Button(input_tab, text="Choose", command=select_csv_file, style='my.TButton')
csv_file_button.grid(column=3, row=3, sticky=W)
csv_series_button = ttk.Button(input_tab, text="Choose", command=select_csv_series, style='my.TButton')
csv_series_button.grid(column=3, row=4, sticky=W)
csv_series_button.config(state=DISABLED)
output_file_button = ttk.Button(input_tab, text="Choose", command=select_output_file, style='my.TButton')
output_file_button.grid(column=3, row=5, sticky=W)
feff_folder_button = ttk.Button(input_tab, text="Select Folder", command=select_feff_folder, style='my.TButton')
feff_folder_button.grid(column=3, row=6, sticky=W)

# paths tab
individual_path_checkbutton = ttk.Checkbutton(path_tab, var=individual_path, command=activate_check)
individual_path_checkbutton.grid(column=2, row=2, sticky=W)
path_optimize_checkbutton = ttk.Checkbutton(path_tab, var=path_optimize)
path_optimize_checkbutton.grid(column=2, row=5, sticky=W)

# background tab
bkg_csv_file_button = ttk.Button(bkg_tab, text="Choose xmu/csv", command=select_csv_file, style='my.TButton')
bkg_csv_file_button.grid(column=1, row=4, sticky=W)
bkg_csv_series_button = ttk.Button(bkg_tab, text="Choose series", command=select_csv_series, style='my.TButton')
bkg_csv_series_button.grid(column=1, row=5, sticky=W)
bkg_csv_series_button.config(state=DISABLED)
plot_bkg_1_button = ttk.Button(bkg_tab, text="Plot Background and Mu", command=plot_bkg_1, style='my.TButton')
plot_bkg_1_button.grid(column=2, row=6, sticky=W)
plot_bkg_2_button = ttk.Button(bkg_tab, text="Plot K-Space", command=plot_bkg_2, style='my.TButton')
plot_bkg_2_button.grid(column=2, row=7, sticky=E)
plot_bkg_3_button = ttk.Button(bkg_tab, text="Plot R-Space", command=plot_bkg_3, style='my.TButton')
plot_bkg_3_button.grid(column=2, row=8, sticky=E)

#create_button = ttk.Button(output_tab, text="Create File", command=create_file, style='my.TButton')
#create_button.grid(column=4, row=2, sticky=E)

# outputs tab
print_graph_checkbutton = ttk.Checkbutton(output_tab, var=print_graph)
print_graph_checkbutton.grid(column=2, row=2, sticky=W)
num_output_checkbutton = ttk.Checkbutton(output_tab, var=num_output_paths)
num_output_checkbutton.grid(column=2, row=3, sticky=W)
steady_state_exit_checkbutton = ttk.Checkbutton(output_tab, var=steady_state_exit)
steady_state_exit_checkbutton.grid(column=2, row=4, sticky=W)
### empty labels for spacing
for i in range(4):
    ttk.Label(output_tab, text = "                                            ").grid(row=i+2,column=3)


#ttk.Label(bkg_tab, text="                                            ").grid(row=1,column=3)

# outputs tab
create_button = ttk.Button(output_tab, text="Create File", command=create_file, style='my.TButton')
create_button.grid(column=4, row=2, sticky=E)
run_button = ttk.Button(output_tab, text="Run", style='my.TButton')
run_button.grid(column=4, row=3, sticky=E)
stop_button = ttk.Button(output_tab, text="Stop", style='my.TButton')
stop_button.grid(column=4, row=4, sticky=E)

root.protocol("WM_DELETE_WINDOW",on_closing)
root.mainloop() # tells Tk to enter its event loop, needed to make everything run
