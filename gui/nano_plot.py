import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import NanoIndent_Analysis

class Analysis_plot:
    def __init__(self, frame):
        self.frame = frame
        self.fig = Figure(figsize=(3.5,3.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        # Create initial figure canvas
        self.canvas.get_tk_widget().grid(column=2, row=1, rowspan=13, columnspan=5, sticky="nsew",
                                         padx=5, pady=5)
        self.ax = self.fig.add_subplot(111)
        # create toolbar
        self.toolbarFrame = tk.Frame(master=self.frame)
        self.toolbarFrame.grid(column=2, row=0, rowspan=1, columnspan=5, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.params = {}

    def initial_parameters(self,dir,params,title):
        dir = str(dir.get())
        self.nano_analysis = NanoIndent_Analysis.NanoIndent_Analysis(dir,1,params)
        self.nano_analysis.extract_data(plot_err=False)
        self.nano_analysis.score()
        self.nano_analysis.calculate_parameters(verbose=False)
        self.fig.clf()
        self.nano_analysis.plot_data(title=title,fig_gui=self.fig)
        self.canvas.draw()
        return self.nano_analysis.get_params()
class Data_plot:
    def __init__(self, frame):
        self.frame = frame
        self.fig = Figure(figsize=(5.5, 2.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        # Create initial figure canvas
        self.canvas.get_tk_widget().grid(column=0, row=1, rowspan=4, columnspan=8, sticky="nsew",
                                         padx=5, pady=5)
        self.ax = self.fig.add_subplot(111)
        # create toolbar
        self.toolbarFrame = tk.Frame(master=self.frame)
        self.toolbarFrame.grid(column=3, row=0, rowspan=1, columnspan=5, sticky="w")
        toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)
        self.params = {}

    def initial_parameters(self,data_obj,title):
        """
        fileraw
        """
        self.data_obj = data_obj
        self.title = title

        self.xlabel = 'Depth: h'
        self.ylabel= 'Load: P'

    def plot_raw(self):
        self.ax.clear()
        raw_data = self.data_obj.get_raw_data()
        self.ax.plot(raw_data[:,0],raw_data[:,1], 'b.', label='Data')
        self.ax.legend()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_selected(self):
        self.ax.clear()
        slice_data = self.data_obj.get_slice_data()
        self.ax.plot(slice_data[:,0], slice_data[:,1], 'r.-', label='Selected Data')
        self.ax.legend()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        self.canvas.draw()

    def plot_raw_and_selected(self):
        self.ax.clear()
        raw_data = self.data_obj.get_raw_data()
        slice_data = self.data_obj.get_slice_data()
        self.ax.plot(raw_data[:,0], raw_data[:,1], 'b.', label='Data')
        self.ax.plot(slice_data[:,0], slice_data[:,1], 'r.-', label='Selected Data')
        # self.ax.axvspan(slice_data[0,0], slice_data[0,-1],color='red', alpha = 0.3)
        # Here add the best fit from the predictions
        self.ax.legend()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_title(self.title)
        self.fig.tight_layout()
        self.canvas.draw()
