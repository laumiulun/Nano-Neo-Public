import tkinter as tk
#from tkinter import Button, mainloop

from tkinter import ttk, Tk, N, W, E, S, StringVar, IntVar, DoubleVar, BooleanVar, Checkbutton, NORMAL, DISABLED, \
    scrolledtext, filedialog, messagebox, LabelFrame, Toplevel, END, TOP
from tkinter.font import Font
class App():
    def __init__(self):
        self.__version__ = 0.0
        self.root = Tk(className='Nano Neo GUI')
        # use a lambda expression to pass arg to callback in button commands
        tk.Button(self.root, text='start', command=lambda: self.callback(True)).pack()
        tk.Button(self.root, text='stop', command=lambda: self.callback(False)).pack()
    def callback(self, flag):
        message = "message"
        for i in range(10):
            message += str(i)

        global loop
        if flag is True:
            print(message)
            # start the after method to call the callback every 100ms
            loop = self.root.after(100, self.callback, True)
        else:
            # cancel the after method if the flag is False
            self.root.after_cancel(loop)



    def Run(self):
        """
        Run the code
        """
        self.root.mainloop()

root = App()
root.Run()
