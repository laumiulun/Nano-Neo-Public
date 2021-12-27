from multiprocessing import Process, Queue
import os
import signal
import subprocess
from multiprocessing import Pool
from tkinter import ttk, Tk, N, W, E, S, StringVar, IntVar, DoubleVar, BooleanVar, Checkbutton, NORMAL, DISABLED, \
    scrolledtext, filedialog, messagebox, LabelFrame, Toplevel, END, TOP

#proc_lst = []

def F(x):
    #global proc_lst
    proc = subprocess.Popen(x, shell=True, preexec_fn=os.setsid)
    #proc_lst.append(proc.pid)
    #proc_lst.append(os.getpgid(proc.pid))
    #print(proc_lst)
    print("PROC ID: ", proc.pid)
    print(type(proc.pid))
    # proc.wait()
    return proc.pid

class SeriesInstance():
    def __init__(self):
        self.root = Tk()
        #self.numbers = Queue()
        #for i in range(4):
         #   self.numbers.put('exec nano_neo -i test/test_nanoindent_w_Params.ini')
        self.numbers = ['exec nano_neo -i test/test_nanoindent_w_Params.ini', 'exec nano_neo -i test/test_nanoindent_w_Params.ini', 'exec nano_neo -i test/test_nanoindent_w_Params.ini']
        self.F = F
        self.proc_lst = []
        ttk.Button(self.root, text='start', command=self.start).pack()
        ttk.Button(self.root, text='stop', command=self.stop).pack()
        #semaphore = multiprocessing.Semaphore(1)
        self.p = Pool(maxtasksperchild=1)
        #self.p = multiprocessing.Semaphore(1)

    def run(self):
        self.root.mainloop()

    def start(self):
        #while not self.numbers.empty():
        #p = Process(target=F, args=(self.numbers,))
        #p.start()
        #self.proc_lst.append(p)
        #p.join()
        #p.close()
        self.proc_lst.append(self.p.map(self.F, self.numbers))
        print("Proc list inside start: ", self.proc_lst)
        self.p.join()
        self.p.close()
        #return self.proc_lst

    def stop(self):
        print(self.proc_lst)
        print("IN stop")
        print("IN STOP")
        print("IN stop")
        print("IN STOP")
        print("IN stop")
        print("IN STOP")
        print("IN stop")
        print("IN STOP")
        print("IN stop")
        print("IN STOP")
        print("IN stop")
        print("IN STOP")
        for each in self.proc_lst[0]:
            print("KILLING: ", each)
            os.kill(each, signal.SIGTERM)
        if hasattr(self, 'proc'):
            self.proc.kill()
        print("Attempting to kill pool:")
        self.p.close()
        self.p.terminate()
        self.p.join()

root = SeriesInstance()
root.run()
