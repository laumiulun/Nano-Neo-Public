from tkinter import *
import time
from threading import *

# Create Object
root = Tk()
var = "bobthebuilder"
# Set geometry
root.geometry("400x400")

# use threading
def doSomething():
    t1=Thread(target=printanything)
    print("Hello")
    t1.start()

def printanything():
    global var
    var = "new"

def threading():
    # Call work function
    t1=Thread(target=work)
    t1.start()

# work function
def work():

    print("sleep time start")

    for i in range(10):
        global var
        print(var)
        time.sleep(1)

    print("sleep time stop")

# Create Button
Button(root,text="Click Me",command = threading).pack()
Button(root, text="Anythin", command=doSomething).pack()

# Execute Tkinter
root.mainloop()
