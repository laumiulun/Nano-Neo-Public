import time
from .__init__  import __version__

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def timecall():
    return time.time()

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

def str_to_list(s):
    arr = [float(i) for i in list(s.split(","))]
    return arr

def norm(val):
    return np.linalg.norm(val)


def banner():
    """
    https://patorjk.com/software/taag/#p=display&h=1&v=1&f=Big%20Chief&t=Nano%20Neo
    """
    banner_str = ('''
                    Nano_Neo ver %s
_____________________________________________________________
    _     _                             _     _
    /|   /                              /|   /
---/-| -/------__-----__-----__--------/-| -/------__-----__-
  /  | /     /   )  /   )  /   )      /  | /     /___)  /   )
_/___|/_____(___(__/___/__(___/______/___|/_____(___ __(___/_
                     
    '''% __version__)

    return banner_str
