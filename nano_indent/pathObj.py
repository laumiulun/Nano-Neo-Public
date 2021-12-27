import numpy as np
import random
"""
Author: Andy Lau
"""
def select_opt(def_range,alt_range):
    """
    Used for selecting ranges of using default range

    def_range = if inputs is selected
    alt_range = if inputs is not selected
    """
    if len(def_range) == 0:
        return_range = np.arange(alt_range[0],alt_range[1],alt_range[2])
    else:
        return_range = np.arange(def_range[0],def_range[1],def_range[2])
    return return_range

class OliverPharr:
    """
    Power-Law used for nano-indentation

    y = A(x-hf)^m

    """
    def __init__(self,path_range,pars_range):

        self.A_range = select_opt(pars_range['A_range'],(1e-6,5e-4,1e-7))
        self.hf_range = select_opt(pars_range['hf_range'],(0.001,1100,1))
        self.m_range = select_opt(pars_range['m_range'],(0.001,4,0.01))

        self.A = np.random.choice(self.A_range)
        self.h_f = np.random.choice(self.hf_range)
        self.m = np.random.choice(self.m_range)

    def get_A(self):
        return self.A

    def get_hf(self):
        return self.h_f

    def get_m(self):
        return self.m

    def get(self):
        return [self.A,self.h_f,self.m]

    def get_func(self,h):
        return self.A*(h-self.h_f)**self.m

    def set_A(self,A):
        self.A = A

    def set_hf(self,h_f):
        self.h_f = h_f

    def set_m(self,m):
        self.m = m

    def set(self,A,h_f,m):
        self.set_A(A)
        self.set_hf(h_f)
        self.set_m(m)

    def mutate_A(self,chance):
        if random.random()*100 < chance:
            self.A = np.random.choice(self.A_range)

    def mutate_hf(self,chance):
        if random.random()*100 < chance:
            self.h_f = np.random.choice(self.hf_range)

    def mutate_m(self,chance):
        if random.random()*100 < chance:
            self.m = np.random.choice(self.m_range)

    def mutate(self,chance):
        self.mutate_A(chance)
        self.mutate_hf(chance)
        self.mutate_m(chance)
class StraightLine:
    """
    Stright Line

    y = mx+b
    """
    def __init__(self,path_range,m,b):
        m_range = np.arange(0,100,1)
        b_range = np.arange(0,100,1)

        self.m = m
        self.b = b

    def get_m(self):
        return self.m

    def get_b(self):
        return self.b

    def get_func(self,x):
        return self.m * x + self.b
