"""
Initalized path range
"""

import numpy as np

class Pathrange_limits():

    def __init__(self,path):
        self.path = path
        self.rangeS02 = (np.linspace(5, 95, 91) * 0.01)
        self.rangeE0 = (np.linspace(-100, 100, 201) * 0.01) # <- e0, for everything
        # self.rangeE0_large = (np.linspace(-600, 600, 1201) * 0.01)  # <- Larger range B
        self.rangeSigma2 = (np.linspace(1, 15, 15) * 0.001) # <- should be separate
        self.rangeDeltaR = (np.linspace(-10, 10, 21) * 0.01)  # <-
        self.rangeS02 = np.insert(self.rangeS02,0,0)


    def get_path(self):
        return self.path

    def getrange_S02(self):
        return self.rangeS02

    def getrange_E0(self):
        return self.rangeE0

    def getrange_Sigma2(self):
        return self.rangeSigma2

    def getrange_DeltaR(self):
        return self.rangeDeltaR

    # Modify parameters of each objects
    def mod_S02(self,S02range):
        self.rangeS02 = S02range

    def mod_E0(self,E0range):
        self.rangeE0 = E0range

    def mod_Sigma2(self,Sigma2range):
        self.rangeSigma2 = Sigma2range

    def mod_DeltaR(self,DeltaRrange):
        self.DeltaR = DeltaRrange
