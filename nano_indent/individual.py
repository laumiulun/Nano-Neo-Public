from .pathObj import OliverPharr

class Individual():
    def __init__(self,npaths,pars_range):
        self.npaths = npaths
        self.Population = [None]* self.npaths

        for i in range(self.npaths):
            self.Population[i] = OliverPharr(i,pars_range)

    def get(self):
        Population = []
        for i in range(self.npaths):
            Population.append(self.Population[i].get())
        return Population

    def get_func(self):
        Population = []
        for i in range(self.npaths):
            Population.append(self.Population[i])
        return Population

    def get_path(self,i):
        return self.Population[i].get()

    def mutate_paths(self,chance):
        for path in self.Population:
            path.mutate(chance)

    def verbose(self):
        """
        Print out the Populations
        """
        for i in range(self.npaths):
            self.Population[i].verbose()

    def set_path(self,i,A,h_f,m):
        self.Population[i].set(A,h_f,m)
