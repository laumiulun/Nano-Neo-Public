from .helper import *
from .import_lib import *
from .ini_parser import *
from .pathObj import OliverPharr
from .individual import Individual
from .pathrange import Pathrange_limits
from .nano_neo_data import NanoIndent_Data

"""
Author: Andy Lau
"""

# Setup some default constraints
MAX = 3.40282e+38
MIN = 1.17549e-38

class NANO_GA:

    def initialize_params(self,verbose = False):
        """
        Initialize Parameters
        """
        # print("Initialize Parameters")
        self.intervalK = 0.05
        self.tol = np.finfo(np.float64).resolution

    def initialize_variable(self):
        """
        Initalize variables
        """
        self.genNum = 0
        self.nChild = 4
        self.globBestFit = [0,0]
        self.currBestFit = [0,0]
        self.bestDiff = 9999e11
        self.bestBest = 999999999e11
        self.diffCounter = 0

        self.pathDictionary = {}
        self.data_file = data_file
        self.data_cutoff = data_cutoff
        # Paths
        self.npaths = npaths
        self.fits = fits

        # Populations
        self.npops = size_population
        self.ngen = number_of_generation
        self.steady_state = steady_state

        # Mutation Parameters
        self.mut_opt = mutated_options
        self.mut_chance = chance_of_mutation
        # self.mut_chance_e0 = chance_of_mutation_e0

        # Crosover Parameters
        self.n_bestsam = int(best_sample*self.npops*(0.01))
        self.n_lucksam = int(lucky_few*self.npops*(0.01))

        # Time related
        self.time = False
        self.tt = 0

    def initialize_file_path(self,i=0):
        """
        Initalize file paths for each of the file first
        """
        self.base = os.getcwd()
        self.output_path = os.path.join(self.base,output_file)
        self.check_output_file(self.output_path)
        self.log_path = os.path.splitext(copy.deepcopy(self.output_path))[0] + ".log"
        self.check_if_exists(self.log_path)

        # Initialize logger
        self.logger = logging.getLogger('')
        # Delete handler
        self.logger.handlers=[]
        file_handler = logging.FileHandler(self.log_path,mode='a+',encoding='utf-8')
        stdout_handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

        self.logger.setLevel(logging.INFO)
        self.logger.info(banner())

    def check_if_exists(self,path_file):
        """
        Check if the directory exists
        """
        if os.path.exists(path_file):
            os.remove(path_file)
        # Make Directory when its missing
        path = pathlib.Path(path_file)
        path.parent.mkdir(parents=True, exist_ok=True)

    def check_output_file(self,file):
        """
        check if the output file for each of the file
        """
        file_base= os.path.splitext(file)[0]
        self.check_if_exists(file)
        self.file = file

        self.file_initial = open(self.output_path,"a+")
        self.file_initial.write("Gen,TPS,FITTNESS,CURRFIT,CURRIND,BESTFIT,BESTIND\n")  # writing header
        self.file_initial.close()

        file_data = os.path.splitext(file)[0] + '_data.csv'
        self.check_if_exists(file_data)
        self.file_data = file_data


    def initialize_range(self,i=0,BestIndi=None):
        """
        Initalize range

        To Do list:
            Initalize range will be difference for each paths depend if the run are
            in series, therefore the ranges will self-adjust
        """

        # data = np.genfromtxt(self.data_file,delimiter=',',skip_header=1)
        self.data_obj = NanoIndent_Data(self.data_file)

        self.data_obj.pre_processing(limits=(self.data_cutoff[0],self.data_cutoff[1]))
        self.x_slice = self.data_obj.get_slice_data()[:,0]
        self.y_slice = self.data_obj.get_slice_data()[:,1]

        self.pars_range = {
            'A_range': A_range,
            'hf_range': hf_range,
            'm_range': m_range
        }

    def create_range(self,value,percentage,dt,prec):
        """
        Create delta to calculate the ranges
        """
        minus = round(value - percentage*value,prec)
        plus = round(value + percentage*value,prec)
        range = np.arange(minus,plus+dt,dt)
        return range

    def generateIndividual(self):
        """
        Generate singular individual
        """

        ind = Individual(self.npaths,self.pars_range)
        return ind

    def generateFirstGen(self):
        self.Populations=[]

        for i in range(self.npops):
            self.Populations.append(self.generateIndividual())

        self.eval_Population()
        self.globBestFit = self.sorted_population[0]

    # @profile
    def fitness(self,indObj):
        """
        Evaluate fitness of a individual
        """

        loss = 0
        Individual = indObj.get_func()

        yTotal = np.zeros(len(self.x_slice))

        for i,paths in enumerate(Individual):

            y = paths.get_func(self.x_slice)

            yTotal += y

        for j in range(len(self.x_slice)):

            loss = loss + (yTotal[j]*self.x_slice[j]**2 - self.y_slice[j]* self.x_slice[j]**2 )**2

        return loss

    def eval_Population(self):
        """
        Evalulate populations
        """

        score = []
        populationPerf = {}
        self.nan_counter = 0
        for i,individual in enumerate(self.Populations):

            temp_score = self.fitness(individual)
            # Calculate the score, if encounter nan, discard and generate new individual later
            if np.isnan(temp_score):
                self.nan_counter +=1
            else:
                score.append(temp_score)
                populationPerf[individual] = temp_score
        self.sorted_population = sorted(populationPerf.items(), key=operator.itemgetter(1), reverse=False)

        self.currBestFit = self.sorted_population[0]

        return score


    def next_generation(self):
        """
        Calculate next generations

        """
        self.st = time.time()
        # ray.init()
        self.logger.info("---------------------------------------------------------")
        self.logger.info(datetime.datetime.fromtimestamp(self.st).strftime('%Y-%m-%d %H:%M:%S'))
        self.logger.info(f"{bcolors.BOLD}Gen: {bcolors.ENDC}{self.genNum+1}")

        self.genNum += 1

        # Evaluate Fittness
        score = self.eval_Population()
        self.bestDiff = abs(self.globBestFit[1]-self.currBestFit[1])
        if self.currBestFit[1] < self.globBestFit[1]:
            self.globBestFit = self.currBestFit


        with np.printoptions(precision=5, suppress=True):
            self.logger.info("Different from last best fit: " +str(self.bestDiff))
            self.logger.info(bcolors.BOLD + "Best fit: " + bcolors.OKBLUE + str(self.currBestFit[1]) + bcolors.ENDC)
            self.logger.info("Best fit combination:\n" + str(np.asarray(self.sorted_population[0][0].get())))
            self.logger.info(bcolors.BOLD + "History Best: " + bcolors.OKBLUE + str(self.globBestFit[1]) +bcolors.ENDC)
            self.logger.info("NanCounter: " + str(self.nan_counter))
            self.logger.info("History Best Indi:\n" + str(np.asarray(self.globBestFit[0].get())))

        nextBreeders = self.selectFromPopulation()
        self.logger.info("Number of Breeders: " + str(len(self.parents)))
        self.logger.info("DiffCounter: " + str(self.diffCounter))
        self.logger.info("Diff %: " + str(self.diffCounter / self.genNum))
        self.logger.info("Mutation Chance: " + str(self.mut_chance))
        self.mutatePopulation()
        self.createChildren()


        self.et = timecall()
        self.tdiff = self.et - self.st
        self.tt = self.tt + self.tdiff
        self.logger.info("Time: "+ str(round(self.tdiff,5))+ "s")

    def mutatePopulation(self):
        """
        # Mutation operators
        # 0 = original: generated a new versions:
        # 1 = mutated every genes in the total populations
        # 2 = mutated genes inside population based on secondary probability

        # TODO:
            options 2 and 3 needs to reimplmented
        """
        self.nmutate = 0

        if self.mut_opt  == 0:
            # Rechenberg mutation
            if self.genNum > 20:
                if self.bestDiff < 0.1:
                    self.diffCounter += 1
                else:
                    self.diffCounter -= 1
                if (abs(self.diffCounter)/ float(self.genNum)) > 0.2:
                    self.mut_chance += 0.5
                    self.mut_chance = abs(self.mut_chance)
                elif (abs(self.diffCounter) / float(self.genNum)) < 0.2:
                    self.mut_chance -= 0.5
                    self.mut_chance = abs(self.mut_chance)


        for i in range(self.npops):
            if random.random()*100 < self.mut_chance:
                self.nmutate += 1
                self.Populations[i] = self.mutateIndi(i)

        self.logger.info("Mutate Times: " + str(self.nmutate))


    def mutateIndi(self,indi):
        """
        Generate new individual during mutation operator
        """
        if self.mut_opt == 0:
            # Create a new individual with Rechenberg
            newIndi = self.generateIndividual()
        # Random pertubutions
        if self.mut_opt == 1:
            # Random Pertubutions
            self.Populations[indi].mutate_paths(self.mut_chance)
            newIndi = self.Populations[indi]
            # Mutate every gene in the Individuals
        if self.mut_opt == 2:
            # initalize_variable:
            self.nmutate_success = 0
            og_indi = copy.deepcopy(self.Populations[indi])
            og_score = self.fitness(og_indi)
            mut_indi = copy.deepcopy(self.Populations[indi])
            mut_indi.mutate_paths(self.mut_chance)
            mut_score = self.fitness(mut_indi)

            with np.errstate(divide='raise', invalid='raise'):
                try:
                    t_bot = (np.log(1-(self.genNum/self.ngen)+self.tol))
                except FloatingPointError:
                    print(self.genNum)
                    print(self.ngen)
                    print(1-(self.genNum/self.ngen))
                    t_bot = (np.log(1-(self.genNum/self.ngen)+self.tol))

            T = - self.bestDiff/t_bot
            if mut_score < og_score:
                self.nmutate_success = self.nmutate_success + 1;
                newIndi = mut_indi
            elif np.exp(-(mut_score-og_score)/(T+self.tol)) > np.random.uniform():

                self.nmutate_success = self.nmutate_success + 1;
                newIndi = mut_indi
            else:
                newIndi = og_indi

        if self.mut_opt == 3:
            def delta_fun(t,delta_val):
                rnd = np.random.random()
                return delta_val*(1-rnd**(1-(t/self.ngen))**5)

            og_indi = copy.deepcopy(self.Populations[indi])
            og_data = og_indi.get_var()
            for i,path in enumerate(og_data):
                print(i,path)
                arr = np.random.randint(2,size=3)
                for j in range(len(arr)):
                    new_path = []
                    val = path[j]
                    if arr[j] == 0:
                        UP = self.pathrange_Dict[i].get_lim()[j+1][1]
                        del_val = delta_fun(self.genNum,UP-val)
                        val = val + del_val
                    if arr[j] == 1:
                        LB = self.pathrange_Dict[i].get_lim()[j+1][0]
                        del_val = delta_fun(self.genNum,val-LB)
                    new_path.append(val)
                self.Populations[indi].set_path(i,new_path[0],new_path[1],new_path[2])
        if self.mut_opt == 4:
            newIndi = self.generateIndividual(self.bestE0)
        return newIndi

    def selectFromPopulation(self):
        self.parents = []

        select_val = np.minimum(self.n_bestsam,len(self.sorted_population))
        self.n_recover = 0
        if len(self.sorted_population) < self.n_bestsam:
            self.n_recover = self.n_bestsam - len(self.sorted_population)
        for i in range(select_val):
            self.parents.append(self.sorted_population[i][0])

    def crossover(self,individual1, individual2):
        """
        Uniform Cross-Over, 50% percentage chance
        """
        child = self.generateIndividual()

        for i in range(self.npaths):
            individual1_path = individual1.get_path(i)
            individual2_path = individual2.get_path(i)

            temp_path = []
            for j in range(3):
                if np.random.randint(0,2) == True:
                    temp_path.append(individual1_path[j])
                else:
                    temp_path.append(individual2_path[j])

            child.set_path(i,temp_path[0],temp_path[1],temp_path[2])

        return child

    def createChildren(self):
        """
        Generate Children
        """
        self.nextPopulation = []
        # --- append the breeder ---
        for i in range(len(self.parents)):
            self.nextPopulation.append(self.parents[i])
        # print(len(self.nextPopulation))
        # --- use the breeder to crossover
        # print(abs(self.npops-self.n_bestsam)-self.n_lucksam)

        for i in range(abs(self.npops-self.n_bestsam)-self.n_lucksam):
            par_ind = np.random.choice(len(self.parents),size=2,replace=False)
            child = self.crossover(self.parents[par_ind[0]],self.parents[par_ind[1]])
            self.nextPopulation.append(child)
        # print(len(self.nextPopulation))

        for i in range(self.n_lucksam):
            self.nextPopulation.append(self.generateIndividual())
        # print(len(self.nextPopulation))

        for i in range(self.n_recover):
            self.nextPopulation.append(self.generateIndividual())

        # for i in range(self.nan_counter):
        #     self.nextPopulation.append(self.generateIndividual())

        random.shuffle(self.nextPopulation)
        self.Populations = self.nextPopulation

    def run_verbose_start(self):
        self.logger.info("-----------Inputs File Stats---------------")
        self.logger.info(f"{bcolors.BOLD}File{bcolors.ENDC}: {self.data_file}")
        self.logger.info(f"{bcolors.BOLD}File Type{bcolors.ENDC}: {self.data_obj._ftype}")
        self.logger.info(f"{bcolors.BOLD}File{bcolors.ENDC}: {self.output_path}")
        self.logger.info(f"{bcolors.BOLD}Population{bcolors.ENDC}: {self.npops}")
        self.logger.info(f"{bcolors.BOLD}Num Gen{bcolors.ENDC}: {self.ngen}")
        self.logger.info(f"{bcolors.BOLD}Mutation Opt{bcolors.ENDC}: {self.mut_opt}")
        self.logger.info("-------------------------------------------")

    def run_verbose_end(self):
        self.logger.info("-----------Output Stats---------------")
        # self.logger.info(f"{bcolors.BOLD}Total)
        self.logger.info(f"{bcolors.BOLD}Total Time(s){bcolors.ENDC}: {round(self.tt,4)}")
        self.logger.info("-------------------------------------------")

    def run(self):
        self.run_verbose_start()
        self.historic = []
        self.historic.append(self.Populations)

        for i in range(self.ngen):
            temp_gen = self.next_generation()
            self.output_generations()

        self.run_verbose_end()
        # test_y = self.export_paths(self.globBestFit[0])
        # plt.plot(self.data_obj.get_raw_data()[:,0],self.data_obj.get_raw_data()[:,1],'b-.')
        # plt.plot(self.x_slice,self.y_slice,'o--',label='data')
        # plt.plot(self.x_slice,test_y,'r--',label='model')
        # plt.legend()
        # plt.show()

    def export_paths(self,indObj):
        area_list=[]
        Individual = indObj.get_func()

        yTotal = np.zeros(len(self.x_slice))
        plt.figure()
        for i,paths in enumerate(Individual):
            y = paths.get_func(self.x_slice)

            yTotal += y
            # area = np.trapz(y.flatten(),x=self.x_slice.flatten())
            # component = paths.get_func(self.x_slice).reshape(-1,1)

            # area_list.append(area)

        Total_area = np.sum(area_list)
        return yTotal

    def output_generations(self):
        """
        Output generations result into two files
        """
        try:
            f1 = open(self.file,"a")
            f1.write(str(self.genNum) + "," + str(self.tdiff) + "," +
                str(self.currBestFit[1]) + "," + str(self.currBestFit[0].get()) +"," +
                str(self.globBestFit[1]) + "," + str(self.globBestFit[0].get()) +"\n")
        finally:
            f1.close()
        try:
            f2 = open(self.file_data,"a")
            write = csv.writer(f2)
            bestFit = self.globBestFit[0].get()
            for i in range(self.npaths):
                write.writerow((bestFit[i][0], bestFit[i][1], bestFit[i][2]))
            f2.write("#################################\n")
        finally:
            f2.close()

    def __init__(self):
        """
        Steps to Initalize EXAFS
            EXAFS
        """
        # initialize params
        self.initialize_params()
        # variables
        self.initialize_variable()
        # initialze file paths
        self.initialize_file_path()
        # initialize range
        self.initialize_range()
        # Generate first generation
        self.generateFirstGen()

        self.run()

def main():
    NANO_GA()
