# This module defines functions of condition evolution path of a simple example.

import geatpy as ea

import numpy as np

class Condition_evolution(ea.Problem):

    def __init__(self,Net,Loaders,N_elayers,Bp_loop,cycles=20,epochs=10):
        '''
        This class defines the training path of condition evolution.  

        The fitness of SEA is computed under each combanation of electronic configurations with several
        Bp loops, so that the electronic network need to be defined first.

        Net:  SEA model

        Loaders: A list including different data loaders in pytorch.

        N_elayers: The number of electronic layers.

        Bp_loop: The function of Bp loop that is defined in the main program.

        cycles: The number of alternating training cycles.

        The hyperparameters could be rewritten with different requirements.

        '''
        # Electonic layers configurations
        self.Net=Net
        self.Loaders=Loaders
        self.n_dataflows=len(Loaders)
        self.n_elayers=N_elayers
        self.Bp_loop=Bp_loop
        self.cycle=cycles
        self.epochs=epochs

        # Evolution parameters
        name = 'CE'  
        M = 1  
        maxormins = [-1] * M  
        Dim = self.n_elayers*self.n_dataflows 
        varTypes = [1]*Dim  
        lb = [0]*Dim  
        ub = [2]*Dim  
        lbin = [1]*Dim 
        ubin = [1]*Dim 

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def compute_fitness(self,C):
        '''
        In SEA, the fitness evaluates the output of multiple dataflows under various conditions.

        The BP path will loop several times (loop) under the condition (c), each time including training dataflows 
        several times (epochs or even a sample).

        Note that the evolved conditions use integers to represent different physical configurations 
        to simplify computing process in this example after the pre-evaluation of the REU.
        In the experimental test process, the conditions include real electronic parameters, such as voltage.

        *Capital "C" refers to a population, each individual of the pop representing the electronic layer conditions 
        of dataflows. 
        
        '''
        
        pop_error=[]
        idv_error=[]
        j=1
        for idv in C:
            print('---------------Idv {}-----------------'.format(j))
            reset_weights(self.Net)
            j+=1
            for i in range(self.cycle):
                for j in range(self.n_dataflows):
                    idv_error.append(self.Bp_loop(self.Net,idv[j*self.n_elayers:(j+1)*self.n_elayers],self.Loaders[j],self.epochs)[-1])
            pop_error.append(idv_error[-self.n_dataflows:])    

        final_error=np.array(pop_error)

        fitness=1/np.log(1+(np.mean(final_error,axis=1))) 
        # The mean was replaced with adjustable weights in the real experiment 

        return fitness.reshape(-1,1)

    def evalVars(self,Vars):
        print(Vars)
        f= self.compute_fitness(Vars)
        return f

def reset_weights(model):
    '''
    In CE loop, each cycle refreshes the weights(more exactly, modulation parameters) to maintain comparability.
    '''
    for layer in model.children():   
       if hasattr(layer, 'reset_parameters'):     
           layer.reset_parameters() 

def CE_loop(Net,Loaders,N_elayers,Bp_loop,cycles=10,epochs=10):
    '''
    This function defines the evolution algorithm and optimization.

    The hyperparameters could be rewritten for different requirements.

    '''
    CE_problem=Condition_evolution(Net,Loaders,N_elayers,Bp_loop,cycles=cycles,epochs=epochs)

    CE_model=ea.soea_SEGA_templet(CE_problem,
                                ea.Population(Encoding='BG',NIND=10),
                                MAXGEN=10, 
                                maxTrappedCount=8,
                                logTras=1)

    CE_res=ea.optimize(CE_model,seed=None,verbose=True,drawing=0,drawLog=False)

    CE_trace=CE_model.trace

    return CE_res,CE_trace
