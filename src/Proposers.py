import numpy as np
import random
import copy
import scipy.stats as st


class LayersProposer:

    def __init__(self, params: np.array = [2,[0,0],[0.5]], proposal_type = 'individual', lnk_sigma = 0, width_sigma = 0, dim_min: int = 1, dim_max: int = np.inf, std: float = 1, psplit: float = 0, pmerge: float = 0):
        self.params = params #Initial parameters, to be modified by the sampler. Structure: [dim, lnks, centers]
        self.proposal_type = proposal_type
        self.lnk_rvs = st.norm(scale=lnk_sigma)
        self.width_rvs = st.norm(scale=width_sigma)
        self.dim_min = dim_min #Minimum dimension allowed
        self.dim_max = dim_max #Maximum dimension allowed
        self.split_rvs =  st.norm(scale=std) #Probability distribution of birth padding params
        self.psplit = psplit #Probability of split
        self.pmerge = pmerge #Probability of merge
        self.p = 1 #Coeffient by which to correct acceptance probability
        self.detjacobian = 1 #Jacobian of the birth-death transformation

    def compute_detjacobian(self, x):
        self.detjacobian = abs(2*x)

    def move(self, new_params):
        self.params = copy.deepcopy(new_params)

    def perturb(self):
        dim, lnks, widths = self.params[0], copy.deepcopy(self.params[1]), copy.deepcopy(self.params[2])

        
        if self.proposal_type == 'individual':
            index = np.random.choice(dim)
            lnks[index] += self.lnk_rvs.rvs()
        
        elif self.proposal_type == 'simultaneous':
            for index in range(dim-1):
                lnks[index] += self.lnk_rvs.rvs()
            lnks[-1] += self.lnk_rvs.rvs()
        proposal = [dim, lnks, widths]
        return proposal, None
    
    def split(self):
        padding_params = [self.split_rvs.rvs(), np.random.uniform(0,1)]
        proposal = copy.deepcopy(self.params)
        proposal[2].append(1-sum(proposal[2]))
        index = random.randint(0,proposal[0]-1)
        self.compute_detjacobian(proposal[2][index], padding_params[1])
        #proposal[2].insert(index+1,round(proposal[2][index]*(1-padding_params[1])/0.05)*0.05)
        #proposal[2][index] = round(proposal[2][index]*padding_params[1]/0.05)*0.05
        proposal[2].insert(index+1,proposal[2][index]*(1-padding_params[1]))
        proposal[2][index] = proposal[2][index]*padding_params[1]
        proposal[1].insert(index+1,proposal[1][index]-padding_params[0])
        proposal[1][index] += padding_params[0]
        proposal[0] += 1
        proposal[2].pop()
        return proposal, padding_params, index

    def merge(self):
        proposal = copy.deepcopy(self.params)
        proposal[2].append(1-sum(proposal[2]))
        index = random.randint(0,proposal[0]-2)
        padding_params = [
                    0.5*(proposal[1][index]-proposal[1][index+1]),
                    proposal[2][index]/(proposal[2][index]+proposal[2][index+1])
                ]
        proposal[1][index] = 0.5*(proposal[1][index]+proposal[1][index+1])
        proposal[2][index] = proposal[2][index+1]+proposal[2][index]
        self.compute_detjacobian(proposal[2][index], padding_params[1])
        proposal[2].pop(index+1)
        proposal[1].pop(index+1)
        proposal[2].pop()
        proposal[0] -= 1
        
        return proposal, padding_params, index
    
    def propose_process(self):
        #0 split, 1 merge, 2 move
        if self.params[0]<self.dim_max and self.params[0]>self.dim_min:
            u = random.uniform(0,1)
            if u<self.psplit:
                return 0
            elif u<self.psplit+self.pmerge:
                return 1
            else:
                return 2
        elif self.params[0]>=self.dim_max and self.params[0]>self.dim_min: #Cannot split
            u = random.uniform(0,1-self.psplit)
            if u<self.pmerge:
                return 1
            else:
                return 2
        elif self.params[0]<self.dim_max and self.params[0]>=self.dim_min: #Cannot merge
            u = random.uniform(0,1-self.psplit)
            if u<self.pmerge:
                return 0
            else:
                return 2
        else: #Cannot split nor merge
            return 2
    
    def propose(self):
        choice = self.propose_process()
        if choice==0:
                proposal, padding_params, index = self.split()    
                self.p = self.pmerge/self.psplit/self.split_rvs.pdf(padding_params[0])*self.detjacobian
        elif choice==1:
                proposal, padding_params, index = self.merge()
                self.p = self.psplit*self.split_rvs.pdf(padding_params[0])/self.pmerge/self.detjacobian
        else:
            proposal, index = self.perturb()
            self.p = 1 #If no split or merge, it is a symmetrical rwmh
        return proposal, index