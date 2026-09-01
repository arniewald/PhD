from Priors import LayersPrior
from Proposers import LayersProposer
from Models import LayersPoissonModel

import numpy as np
import copy
import scipy.stats as st
from time import time

class LayersSampler:

    def __init__(self, Model: LayersPoissonModel = None, Prior: LayersPrior = None, Proposer: LayersProposer = None, init_dim=None, init_lnks=None, init_widths=None):

        self.model = Model
        self.prior = Prior
        self.proposer = Proposer
        self.init_params = [init_dim, init_lnks, init_widths]
        self.params_chains = []
        self.proposal_chains = []
        self.acceptance_chains = []
        self.u_chains = []
        self.lnk_sigma_chains = []
        self.width_sigma_chains = []
        self.ull_chains = []
        self.proposals = []
        self.times = []

        self.reset(self.init_params)

    def reset(self, init_params):
            if init_params[0]!=None: #If dimension is predetermined, we must check if anything else is
                self.params0 = self.prior.draw(dim=init_params[0])
                for i in range(1,len(init_params)):
                    if init_params[i] is not None:
                        self.params0[i] = copy.deepcopy(init_params[i]) #We asume all non-dimension parameters are iterable
            else: #If dimension is not predetermined, draw all parameters randomly
                self.params0 = self.prior.draw()
            i = 0
            while self.prior.eval_transform(self.params0)==0 and i<=100:
                
                i += 1
                if init_params[0]!=None: #If dimension is predetermined, we must check if anything else is
                    self.params0 = self.prior.draw(dim=init_params[0])
                    for i in range(1,len(init_params)):
                        if init_params[i] is not None:
                            self.params0[i] = copy.deepcopy(init_params[i]) #We asume all non-dimension parameters are iterable
                else: #If dimension is not predetermined, draw all parameters randomly
                    self.params0 = self.prior.draw()
            if i>100:
                print("Noup :(")

            self.proposer.move(self.params0)
            self.model.set_conductivity(self.params0)
            self.lnprior0 = self.prior.eval_transform(self.params0) #Evaluate probability
            self.model.solve() #Find new temperature
            self.u0 = self.model.eval(self.model.measurements[:,:-1],"temperature")
            self.ull0= self.compute_unnormalized_loglikelihood()
            self.paramsf = copy.deepcopy(self.params0)
            self.lnpriorf = self.lnprior0
            self.uf = copy.deepcopy(self.u0)
            self.ullf = self.compute_unnormalized_loglikelihood()
        
    def compute_unnormalized_loglikelihood(self):
        dif = self.model.measurements[:,-1]-self.model.eval(self.model.measurements[:,:-1],"temperature")
        ull = -0.5*np.matmul(np.matmul(dif,self.model.invcov),dif.T)
        return ull

    def accept_reject(self):
        if self.lnprior0 == 0 and self.lnpriorf != 0:
            return True
        elif self.lnpriorf == 0 and self.lnprior0 != 0:
            return False
        else:
            alpha = min(0, self.ullf-self.ull0+np.log(self.lnpriorf*self.proposer.p/self.lnprior0))
            u = np.random.uniform()
            accept = (alpha>np.log(u))
            return accept
    
    def propose(self):
        start = time()
        self.paramsf, _ = self.proposer.propose()
        propose_time = time()-start

        start = time()
        self.model.set_conductivity(self.paramsf)
        conductivity_time = time()-start

        start = time()
        self.lnpriorf = self.prior.eval_transform(self.paramsf) #Evaluate probability
        prior_time = time()-start

        start = time()
        self.model.solve() #Find new temperature
        solve_time = time()-start

        self.uf = self.model.eval(self.model.measurements[:,:-1],"temperature")

        start = time()
        self.ullf = self.compute_unnormalized_loglikelihood()
        ull_time =  time()-start
        return [propose_time, conductivity_time, prior_time, solve_time, ull_time]

    def adapt_proposer(self, t, n_iter, acceptance_rate):
        if self.proposer.psplit*self.proposer.pmerge==0:
            delta = 0.01
            if t%50==0 and t<n_iter//2:
                lnk_sigma = np.exp(np.log(self.proposer.lnk_rvs.std())+(-int(acceptance_rate<=0.234)+int(acceptance_rate>=0.3))*delta)
                self.proposer.lnk_rvs = st.norm(scale=lnk_sigma)

    def perform_step(self, acceptance):
        times = self.propose()
        self.proposals[-1].append(copy.deepcopy(self.paramsf))
        start = time()
        if self.accept_reject():
            self.params0, self.lnprior0, self.u0, self.ull0  = copy.deepcopy(self.paramsf), self.lnpriorf, copy.deepcopy(self.uf), self.ullf
            self.proposer.move(self.params0)
            acceptance += 1
        times.append(time()-start)
        self.params_chains[-1].append(self.params0)
        self.proposal_chains[-1].append(self.paramsf)
        self.acceptance_chains[-1].append(acceptance/len(self.params_chains[-1]))
        self.u_chains[-1].append(copy.deepcopy(self.u0))
        self.lnk_sigma_chains[-1].append(self.proposer.lnk_rvs.std())
        self.width_sigma_chains[-1].append(self.proposer.width_rvs.std())
        self.ull_chains[-1].append(self.ull0)
        self.times[-1].append(times)
        return acceptance

    def draw_chain(self, n_iter):
        self.reset(self.init_params)
        self.params_chains.append([self.params0])
        self.proposal_chains.append([self.paramsf])
        self.acceptance_chains.append([0])
        self.u_chains.append([self.model.eval(self.model.measurements[:,:-1],"temperature")])
        self.lnk_sigma_chains.append([self.proposer.lnk_rvs.std()])
        self.width_sigma_chains.append([self.proposer.width_rvs.std()])
        self.ull_chains.append([self.ull0])
        self.proposals.append([copy.deepcopy(self.paramsf)])
        self.times.append([[0,0,0,0,0,0]])
        acceptance = 0
        print("Start successfull!")
        for t in range(n_iter):
            acceptance = self.perform_step(acceptance)
            acceptance_rate = acceptance/(t+1)
            self.adapt_proposer(t, n_iter, acceptance_rate)
    
    def draw_chains(self, n_iter, n_chains=1):
        for _ in range(n_chains):
            self.draw_chain(n_iter)
             
