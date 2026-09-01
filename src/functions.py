import os
import numpy as np
np.set_printoptions(legacy='1.25')
import json
import pandas as pd
from datetime import datetime
from time import time

from Models import LayersPoissonModel
from Priors import LayersPrior
from Proposers import LayersProposer
from Samplers import LayersSampler


def extract_model(model_name):
    with open('../test_data/models.json', 'r') as f:
            model_metadata = json.load(f)[model_name]
    f.close()
    true_params = [model_metadata['true_dim'],model_metadata['true_lnks'],model_metadata['true_widths']]
    measurements_df = pd.read_csv('../test_data/'+model_metadata['measurements_name']+'.csv', index_col=0)
    measurements = measurements_df[['x','y','measured']].to_numpy()
    Model = LayersPoissonModel(true_params, measurements, n_squares=model_metadata['n_squares'], f_val=model_metadata['f_val'], lksigma=model_metadata['lksigma'], orientation=model_metadata['orientation'])
    return Model

def extract_prior(prior_name):
    with open('../test_data/priors.json', 'r') as f:
        prior_metadata = json.load(f)[prior_name]
    f.close()
    Prior = LayersPrior(dim_min=prior_metadata['dim_min'], dim_max=prior_metadata['dim_max'], k_min=prior_metadata['k_min'], k_max=prior_metadata['k_max'], precision=prior_metadata['precision'])
    return Prior

def extract_proposer(proposer_name):
    with open('../test_data/proposers.json', 'r') as f:
        proposer_metadata = json.load(f)[proposer_name]
    f.close()
    Proposer = LayersProposer(
            params=proposer_metadata['params'],
            proposal_type=proposer_metadata['proposal_type'],
            lnk_sigma=proposer_metadata['lnk_sigma'],
            width_sigma=proposer_metadata['width_sigma'],
            dim_min=proposer_metadata['dim_min'],
            dim_max=proposer_metadata['dim_max'],
            std=proposer_metadata['std'],
            psplit=proposer_metadata['psplit'],
            pmerge=proposer_metadata['pmerge']
        )
    return Proposer

def extract_sampler(model, prior, proposer, init_point):
    init_dim, init_lnks, init_widths = init_point[0], init_point[1], init_point[2]
    Sampler = LayersSampler(model, prior, proposer, init_dim=init_dim, init_lnks=init_lnks, init_widths=init_widths)
    return Sampler

def save_chain_data(model_name, prior_name, proposer_name, init_point, n_iter, sampler, folder_name):
    chain_metadata = {
        "model": model_name,
        "prior": prior_name,
        "posterior": proposer_name,
        "init_point": init_point,
        "n_iter": n_iter
    }

    columns = ['lnk_sigma', 'dim']
    columns += ['lnk_'+str(i) for i in range(sampler.proposer.dim_max)]
    columns += ['width_'+str(i) for i in range(sampler.proposer.dim_max-1)]
    columns += ['u_'+str(i) for i in range(sampler.model.measurements.shape[0])]
    columns += ['acceptance', 'ull']
    columns += ['propose_time', 'conductivity_time', 'prior_time', 'solve_time', 'ull_time', 'accept-reject_time']

    data = []
    data.append(list(sampler.lnk_sigma_chains[-1]))
    data.append([sampler.params_chains[-1][i][0] for i in range(n_iter+1)])
    for i in range(sampler.proposer.dim_max):
        data.append(
            [
                None
                if sampler.params_chains[-1][j][0]<i+1
                else sampler.params_chains[-1][j][1][i]
                for j in range(n_iter+1)
            ]
        )
    for i in range(sampler.proposer.dim_max-1):
        data.append(
            [
                None
                if sampler.params_chains[-1][j][0]<i+2
                else sampler.params_chains[-1][j][2][i]
                for j in range(n_iter+1)
            ]
        )
    for i in range(sampler.model.measurements.shape[0]):
        data.append([sampler.u_chains[-1][j][i] for j in range(n_iter+1)])
    data.append(list(sampler.acceptance_chains[-1]))
    data.append(list(sampler.ull_chains[-1]))
    times_array = np.array(sampler.times[-1])
    for i in range(6):
        data.append(list(times_array[:,i]))

    df = pd.DataFrame(np.array(data).T, columns=columns)

    if folder_name==None:
        folder_name = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.isdir("../simulation_data/"+folder_name):
        os.makedirs("../simulation_data/"+folder_name)

    fileindex = len([filename for filename in os.listdir("../simulation_data/"+folder_name+"/") if "chain_" in filename])+1

    df.to_csv("../simulation_data/"+folder_name+"/chain_"+str(fileindex)+".csv")
    with open("../simulation_data/"+folder_name+"/metadata_"+str(fileindex)+".json", "w") as f:
        json.dump(chain_metadata, f)
    f.close()
    

    

#Sampler arguments
n_iter = 10

def worker(model_name, prior_name, proposer_name, init_point, n_iter, folder_name):

    #Model
    Model = extract_model(model_name)

    #Prior
    Prior = extract_prior(prior_name)

    #Proposer
    Proposer = extract_proposer(proposer_name)

    #Sampler
    Sampler = extract_sampler(Model, Prior, Proposer, init_point)

    #Samle chain
    Sampler.draw_chain(n_iter=n_iter)

    #Save metadata
    save_chain_data(model_name, prior_name, proposer_name, init_point, n_iter, Sampler, folder_name)
