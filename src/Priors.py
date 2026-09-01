import numpy as np
np.set_printoptions(legacy='1.25')
import math
import scipy.stats as st


class LayersPrior:

    def __init__(self, dim_min, dim_max, k_min, k_max, precision):
        self.dimprior = st.randint(low=dim_min, high=dim_max+1)
        self.lnkrvs = st.uniform(np.log(k_min), np.log(k_max)-np.log(k_min))
        self.precision = precision

    def draw(self, dim=None):
        if dim==None:
            dim = int(self.dimprior.rvs(1))
        widths = [width for width in np.random.uniform(0,1,dim)]
        sumwidths = sum(widths)
        widths = [width/sumwidths for width in widths]
        for i in range(dim):
            widths[i] = round(widths[i]/self.precision)*self.precision
        widths = widths[:-1]
        return [dim, list(self.lnkrvs.rvs(size=dim)), widths]
    
    def eval_transform(self, params):
        case1 = int(np.all(np.array(params[2])>=self.precision))
        case2 = 1-sum(params[2])>=self.precision
        case3 = np.isclose(1-sum(params[2]), self.precision)
        case = case1*(int(case2 or case3))
        return self.dimprior.pmf(params[0])*np.prod([self.lnkrvs.pdf(lnk) for lnk in params[1]])*case*int(sum(params[2])<1)*math.factorial(max(0,params[0]-2))
