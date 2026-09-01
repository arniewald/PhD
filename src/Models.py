from mpi4py import MPI
from dolfinx import mesh, fem, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
np.set_printoptions(legacy='1.25')
import copy
import ufl
import pyvista
import scipy.stats as st
import scipy.spatial as sp


class MyPoissonModel:

    def __init__(self, n_squares=20, f_val=1e6, lksigma=1, k_space=("DG",0), u_space=("Lagrange",1), dirichlet_boundary_function = lambda x: np.isclose(x[1], 1), dirichlet_function = lambda x: x[1], neumann_boundary_function = lambda x: np.isclose(x[1], 0), neumann_function = lambda x: x[0]):

        self.create_domain(n_squares)
        self.create_function_spaces(k_space, u_space)
        self.set_dirichlet_boundary_conditions(dirichlet_boundary_function=dirichlet_boundary_function, dirichlet_function=dirichlet_function)
        self.set_neumann_boundary_conditions(neumann_boundary_function=neumann_boundary_function, neumann_function=neumann_function)
        self.set_robin_boundary_conditions()

        #f
        self.f = fem.Constant(self.domain, default_scalar_type(f_val))

        #Conductivity  (only defined)
        self.true_params = None
        self.true_k = fem.Function(self.K)
        self.k = fem.Function(self.K)
        
        #Temperature (only defined)
        self.uh = fem.Function(self.V)
        self.true_uh = fem.Function(self.V)
        self.measured_uh = fem.Function(self.V)

        #Likelihood
        self.lksigma = lksigma
        self.invcov = np.linalg.inv(lksigma**2*np.identity((n_squares+1)**2))

        return None

    def create_domain(self, n_squares):
        self.n_squares = n_squares
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, n_squares, n_squares, mesh.CellType.quadrilateral)
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1
        self.domain.topology.create_connectivity(self.fdim, self.tdim)
        self.boundary_facets = mesh.exterior_facet_indices(self.domain.topology)

    def create_function_spaces(self, k_space, u_space):
        #Function spaces
        self.K = fem.functionspace(self.domain,k_space)
        self.V = fem.functionspace(self.domain,u_space)
        self.v = ufl.TestFunction(self.V) #Test functions
        self.u = ufl.TrialFunction(self.V) #Trial functions

    def set_dirichlet_boundary_conditions(self, dirichlet_boundary_function, dirichlet_function):
        #Dirichlet boundary conditions
        dofs_D = fem.locate_dofs_geometrical(self.V, dirichlet_boundary_function)
        uD = fem.Function(self.V)
        uD.interpolate(dirichlet_function)
        self.bc = fem.dirichletbc(uD, dofs_D)
    
    def set_neumann_boundary_conditions(self, neumann_boundary_function, neumann_function):
        #Neumann boundary conditions
        fdim = self.domain.topology.dim - 1
        facet = mesh.locate_entities(self.domain, fdim, neumann_boundary_function)
        facet_tag = mesh.meshtags(self.domain, fdim, facet[np.argsort(facet)], [1 for _ in facet])
        self.facet_tag = facet_tag
        self.ds = ufl.Measure("ds", domain=self.domain, subdomain_data=facet_tag)
        x = ufl.SpatialCoordinate(self.domain) 
        self.x = x
        self.uN = neumann_function(x)
    
    def set_robin_boundary_conditions(self):
        pass

    def set_conductivity(self, k):
        #Sets the conductivity to a given function k
        self.k.x.array[0:] = k

    def solve(self):
        a = ufl.dot(self.k * ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        L = self.f * self.v * ufl.dx + self.uN * self.v * self.k * self.ds
        problem = LinearProblem(a, L, bcs=[self.bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.uh = problem.solve()

    def eval(self, points: np.array, magnitude):
        magnitude_dict = {
            "temperature": self.uh,
            "true_temperature": self.true_uh,
            "measured_temperature": self.measured_uh,
            "conductivity": self.k,
            "true_conductivity": self.true_k
        }

        x = points.reshape(-1,2)
        x = np.hstack((x,np.zeros((x.shape[0],1))))
        tree = bb_tree(self.domain, self.domain.geometry.dim)
        cell_candidates = compute_collisions_points(tree, x)
        cell = compute_colliding_cells(self.domain, cell_candidates, x)
        values = magnitude_dict[magnitude].eval(x, cell.array[[cell.offsets[i] for i in range(x.shape[0])]])
        return values.reshape(values.shape[0])

    def compute_l1error(self, magnitude):
        if magnitude=="temperature":
            empirical = self.uh.x.array.reshape((self.n_squares+1,self.n_squares+1))
            real = self.true_uh.x.array.reshape((self.n_squares+1,self.n_squares+1))
            abs_real = abs(real)
            integral = np.trapezoid(np.trapezoid(abs_real, np.linspace(0,1,self.n_squares+1), axis=1), np.linspace(0,1,self.n_squares+1), axis=0)
            diff = abs(empirical-real)
            diffintegral = np.trapezoid(np.trapezoid(diff, np.linspace(0,1,self.n_squares+1), axis=1), np.linspace(0,1,self.n_squares+1), axis=0)
            error = diffintegral/integral
        elif magnitude=="conductivity":
            empirical = self.k.x.array[0:]
            real = self.true_k.x.array[0:]
            diff = abs(empirical-real)
            diffintegral = sum(diff)/self.n_squares/self.n_squares/2
            integral = sum(real)/self.n_squares/self.n_squares/2
            error = diffintegral/integral
        return error

    def visualize(self, magnitude, measurements, title=None):

        if title==None:
            title_dict = {
                "temperature": "Temperature",
                "true_temperature": "True temperature",
                "measured_temperature": "Measured temperature",
                "log-conductivity": "Log-conductivity",
                "true_log-conductivity": "True log-conductivity"
            }
            title = title_dict[magnitude]

        magnitude_dict = {
            "temperature": self.uh.x.array.real,
            "true_temperature": self.true_uh.x.array.real,
            "measured_temperature": self.measured_uh.x.array.real,
            "log-conductivity": self.k.x.array.real,
            "true_log-conductivity": self.true_k.x.array.real
        }

        topology, cell_types, geometry = plot.vtk_mesh(self.domain, self.tdim)
        plotter = pyvista.Plotter(window_size=(1000,1000))
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        if magnitude in ["temperature","true_temperature","measured_temperature"]:
            grid.point_data[magnitude] = magnitude_dict[magnitude]
            points = pyvista.PointSet(np.hstack([measurements[:,:-1], 0.01*np.ones(measurements.shape[0]).reshape(-1,1)]))
            actor = plotter.add_points(points, style='points_gaussian', point_size=10, color='red', label='Measuring points')
            grid.set_active_scalars(magnitude)
            plotter.add_mesh(grid, show_edges=False, cmap=plt.get_cmap('coolwarm'), scalar_bar_args=dict(title='[K]\n', position_x=0.245, width=0.515, title_font_size=25, label_font_size=25, fmt='%.0f'))
        else:
            grid.cell_data[magnitude] = magnitude_dict[magnitude]
            grid.set_active_scalars(magnitude)
            plotter.add_mesh(grid, show_edges=False, scalar_bar_args=dict(title='[ln(W/(m·K))]\n', position_x=0.245, width=0.515, title_font_size=25, label_font_size=25))#, clim = [2.72, 403])
        plotter.add_title(title)
        plotter.view_xy()
        plotter.show()

        return plotter

class LayersPoissonModel(MyPoissonModel):

    def __init__(self, true_params, measurements, n_squares=20, f_val=1000000, lksigma=1, orientation = 'v', k_space=("DG", 0), u_space=("Lagrange", 1), dirichlet_boundary_function=lambda x: np.isclose(x[1], 1), dirichlet_function=lambda x: x[1], neumann_boundary_function=lambda x: np.isclose(x[1], 0), neumann_function=lambda x: x[0]):
        super().__init__(n_squares, f_val, lksigma, k_space, u_space, dirichlet_boundary_function, dirichlet_function, neumann_boundary_function, neumann_function)

        self.orientation_index = 0 if orientation=='v' else 1
        #True conductivity and conducitivity
        #Structure: number of bands (N), conductivities (N), width of band left-right ('v') or down-up ('h') (N-1)
        self.true_params = true_params 
        self.params = copy.deepcopy(true_params)
        self.set_true_conductivity()
        self.set_conductivity(self.params)

        #Initial solution
        a = ufl.dot(self.true_k * ufl.grad(self.u), ufl.grad(self.v)) * ufl.dx
        L = self.f * self.v * ufl.dx + self.uN * self.v * self.true_k * self.ds(1) #+ self.uN * self.v * self.k * ds(4)
        problem = LinearProblem(a, L, bcs=[self.bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        self.true_uh = problem.solve()

        measurements = measurements.copy()
        if len(measurements.shape)==1:
            measurements = measurements.reshape((1,measurements.shape[0]))
        if measurements.shape[1]!=3:
            measurements = np.hstack((measurements,self.eval(measurements,"true_temperature").reshape(-1,1)))
        for i in range(measurements.shape[0]):
            pert = st.norm(scale=self.lksigma).rvs()
            j = 0
            while j<100 and measurements[i,-1]+pert<0:
                j += 1
                pert = st.norm(scale=1e-2).rvs()
            if measurements[i,-1]+pert>0:
                measurements[i,-1] += pert
        self.measurements = measurements
        self.invcov = np.linalg.inv(lksigma**2*np.identity(self.measurements.shape[0]))

    def set_true_conductivity(self):
        self.true_k.interpolate(u0=lambda x: np.searchsorted(np.cumsum([0]+self.true_params[2]),x[self.orientation_index]))
        self.true_k.x.array[0:] = np.array([np.exp(self.true_params[1])[int(index)-1] for index in self.true_k.x.array[0:]])

    def set_conductivity(self, params):
        self.params = params
        self.k.interpolate(u0=lambda x: np.searchsorted(np.cumsum(self.params[2]),x[self.orientation_index]))
        self.k.x.array[0:] = np.array([np.exp(self.params[1])[int(index)] for index in self.k.x.array[0:]])

    def visualize(self, magnitude, title=None):
        return super().visualize(magnitude, self.measurements, title)

class VoronoiPoissonModel(MyPoissonModel):

    def __init__(self, true_params, measurements, n_squares=20, f_val=1000000, lksigma=1, orientation = 'v', k_space=("DG", 0), u_space=("Lagrange", 1), dirichlet_boundary_function=lambda x: np.isclose(x[1], 1), dirichlet_function=lambda x: x[1], neumann_boundary_function=lambda x: np.isclose(x[1], 0), neumann_function=lambda x: x[0]):
        super().__init__(n_squares, f_val, lksigma, k_space, u_space, dirichlet_boundary_function, dirichlet_function, neumann_boundary_function, neumann_function)
        self.true_params = true_params 
        self.set_params(true_params)
        self.measurements = measurements
        self.invcov = np.linalg.inv(self.lksigma**2*np.identity(self.measurements.shape[0]))
        
    def set_params(self, params):
        self.params = params
        midpoints = mesh.compute_midpoints(self.domain,self.tdim,np.array(range(self.n_squares**2)))[:,0:2]
        self.centers = self.params[2]
        self.k_regions = self.params[1]
        self.dim = self.params[0]
        for i in range(self.dim):
            if np.any(np.isclose(self.centers[i],midpoints)):
                self.centers[i] = [self.centers[i][0] + st.norm(scale=1/10/self.n_squares).rvs(), self.centers[i][1] + st.norm(scale=1/10/self.n_squares).rvs()]
        self.voronoi = sp.cKDTree(self.centers)
        self.k.x.array[0:] = np.exp(np.array(self.k_regions)[self.voronoi.query(mesh.compute_midpoints(self.domain,self.tdim,np.array(range(self.n_squares**2)))[:,0:2])[1]])
        self.params = [len(self.centers)]+[self.k_regions]+[self.centers]

    def visualize(self, magnitude, title=None):
        return super().visualize(magnitude, self.measurements, title)