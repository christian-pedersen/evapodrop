from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

class droplet(UserExpression):
    def eval(self, values, x):
        epsilon = 10**-3


        if between(x[0], (0, 1)):
            values[0] = 0.1*(1 - x[0]**2)**2 + epsilon#((cos(0.5*x[0]*np.pi)*cos(0.5*x[0]*np.pi))+100*epsilon)/(1+100*epsilon)#np.sqrt(1 - x[0]**2) + epsilon#init_u(abs(x[0]))
        else:
            values[0] = epsilon#-99*epsilon/ (1-5) * (x[0]-1) + 100*epsilon

class Solver:

    def __init__(self, DT):
        self.dt = Constant(DT)
        self.a = 10**-4
        self.hf = 10**-5
        self.beta = 21

    def load_mesh(self, mesh):
        self.nonrefinedmesh = mesh
        self.mesh = mesh

    def refine_mesh(self, number_of_refine, bottom_point, refine_length, cut_per_refine):
        mesh = self.nonrefinedmesh
        for i in range(number_of_refine):
            cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
            cell_markers.set_all(False)
            for cell in cells(mesh):
                if abs(cell.midpoint().x()) <= bottom_point and abs(cell.midpoint().x()) >= (bottom_point - 0.5*refine_length):
                    cell_markers[cell] = True
                if abs(cell.midpoint().x()) > bottom_point and abs(cell.midpoint().x()) <= (bottom_point + 0.5*refine_length):
                    cell_markers[cell] = True
            mesh = refine(mesh, cell_markers)
            refine_length *= cut_per_refine
        self.mesh = mesh
        mesh_file=File("mesh_refined.xml")
        mesh_file<<self.mesh

    def create_function_space_mesh(self):
        self.elms = FiniteElement('Lagrange', self.mesh.ufl_cell(), degree=1)
        self.mixedelms = MixedElement([self.elms, self.elms])
        self.U = FunctionSpace(self.mesh, self.elms)
        self.P = FunctionSpace(self.mesh, self.elms)
        self.UP = FunctionSpace(self.mesh, self.mixedelms)
        self.x__ = IntervalMesh(2000, 0, 3)
        #self.V_ = FunctionSpace(self.x__, self.elms)

        self.v, self.q = TestFunctions(self.UP)

    def initialize_field(self):

        self.up = Function(self.UP)
        self.up0 = Function(self.UP)
        #self.K_ = Function(self.V_)
        init = droplet()
        init = interpolate(init, self.U)

        FunctionAssigner(self.UP.sub(0), self.U).assign(self.up0.sub(0), init)
        FunctionAssigner(self.UP.sub(0), self.U).assign(self.up.sub(0), init)


    def interpolate_field(self, up_temp):
        self.up0 = project(up_temp, self.UP)
        self.up = interpolate(up_temp, self.UP)

    def split_functions(self):
        self.u, self.p = split(self.up)
        self.u0, self.p0 = split(self.up0)

    def flux(self):

        u0_, p0_ = self.up0.split(deepcopy=True)

        u0__ = project(u0_.dx(0), self.U)

        x = np.linspace(0, 3, 251)
        x = np.concatenate((np.linspace(0, 0.0007, 1), np.linspace(0.00065, 1.2, 1879)), axis=None)
        x = np.concatenate((x, np.linspace(1.205, 3, 121)), axis=None)
        # print(x)
        no =0
        for xx in self.x__.coordinates():
            #print(xx[0], x[no])
            xx[0] = x[no]
            no += 1
        x_ = np.logspace(-11,9, 201)
        u0k, u0k_ = [],[]
        for xx in range(len(x_)):
            if x_[xx] < max(x):
                u0k.append(u0_(x_[xx]))
                u0k_.append(u0__(x_[xx]))
        u0k, u0k_ = np.asarray(u0k), np.asarray(u0k_)
        u0 = np.asarray([u0_(x[i]) for i in range(len(x))])
        u0_ = np.asarray([u0__(x[i]) for i in range(len(x))])

        K_series = np.zeros(len(x))
        K_intermediate = np.zeros(len(x_))
        for k in range(len(x)):
            K1_sum = 0
            K2_sum = 0
            K_sum = 0
            for k_ in range(len(x_)):
                K1_sum = 0
                K2_sum = 0
                K_sum = 0

                if x_[k_] > max(x):
                    #print(len(u0k))
                    K_sum += x[k]**2/2./x_[k_] * 1 / x_[k_]**2 #* (-1)*self.hf**2 

                elif x[k] == 0:
                    K_sum += np.pi/2./x_[k_]*3*self.hf*(self.hf/u0k[k_])**2/u0k[k_]**2*u0k_[k_] * (-1)

                elif x_[k_] < 1.001*x[k] and x_[k_] > 0.999*x[k]:
                    if x_[k_] == x[k]:
                        delta = 10**-6
                    else:
                        delta = abs(x_[k_]-x[k])
                    K_sum += x[k]/2 * np.log(delta)*3*self.hf*(self.hf/u0k[k_])**2*(-1)/u0k[k_]**2*u0k_[k_]
                    #print(delta, x[k]/2 * np.log(delta))
                elif x_[k_] <= 0.999*x[k]:
             #       for i in range(70):
              #          K1_sum += (np.math.factorial(2*i) / 2**(2*i) / (np.math.factorial(i))**2)**2 * (x_[k_]/x[k])**(2*i)
               #         K2_sum += (np.math.factorial(2*i) / 2**(2*i) / (np.math.factorial(i))**2)**2 * (x_[k_]/x[k])**(2*i) / (1-2*i)
                #    K_sum += x[k]*(K1_sum - K2_sum)*3*self.hf*(self.hf/u0k[k_])**2*(-1)/u0k[k_]**2*u0k_[k_]
                    K1_sum = (ellipk((x_[k_]/x[k])**2)-ellipe((x_[k_]/x[k])**2))
                    K2_sum=0
                    K_sum += x[k]*(K1_sum - K2_sum)*3*self.hf*(self.hf/u0k[k_])**2*(-1)/u0k[k_]**2*u0k_[k_]

                else:
                  #  for i in range(70):
                   #     K1_sum += (np.math.factorial(2*i) / 2**(2*i) / (np.math.factorial(i))**2)**2 * (x[k]/x_[k_])**(2*i)
                    #    K2_sum += (np.math.factorial(2*i) / 2**(2*i) / (np.math.factorial(i))**2)**2 * (x[k]/x_[k_])**(2*i) / (1-2*i)
                    K1_sum = (ellipk((x[k]/x_[k_])**2)-ellipe((x[k]/x_[k_])**2))
                    K2_sum=0
                    K_sum += x_[k_]*(K1_sum - K2_sum)*3*self.hf*(self.hf/u0k[k_])**2*(-1)/u0k[k_]**2*u0k_[k_]
                K_intermediate[k_] = 2/np.pi*K_sum
            #print(K_intermediate)

            K_series[k] = np.trapz(K_intermediate, x_)
        #K_series[0] = K_series[1] - 0.1*(K_series[1]+K_series[2])
        self.elms_ = FiniteElement('Lagrange', self.x__.ufl_cell(), degree=1)
        self.V_ = FunctionSpace(self.x__, self.elms_)
        self.K_ = Function(self.V_)
        self.K_.vector().set_local(K_series)
        self.K = interpolate(self.K_, self.U)

        #for i in range(len(self.x__.coordinates())):
        #    print(self.x__.coordinates()[i], self.K_(self.x__.coordinates()[i]), self.K(self.x__.coordinates()[i]))


    def formulate_problem(self):

        self.r = Expression('sqrt(x[0]*x[0])', degree=1)

        left = CompiledSubDomain('on_boundary && sqrt(x[0]*x[0]) < DOLFIN_EPS')
        right = CompiledSubDomain('on_boundary && sqrt(x[0]*x[0]) > 2')
        sub_domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        sub_domains.set_all(0)


        left.mark(sub_domains, 1)
        right.mark(sub_domains, 2)
        ds=Measure('ds', domain=self.mesh, subdomain_data=sub_domains)


        self.eq1 = inner(self.p*self.r, self.q)*dx + inner(self.r*self.u.dx(0), self.q.dx(0))*dx
        self.eq2 = inner(self.u*self.r, self.v)*dx - inner(self.u0*self.r, self.v)*dx  - self.dt*inner(self.r*self.u*self.u*self.u*(self.p + self.a**2/self.u**3).dx(0) , self.v.dx(0))*dx + self.dt*inner(self.r*self.u*self.u*self.u*(self.a**2/self.u**3).dx(0) , self.v)*ds - self.beta*self.dt*inner(self.K, self.v.dx(0))*dx + self.beta*self.dt*inner(self.K, self.v)*ds #+ self.dt*inner(self.r*self.u*self.u*self.u*(self.u.dx(0)/self.r + self.b/self.u**3).dx(0) , self.v)*ds

        self.eq = self.eq1 + self.eq2
