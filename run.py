from solver import Solver
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time as timer
import sys
folder = 'results2/'
DT = 1e-6
time = 0
nonrefinedmesh = IntervalMesh(300, 0, 3)
number_of_refine = 10
bottom_point = 1
refine_length = 0.2
cut_per_refine = 0.5
endT = 70

Problem = Solver(DT)
Problem.load_mesh(nonrefinedmesh)
Problem.refine_mesh(number_of_refine, bottom_point, refine_length, cut_per_refine)
Problem.refine_mesh(number_of_refine, 0, 0.5, cut_per_refine)
Problem.create_function_space_mesh()
Problem.initialize_field()
Problem.split_functions()
Problem.flux()
Problem.formulate_problem()

coord = Problem.mesh.coordinates()
frame = 0

#sys.exit()
###
start_t = timer.time()

J = derivative(Problem.eq, Problem.up)
pb = NonlinearVariationalProblem(Problem.eq, Problem.up, J=J)
sv = NonlinearVariationalSolver(pb)
prm = sv.parameters
#prm['newton_solver']['relative_tolerance'] = 1e-9
#prm['newton_solver']['absolute_tolerance'] = 1e-8

### OPTIONAL:
# write initial condition to csv
h_, p_ = Problem.up0.split(deepcopy=True)
K_ = Problem.K
coord = np.asarray([float(coord[kj]) for kj in range(len(coord))])
coord = np.sort(coord)
u_arr = np.asarray([h_(c) for c in coord])
new_vol = 2*np.pi*np.trapz(coord*u_arr, coord)
timeframe = open(folder+'timeframe%g.csv'%frame, 'w')
timeframe.write('t=%f \n' % time)
timeframe.write('vol=%f \n' % new_vol)
timeframe.write('x\t height \t pressure \t flux \n')

for position in range(len(coord)):
    timeframe.write('%.8f\t %.10f\t %.10f\t %.10f\n' %(float(coord[position]), h_(float(coord[position])), p_(float(coord[position])), K_(float(coord[position]))))
timeframe.close()

while time < endT:

    frame += 1
    time += DT

#solve(Problem.eq == 0, Problem.up)
    no, cvg = sv.solve() # solve
    if no < 5:
        DT = np.min([DT*1.5, 1e-1])
        Problem.dt.assign(Constant(DT))
    else:
        dtp = np.max([DT/2, 1e-6])
        Problem.dt.assign(Constant(DT))
    if frame % 100 == 0:
        print(frame, time)
        print((timer.time()-start_t)/60)
    if frame==2:
        print(u_arr)

    h_, p_ = Problem.up.split(deepcopy=True)
    K_ = Problem.K

    ### OPTIONAL
    # printing to csv file for easy plotting
    u_arr = np.asarray([h_(c) for c in coord])
    new_vol = 2*np.pi*np.trapz(coord*u_arr, coord)
    timeframe = open(folder+'timeframe%g.csv'%frame, 'w')
    timeframe.write('t=%f \n' % time)
    timeframe.write('vol=%f \n' % new_vol)
    timeframe.write('x\t height \t pressure \t flux \n')

    for position in range(len(coord)):
        timeframe.write('%.8f\t %.10f\t %.10f\t %.10f\n' %(float(coord[position]), h_(float(coord[position])), p_(float(coord[position])), K_(float(coord[position]))))
    timeframe.close()

    Problem.up0.assign(Problem.up)
    Problem.flux()

 #   DT *= 1.5
  #  if DT>1e-1:
   #     DT = 1e-1
    #Problem.dt.assign(DT)

end_t = timer.time()
print((end_t-start_t)/60)