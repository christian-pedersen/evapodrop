from solver import Solver
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time as timer

folder = 'results/'
DT = 1e-8
time = 0
nonrefinedmesh = IntervalMesh(200, 0, 3)
number_of_refine = 2
bottom_point = 1
refine_length = 0.2
cut_per_refine = 0.5
endT = .1

Problem = Solver(DT)
Problem.load_mesh(nonrefinedmesh)
Problem.refine_mesh(number_of_refine, bottom_point, refine_length, cut_per_refine)
Problem.create_function_space_mesh()
Problem.initialize_field()
Problem.split_functions()
Problem.flux()
Problem.formulate_problem()

coord = Problem.mesh.coordinates()
frame = 0


###
start_t = timer.time()
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

    solve(Problem.eq == 0, Problem.up)

    h_, p_ = Problem.up.split(deepcopy=True)
    K_ = Problem.K

    ### OPTIONAL
    # printing to csv file for easy plotting
    u_arr = np.asarray([h_(c) for c in coord])
    new_vol = 2*np.pi*np.trapz(coord*u_arr, coord)
    timeframe = open(folder+'timeframe%g.csv'%frame, 'w')
    timeframe.write('t=%f \n' % time)
    timeframe.write('vol=%f \n' % new_vol)
    timeframe.write('x\t height \t pressure \n')

    for position in range(len(coord)):
        timeframe.write('%.8f\t %.10f\t %.10f\t %.10f\n' %(float(coord[position]), h_(float(coord[position])), p_(float(coord[position])), K_(float(coord[position]))))
    timeframe.close()

    Problem.up0.assign(Problem.up)
    Problem.flux()

    DT *= 1.05
    Problem.dt.assign(DT)

end_t = timer.time()
print(end_t-start_t)