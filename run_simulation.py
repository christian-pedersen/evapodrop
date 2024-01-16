import time
import solver

N = 300
dtp = 1e-6
mesh = IntervalMesh(N, 0, 3)
solver = Solver(dtp)
solver.load_mesh(mesh)
solver.refine_mesh(10, 1, 0.5, 0.5)
solver.create_function_space_mesh()
solver.initialize_field()
solver.split_functions()
solver.flux()
solver.formulate_problem()

J = derivative(solver.eq, solver.up)
problem = NonlinearVariationalProblem(solver.eq, solver.up, J=J)
sv = NonlinearVariationalSolver(problem)
prm = sv.parameters
prm['newton_solver']['relative_tolerance'] = 1e-9
prm['newton_solver']['absolute_tolerance'] = 1e-8 


start = time.time()
tt = 0
end_time = 1e-1
st = 0
while tt < end_time + DOLFIN_EPS:
    solver.flux() # compute the flux
   # solver.formulate_problem() only once if I don't refine the mesh
    tt+=dtp
    st+=1
    no, cvg = sv.solve() # solve
    u_temp, _ = solver.up.split(deepcopy=True)
    solver.up0.assign(solver.up) # update the profile
    if no < 5:
        dtp = np.min([dtp*5, 1e-2])
        solver.dt.assign(Constant(dtp))
    else:
        dtp = np.max([dtp/2, 1e-6])
        solver.dt.assign(Constant(dtp))
    if st % 100 == 0:
        print(st)
        print((time.time()-start)/60)
ed_time = time.time()
print('end ', (time.time()-start)/60) # Tooks around 20 minutes

plt.figure()
plot(solver.u, ls='', marker='.')
plt.savefig('profile_end.png') # Profile unchanged

plt.figure()
plt.plot(solver.x__.coordinates(), solver.K_.vector()[:], marker='.', ls='')
plt.savefig('Flux.png') # the flux seems to be updated