from template_1_train_pso import *
#from template_1_train_ga import *
from template_1_car import MyCar

#########PSO

# These can be freely changed
hyper_params = {
    'w': 10, # PSO parameter - coefficient of particle's inertia
    'c1': 0.1, # PSO parameter - coefficient of particle's local movement
    'c2': 0.2, # PSO parameter - coefficient of particle's local movement
    'weight_range': (-3,3), # Range in which the weights can vary
    'iterations': 9, # Number of iteration
    'population_size': 20, # Size of the population
}

'''
Although weight_size can also be considered as a hyper parameters, it can't be
freely changed as changing this would require a corresponding change in your
car's move() function
'''
weight_size = 8

# create a population of cars
print('Creating the population ...')
pop = create_population(weight_size, hyper_params)

# train the Particle Swarm optimization on different tracks
print('Training on track 1 ...')
track1 = 'tracks/sample_track_1.jpg'
pop = PSO(pop, track1, hyper_params, print_every=3)

print('Training on track 2 ...')
track2 = 'tracks/sample_track_2.jpg'
pop = PSO(pop, track2, hyper_params, print_every=3)

# save the car that runs best on track 2
best_particle = max(pop, key=lambda x: x.curr_car.run(track2))
best_particle.curr_car.run(track2, save=None)
best_particle.curr_car.save('weights')

# continue training on track 3
#car.load('pso_best_weights') # load the saved weights

print('Training on track 3 ...')
track3 = 'tracks/sample_track_30.jpg'
pop = PSO(pop, track3, hyper_params, print_every=3)

# save the car that runs best on all tracks combined
best_particle = max(pop, key=lambda x: x.curr_car.run(track1) + x.curr_car.run(track2) + x.curr_car.run(track3))
f1 = best_particle.curr_car.run(track1)
f2 = best_particle.curr_car.run(track2)
f3 = best_particle.curr_car.run(track3)
print(f'Overall fitness: {f1+f2+f3} = {f1} + {f2} + {f3}')
best_particle.curr_car.save(file='weights')
