from template_1_car import MyCar
import numpy as np

fitness = np.load('fitness.npy')
weights = np.load('weights.npy')
print(weights)
weights = np.array([1,1,1,1,1,1,1,1])



track1 = 'tracks/sample_track_1.jpg'
track2 = 'tracks/sample_track_2.jpg'
track3 = 'tracks/sample_track_30.jpg'

car = MyCar(id=0, weights = weights) # weights are dummy as we are not using them in our move function at all

# Visualize on different tracks
f1 = car.run(track1, save= None)
f2 = car.run(track2)
f3 = car.run(track3)
print(f'Overall fitness: {(f1+f2+f3)/3} = {f1} + {f2} + {f3}')
f = (f1+f2+f3)/3
np.append(fitness,f)
#print(fitness)
np.save('fitness.npy', fitness)