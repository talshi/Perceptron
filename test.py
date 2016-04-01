# import numpy as np
#
# threshold = 0.5
# learning_rate = 0.1
# weights = [0, 0, 0]
# training_set = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]
#
# def dot_product(values, weights):
#     return sum(value * weight for value, weight in zip(values, weights))
#
# while True:
#     print('-' * 60)
#     error_count = 0
#     for input_vector, desired_output in training_set:
#         print(weights)
#         print "sum of input_vector*weights =", dot_product(input_vector, weights)
#         result = dot_product(input_vector, weights) > threshold
#         print "result =", result
#         error = desired_output - result
#         print "error =", error
#         if error != 0:
#             error_count += 1
#             for index, value in enumerate(input_vector):
#                 weights[index] += learning_rate * error * value
#     if error_count == 0:
#         break

# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import os, subprocess
#
# class Perceptron:
#     def __init__(self, N):
#         # Random linearly separated data
#         xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
#         self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
#         self.X = self.generate_points(N)
#
#     def generate_points(self, N):
#         X = []
#         for i in range(N):
#             x1,x2 = [random.uniform(-1, 1) for i in range(2)]
#             x = np.array([1,x1,x2])
#             s = int(np.sign(self.V.T.dot(x)))
#             X.append((x, s))
#         return X
#
#     def plot(self, mispts=None, vec=None, save=False):
#         fig = plt.figure(figsize=(5,5))
#         plt.xlim(-1,1)
#         plt.ylim(-1,1)
#         V = self.V
#         a, b = -V[1]/V[2], -V[0]/V[2]
#         l = np.linspace(-1,1)
#         plt.plot(l, a*l+b, 'k-')
#         cols = {1: 'r', -1: 'b'}
#         for x,s in self.X:
#             plt.plot(x[1], x[2], cols[s]+'o')
#         if mispts:
#             for x,s in mispts:
#                 plt.plot(x[1], x[2], cols[s]+'.')
#         if vec != None:
#             aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
#             plt.plot(l, aa*l+bb, 'g-', lw=2)
#         if save:
#             if not mispts:
#                 plt.title('N = %s' % (str(len(self.X))))
#             else:
#                 plt.title('N = %s with %s test points' \
#                           % (str(len(self.X)),str(len(mispts))))
#             plt.savefig('p_N%s' % (str(len(self.X))), \
#                         dpi=200, bbox_inches='tight')
#
#     def classification_error(self, vec, pts=None):
#         # Error defined as fraction of misclassified points
#         if not pts:
#             pts = self.X
#         M = len(pts)
#         n_mispts = 0
#         for x,s in pts:
#             if int(np.sign(vec.T.dot(x))) != s:
#                 n_mispts += 1
#         error = n_mispts / float(M)
#         return error
#
#     def choose_miscl_point(self, vec):
#         # Choose a random point among the misclassified
#         pts = self.X
#         mispts = []
#         for x,s in pts:
#             if int(np.sign(vec.T.dot(x))) != s:
#                 mispts.append((x, s))
#         return mispts[random.randrange(0,len(mispts))]
#
#     def pla(self, save=False):
#         # Initialize the weigths to zeros
#         w = np.zeros(3)
#         X, N = self.X, len(self.X)
#         it = 0
#         # Iterate until all points are correctly classified
#         while self.classification_error(w) != 0:
#             it += 1
#             # Pick random misclassified point
#             x, s = self.choose_miscl_point(w)
#             # Update weights
#             w += s*x
#             if save:
#                 self.plot(vec=w)
#                 plt.title('N = %s, Iteration %s\n' \
#                           % (str(N),str(it)))
#                 plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
#                             dpi=200, bbox_inches='tight')
#         self.w = w
#
#     def check_error(self, M, vec):
#         check_pts = self.generate_points(M)
#         return self.classification_error(vec, pts=check_pts)
#
# p = Perceptron(20)
# p.plot()
# plt.show()

"""
This short code snippet utilizes the new animation package in
matplotlib 1.1.0; it's the shortest snippet that I know of that can
produce an animated plot in python. I'm still hoping that the
animate package's syntax can be simplified further.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# def simData():
# this function is called as the argument for
# the simPoints function. This function contains
# (or defines) and iterator---a device that computes
# a value, passes it back to the main program, and then
# returns to exactly where it left off in the function upon the
# next call. I believe that one has to use this method to animate
# a function using the matplotlib animation package.
#
#     t_max = 10.0
#     dt = 0.05
#     x = 0.0
#     t = 0.0
#     while t < t_max:
#         x = np.sin(np.pi*t)
#         t = t + dt
#         yield x, t
#
# def simPoints(simData):
#     x, t = simData[0], simData[1]
#     time_text.set_text(time_template%(t))
#     line.set_data(t, x)
#     return line, time_text
#
# ##
# ##   set up figure for plotting:
# ##
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # I'm still unfamiliar with the following line of code:
# line, = ax.plot([], [], 'bo', ms=10)
# ax.set_ylim(-1, 1)
# ax.set_xlim(0, 10)
# ##
# time_template = 'Time = %.1f s'    # prints running simulation time
# time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
# ## Now call the animation package: (simData is the user function
# ## serving as the argument for simPoints):
# ani = animation.FuncAnimation(fig, simPoints, simData, blit=False,\
#      interval=10, repeat=True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_line(num, data, line):
    line.set_data(data[..., :num])
    return line,

fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
                                   interval=150, blit=True)

plt.show()