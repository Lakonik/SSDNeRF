import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pose_spherical(theta, phi, r):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def fibonacci_sphere(samples=1000):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(cart2sph(x, y, z))

    return points

def cart2sph(x, y, z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arccos(z / rho)
    theta = np.arctan2(y, x)
    return theta, phi, rho

# Visualize poses on a sphere
poses = [pose_spherical(theta, phi, -1.3) for theta, phi, _ in fibonacci_sphere(6)]

print('!!!')
print(poses[0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for pose in poses:
    ax.scatter(*pose, color='b', s=5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Poses on Sphere')

plt.show()