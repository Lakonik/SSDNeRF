import torch

import math
import numpy as np
import matplotlib.pyplot as plt

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    elev = math.atan2(z,math.sqrt(XsqPlusYsq))     # theta
    az = math.atan2(y,x)                           # phi
    elev = elev / np.pi * 180
    if elev < 0:
        elev = 360 + elev
    az = az / np.pi * 180
    if az < 0:
        az = 360 + az

    return elev, az


def fibonacci_sphere(samples=1000):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append(cart2sph(x,y,z))

    return points


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

rot_eta = lambda eta : torch.Tensor([
    [np.cos(eta),np.sin(eta),0,0],
    [-np.sin(eta), np.cos(eta),0,0],
    [0,0,1,0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    eta = 180
    c2w = rot_eta(eta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],
                                 [0,0,1,0],
                                 [0,1,0,0],
                                 [0,0,0,1]])) @ c2w
    return c2w.numpy()


def read_txt_to_tensor(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        numbers = [float(num) for line in lines for num in line.split()]
        tensor = torch.tensor(numbers).view(4, 4)
    return tensor


def create_six_digit_strings():
    six_digit_strings = []
    for i in range(1, 200, 4):
        six_digit_string = str(i).zfill(6)
        six_digit_strings.append(six_digit_string)
    return six_digit_strings


def main():
    ids = create_six_digit_strings()

    poses = []

    fxy = torch.Tensor([131.2500, 131.2500, 64.00, 64.00])
    intrinsics = fxy.repeat(8, 6, 1)
    #print(intrinsics)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    K = torch.Tensor(
        [[131.25, 0, 64.0, 0.0],
         [0, 131.25, 64.0, 0.0],
         [0, 0, 1.0, 0.0]])

    for id in ids:
        file_path = f"/Users/piotrwojcik/PycharmProjects/SSDNeRF/demo/example_pose/pose/{id}.txt"  # Path to your text file
        pose = read_txt_to_tensor(file_path)

        rounded_pose = torch.round(torch.tensor(pose) * 10000) / 10000
        #print(rounded_pose)

        point = [0, 0, 0, 1]
        point_e = [0, 0, -0.5, 1]
        point_e2 = [0.0, 0.0, 0.75, 1]

        point = torch.tensor(point).float().view(4, 1)
        point_e = torch.tensor(point_e).float().view(4, 1)
        point_e2 = torch.tensor(point_e2).float().view(4, 1)
        p_car = torch.matmul(pose, point)
        p_car_e = torch.matmul(pose, point_e)
        p_car_e2 = torch.matmul(pose, point_e2)

        photo = torch.matmul(K, p_car_e)
        photo2 = torch.matmul(K, p_car_e2)
        print(photo)
        print(photo2)
        print()

        xyz_car = p_car.tolist()
        xyz_car = tuple(xyz_car[:3])

        xyz_car_e = p_car_e.tolist()
        xyz_car_e = tuple(xyz_car_e[:3])

        d = (xyz_car[0][0] - xyz_car_e[0][0])**2 + (xyz_car[1][0] - xyz_car_e[1][0])**2 + \
            (xyz_car[2][0] - xyz_car_e[2][0])**2

        ax.quiver(xyz_car[0][0], xyz_car[1][0], xyz_car[2][0],
                  xyz_car[0][0] - xyz_car_e[0][0],
                  xyz_car[1][0] - xyz_car_e[1][0],
                  xyz_car[2][0] - xyz_car_e[2][0], arrow_length_ratio=0.1, color='blue')

        #print(round(math.sqrt(d), 4))

        poses.append(xyz_car)

    poses_multiplane = []
    poses_multiplane_flip = []
    psphere = [pose_spherical(theta, phi, -1.3) for phi, theta in fibonacci_sphere(6)]

    for p in psphere:
        point = [0, 0, 0, 1]
        point_e = [0, 0, -0.5, 1]
        point = torch.tensor(point).float().view(4, 1)
        point_e = torch.tensor(point_e).float().view(4, 1)

        pm = torch.tensor(p)
        pm = pm @ torch.Tensor([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        p_car = torch.matmul(pm, point)

        #p_car = torch.matmul(torch.tensor(p), point)
        p_car_e = torch.matmul(pm, point_e)
        xyz_car = p_car.tolist()
        xyz_car = tuple(xyz_car[:3])

        xyz_car_e = p_car_e.tolist()
        xyz_car_e = tuple(xyz_car_e[:3])

        #d = (xyz_car[0][0] - xyz_car_e[0][0])**2 + (xyz_car[1][0] - xyz_car_e[1][0])**2 + \
        #    (xyz_car[2][0] - xyz_car_e[2][0])**2
        #print(round(math.sqrt(d), 4))

        ax.quiver(xyz_car[0][0], xyz_car[1][0], xyz_car[2][0],
                  xyz_car[0][0] - xyz_car_e[0][0],
                  xyz_car[1][0] - xyz_car_e[1][0],
                  xyz_car[2][0] - xyz_car_e[2][0], arrow_length_ratio=0.1, color='red')

        poses_multiplane.append(xyz_car)

    for p in psphere:
        point = [0, 0, 0, 1]
        point = torch.tensor(point).float().view(4, 1)
        pm = torch.tensor(p)
        pm = pm @ torch.Tensor([[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        p_car = torch.matmul(pm, point)

        xyz_car = p_car.tolist()
        xyz_car = tuple(xyz_car[:3])
        poses_multiplane_flip.append(xyz_car)


    # for poses in poses_multiplane:
    #     p = list(poses).copy()
    #     p.append(1.0)
    #     print(p)
    #     p = torch.tensor(p).float().view(4, 1)
    #     p = p @ torch.Tensor([[-1, 0, 0, 0], [0, 1, 0, 0],  [0, 0, 1, 0], [0, 0, 0, 1]])
    #     p = p.tolist()



    for pose in poses:

        ax.scatter(*pose, color='b', s=5)

    for pose in poses_multiplane:
        ax.scatter(*pose, color='r', s=5)
    #for pose in poses_multiplane_flip:
    #   ax.scatter(*pose, color='y', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Poses on Sphere')

    plt.show()


if __name__ == "__main__":
    main()