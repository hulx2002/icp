from datapoints import *
import numpy as np
from math import *

class ICP():
    def __init__(self, P: DataPoints, Q: DataPoints, num_P: int, num_Q: int, epsilon1: float, epsilon2: float):
        self.P = P
        self.Q = Q
        self.P_sample = DataPoints()
        self.P_sample.sample(self.P, num_P)
        self.Q_sample = DataPoints()
        self.Q_sample.sample(self.Q, num_Q)
        self.Q_closest = np.zeros_like(self.P_sample.positions)
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.T = np.eye(4, 4)

    def closest_point(self, P: DataPoints, Q:DataPoints):
        Q_closest = np.zeros_like(self.P_sample.positions)
        for i in range(P.positions.shape[0]):
            dist_min = 1e8
            for j in range(Q.positions.shape[0]):
                dist = np.linalg.norm(np.cross(P.positions[i] - Q.positions[j], P.normals[i]))
                if dist < dist_min:
                    Q_closest[i] = Q.positions[j]
                    dist_min = dist
        return Q_closest

    def calculate_T(self, P: np.ndarray, Q: np.ndarray, epsilon1: float, epsilon2: float):
        meanP = np.mean(P, axis=0)
        meanQ = np.mean(Q, axis=0)
        dP = P - meanP
        dQ = Q - meanQ
        B = np.zeros((4, 4))
        d = np.linalg.norm(P - Q, axis=1)
        for i in range(P.shape[0]):
            A = np.array([[0, dQ[i, 0] - dP[i, 0], dQ[i, 1] - dP[i, 1], dQ[i, 2] - dP[i, 2]],
                        [dP[i, 0] - dQ[i, 0], 0, dP[i, 2] + dQ[i, 2], -(dP[i, 1] + dQ[i, 1])],
                        [dP[i, 1] - dQ[i, 1], -(dP[i, 2] + dQ[i, 2]), 0, dP[i, 0] + dQ[i, 0]],
                        [dP[i, 2] - dQ[i, 2], dP[i, 1] + dQ[i, 1], -(dP[i, 0] + dQ[i, 0]), 0]])
            if d[i] <= epsilon1:
                B += A.T @ A
            elif d[i] <= epsilon2:
                B += (epsilon2 - d[i]) / (epsilon2 - epsilon1) * (A.T @ A)
        eigenvalues, eigenvectors = np.linalg.eig(B)
        q4 = eigenvectors[:, np.argmin(eigenvalues)]
        th = 2 * acos(q4[0])
        if th == 0:
            la = 0
            mu = 0
            nu = 0
        else:
            la = q4[1] / sin(th / 2)
            mu = q4[2] / sin(th / 2)
            nu = q4[3] / sin(th / 2)
        ct = cos(th)
        st = sin(th)
        R = np.array([[ct + la ** 2 * (1 - ct), la * mu * (1 - ct) - nu * st, nu * la * (1 - ct) + mu * st],
                    [la * mu * (1 - ct) + nu * st, ct + mu ** 2 * (1 - ct), mu * nu * (1 - ct) - la * st],
                    [nu * la * (1 - ct) - mu * st, mu * nu * (1 - ct) + la * st, ct + nu ** 2 * (1 - ct)]])
        T = meanQ - R @ meanP
        T4 = np.eye(4, 4)
        T4[0:3, 0:3] = R
        T4[0:3, 3] = T
        return T4

    def one_step(self):
        self.Q_closest = self.closest_point(self.P_sample, self.Q_sample)
        T4 = self.calculate_T(self.P_sample.positions, self.Q_closest, self.epsilon1, self.epsilon2)
        self.T = T4 @ self.T
        self.P_sample.transform(T4)
            
    def execute(self):
        for i in range(3):
            self.one_step()
        self.P.transform(self.T)
        return self.P