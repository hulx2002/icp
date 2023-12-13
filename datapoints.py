import numpy as np
import open3d as o3d

class DataPoints():
    def __init__(self, num: int = 1):
        self.positions = np.random.randn(num, 3)
        self.normals = np.zeros_like(self.positions)
    
    def loaddata(self, file_path: str):
        self.positions = np.loadtxt(file_path, skiprows=2)
        self.calculate_normals()

    def calculate_normals(self):
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(self.positions)
        pointcloud.estimate_normals()
        self.normals = np.asarray(pointcloud.normals)

    def sample(self, dp: 'DataPoints', size: int):
        indices = np.random.choice(dp.positions.shape[0], size=size, replace=False)
        self.positions = dp.positions[indices]
        self.normals = dp.normals[indices]

    def transform(self, T4: np.ndarray):
        R = T4[0:3, 0:3]
        T = T4[0:3, 3]
        self.positions = np.einsum('ij,kj->ki', R, self.positions) + T
        self.normals = np.einsum('ij,kj->ki', R, self.normals)

    def savedata(self, file_path):
        np.savetxt(file_path, self.positions, delimiter=" ")