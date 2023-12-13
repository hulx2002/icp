from datapoints import *
from icp import *

P = DataPoints()
Q = DataPoints()
P.loaddata("./data/1.asc")
Q.loaddata("./data/2.asc")
icp = ICP(P, Q, 10, 10000, 0.1, 0.2)
P_new = icp.execute()
P_new.savedata('./data/1_new.asc')