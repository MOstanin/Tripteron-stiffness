import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def Rx(q):
    T = np.array([[1,         0,          0, 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, np.sin(q),  np.cos(q), 0],
                  [0,         0,          0, 1]], dtype=float)
    return T


def dRx(q):
    T = np.array([[0,          0,          0, 0],
                  [0, -np.sin(q), -np.cos(q), 0],
                  [0,  np.cos(q), -np.sin(q), 0],
                  [0,          0,          0, 0]], dtype=float)
    return T


def Ry(q):
    T = np.array([[ np.cos(q), 0, np.sin(q), 0],
                  [         0, 1,         0, 0],
                  [-np.sin(q), 0, np.cos(q), 0],
                  [         0, 0,         0, 1]], dtype=float)
    return T


def dRy(q):
    T = np.array([[-np.sin(q), 0,  np.cos(q), 0],
                  [         0, 0,          0, 0],
                  [-np.cos(q), 0, -np.sin(q), 0],
                  [         0, 0,          0, 0]], dtype=float)
    return T


def Rz(q):
    T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                  [np.sin(q),  np.cos(q), 0, 0],
                  [        0,          0, 1, 0],
                  [        0,          0, 0, 1]], dtype=float)
    return T


def dRz(q):
    T = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                  [ np.cos(q), -np.sin(q), 0, 0],
                  [         0,          0,  0, 0],
                  [         0,          0,  0, 0]], dtype=float)
    return T


def Tx(x):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTx(x):
    T = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTy(y):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTz(z):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]], dtype=float)
    return T


def fkLeg(T_base, T_tool, q_active, q_passive, theta, link):
    # T_base - transform from global coordinate frame to local one of the leg
    # T_tool - transform from the lask joint of the leg to the tool frame
    # a - active joint variable
    # p - passive joint variable
    # t - vitual joint (spring) variable
    # l - link length

    T_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    T_leg = np.linalg.multi_dot([T_base, T_leg_local, T_tool])
    return T_leg


def fkTripteron(T_base, T_tool, q_active, q_passive, theta, link):
    T = []
    for leg in range(len(T_base)):
        T_leg = fkLeg(T_base[leg], T_tool[leg], q_active[leg], q_passive[leg], theta[leg], link)
        T.append(T_leg)
    return T


def ikLeg(T_base, p_global, link, flag):
    # T_base - transform from global coordinate frame to local one of the leg
    R_base = T_base[0:3, 0:3]
    p_base = T_base[0:3, 3]
    p_local = np.transpose(R_base).dot(p_global - p_base)

    x = p_local[0]
    y = p_local[1]
    z = p_local[2]

    cos_q2 = (x**2 + y**2 - link[0]**2 - link[1]**2)/(2*link[0]*link[1])
    sin_q2 = flag*np.sqrt(1 - cos_q2**2)
    q2 = np.arctan2(sin_q2, cos_q2)
    q1 = np.arctan2(y, x) - np.arctan2(link[1]*np.sin(q2), link[0] + link[1]*np.cos(q2))
    q3 = -(q1 + q2)
    q = np.array([q1, q2, q3], dtype=float)
    return q


def ikTripteron(T_base, p_global, link, flag):
    q = []
    for leg in range(len(T_base)):
        q_leg = ikLeg(T_base[leg], p_global, link, flag)
        q.append(q_leg)
    return q


def plotTripteron(T_base, p_global, q_passive, theta, link):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(0, space_x)
    ax.set_ylim3d(0, space_y)
    ax.set_zlim3d(0, space_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    r = [0,1]
    X, Y = np.meshgrid(r, r)
    ones = np.ones(4).reshape(2, 2)
    zeros = np.zeros(4).reshape(2, 2)
    ax.plot_wireframe(X,Y,ones, alpha=0.5, color='slategray')
    ax.plot_wireframe(X,Y,zeros, alpha=0.5, color='slategray')
    ax.plot_wireframe(X,zeros,Y, alpha=0.5, color='slategray')
    ax.plot_wireframe(X,ones,Y, alpha=0.5, color='slategray')
    ax.plot_wireframe(ones,X,Y, alpha=0.5, color='slategray')
    ax.plot_wireframe(zeros,X,Y, alpha=0.5, color='slategray')

    for i in range(len(T_base)):
        q = q_passive[i]
        toOrigin = T_base[i]
        t = theta[i]
        origin = toOrigin[0:3, 3]

        toActive1 = np.linalg.multi_dot([toOrigin, # T_base transform
                                        Tz(p_global[i]), # active joint
                                        Tz(t[0])]) # 1 DOF virtual spring 
        active1 = toActive1[0:3, 3]

        toPassive1 = np.linalg.multi_dot([toActive1, # transfrom to the active joint
                                          Rz(q[0])]) # passive joint
        passive1 = toPassive1[0:3, 3]

        toPassive2 = np.linalg.multi_dot([toPassive1, # transform to the passive joint
                                          Tx(link[0]), # rigid link
                                          Tx(t[1]), Ty(t[2]), Tz(t[3]), Rx(t[4]), Ry(t[5]), Rz(t[6]), # 6 DOF virtual spring
                                          Rz(q[1])]) # passive joint
        passive2 = toPassive2[0:3, 3]

        toPassive3 = np.linalg.multi_dot([toPassive2, # transform to the passive joint
                                          Tx(link[1]), # rigid link
                                          Tx(t[7]), Ty(t[8]), Tz(t[9]), Rx(t[10]), Ry(t[11]), Rz(t[12]), # 6 DOF virtual spring
                                          Rz(q[2])]) # passive joint
        passive3 = toPassive3[0:3, 3]

        leg = [[], [], []]
        active = [[], [], []]

        for i in range(len(leg)):
            active[i].append(origin[i])
            active[i].append(active1[i])
            leg[i].append(active1[i])
            leg[i].append(passive1[i])
            leg[i].append(passive2[i])
            leg[i].append(passive3[i])

        ax.plot3D(active[0], active[1], active[2], c='navy', linewidth=5)
        ax.plot3D(leg[0], leg[1], leg[2], c='steelblue', linewidth=3)

    point = []
    for i in range(len(T_base)):
        point.append(p_global[i])
    ax.scatter3D(point[0], point[1], point[2], c='red', s=10)
    plt.show()


def JacobianPassiveLeg(T_fk, T_base, T_tool, q_active, q_passive, theta, link):
    T_fk[0:3, 3] = 0
    inv_T_fk = np.transpose(T_fk)

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       dRz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J1 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       dRz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J2 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       dRz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J3 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    J = np.hstack([J1, J2, J3])
    return J


def JacobianPassiveTripteron(T_base, T_tool, q_active, q_passive, theta, link):
    T_fk = fkTripteron(T_base, T_tool, q_active, q_passive, theta, link)

    Jq = []
    for leg in range(len(T_base)):
        J = JacobianPassiveLeg(T_fk[leg], T_base[leg], T_tool[leg], q_active[leg], q_passive[leg], theta[leg], link)
        Jq.append(J)
    return Jq

def JacobianThetaLeg(T_fk, T_base, T_tool, q_active, q_passive, theta, link):
    T_fk[0:3, 3] = 0
    inv_T_fk = np.transpose(T_fk)

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       dTz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J1 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       dTx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J2 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), dTy(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J3 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), dTz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J4 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])
    
    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), dRx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J5 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), dRy(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J6 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), dRz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J7 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       dTx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J8 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), dTy(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J9 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), dTz(theta[9]), Rx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J10 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), dRx(theta[10]), Ry(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J11 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), dRy(theta[11]), Rz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J12 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]), # active joint 
                                       Tz(theta[0]), # 1 DOF virtual spring
                                       Rz(q_passive[0]), # passive joint
                                       Tx(link[0]), # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]), Rz(theta[6]), # 6 DOF virtual spring
                                       Rz(q_passive[1]), # passive joint
                                       Tx(link[1]), # rigid link 
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]), dRz(theta[12]), # 6 DOF virtual spring
                                       Rz(q_passive[2]) # passive joint
                                      ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J13 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    J = np.hstack([J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13])
    return J


def JacobianThetaTripteron(T_base, T_tool, q_active, q_passive, theta, link):
    T_fk = fkTripteron(T_base, T_tool, q_active, q_passive, theta, link)

    Jtheta = []
    for leg in range(len(T_base)):
        J = JacobianThetaLeg(T_fk[leg], T_base[leg], T_tool[leg], q_active[leg], q_passive[leg], theta[leg], link)
        Jtheta.append(J)
    return Jtheta


def elementStiffness11(E, G, d, link):
    S = np.pi*(d**2)/4
    Iy = np.pi*(d**4)/64
    Iz = np.pi*(d**4)/64
    J = Iy + Iz
    
    K = np.array([[E*S/link,                 0,                 0,        0,                 0,                0],
                  [       0, 12*E*Iz/(link**3),                 0,        0,                 0, 6*E*Iz/(link**2)],
                  [       0,                 0, 12*E*Iy/(link**3),        0, -6*E*Iy/(link**2),                0],
                  [       0,                 0,                 0, G*J/link,                 0,                0],
                  [       0,                 0, -6*E*Iy/(link**2),        0,       4*E*Iy/link,                0],
                  [       0,  6*E*Iz/(link**2),                 0,        0,                 0,      4*E*Iz/link]], dtype=float)
    
    return K

def elementStiffness12(E, G, d, link):
    S = np.pi*(d**2)/4
    Iy = np.pi*(d**4)/64
    Iz = np.pi*(d**4)/64
    J = Iy + Iz
    
    K = np.array([[-E*S/link,                 0,                 0,        0,                 0,                0],
                  [       0, -12*E*Iz/(link**3),                 0,        0,                 0, 6*E*Iz/(link**2)],
                  [       0,                 0, -12*E*Iy/(link**3),        0, -6*E*Iy/(link**2),                0],
                  [       0,                 0,                 0, -G*J/link,                 0,                0],
                  [       0,                 0, 6*E*Iy/(link**2),        0,       2*E*Iy/link,                0],
                  [       0,  -6*E*Iz/(link**2),                 0,        0,                 0,      2*E*Iz/link]], dtype=float)
    
    return K

def elementStiffness22(E, G, d, link):
    S = np.pi*(d**2)/4
    Iy = np.pi*(d**4)/64
    Iz = np.pi*(d**4)/64
    J = Iy + Iz
    
    K = np.array([[E*S/link,                 0,                 0,        0,                 0,                0],
                  [       0, 12*E*Iz/(link**3),                 0,        0,                 0, -6*E*Iz/(link**2)],
                  [       0,                 0, 12*E*Iy/(link**3),        0, 6*E*Iy/(link**2),                0],
                  [       0,                 0,                 0, G*J/link,                 0,                0],
                  [       0,                 0, 6*E*Iy/(link**2),        0,       4*E*Iy/link,                0],
                  [       0, -6*E*Iz/(link**2),                 0,        0,                 0,      4*E*Iz/link]], dtype=float)
    
    return K

def KThetaLeg(K_active, E, G, d, link):
    K0 = np.zeros(13, dtype=float)
    K0[0] = K_active

    zeros_6_1 = np.zeros((6,1), dtype=float)
    zeros_6_6 = np.zeros((6,6), dtype=float)

    K1 = elementStiffness22(E, G, d[0], link[0])
    K1 = np.hstack([zeros_6_1, K1, zeros_6_6])

    K2 = elementStiffness22(E, G, d[1], link[1])
    K2 = np.hstack([zeros_6_1, zeros_6_6, K2])

    K = np.vstack([K0, K1, K2])
    return K


def KcTripteronVJM(Ktheta, Jq, Jtheta):
    Kc_total = []
    for i in range(len(Ktheta)):
        Kc0 = np.linalg.inv(np.linalg.multi_dot([Jtheta[i], np.linalg.inv(Ktheta[i]), np.transpose(Jtheta[i])]))
        Kc = Kc0 - np.linalg.multi_dot([Kc0, Jq[i], np.linalg.inv(np.linalg.multi_dot([np.transpose(Jq[i]), Kc0, Jq[i]])), np.transpose(Jq[i]), Kc0])
        Kc_total.append(Kc)

    Kc_total = Kc_total[0] + Kc_total[1] + Kc_total[2]
    return Kc_total


def dtTripteron(Kc, F):
    dt = np.linalg.inv(Kc).dot(F)
    return dt

def plotDeflection(x, y, z, deflection, cmap, s):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(0, space_x)
    ax.set_ylim3d(0, space_y)
    ax.set_zlim3d(0, space_z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    r = [0,1]
    X, Y = np.meshgrid(r, r)
    ones = np.ones(4).reshape(2, 2)
    zeros = np.zeros(4).reshape(2, 2)
    ax.plot_wireframe(X,Y,ones, alpha=0.5, color='slategray')
    ax.plot_wireframe(X,Y,zeros, alpha=0.5, color='slategray')
    ax.plot_wireframe(X,zeros,Y, alpha=0.5, color='slategray')
    ax.plot_wireframe(X,ones,Y, alpha=0.5, color='slategray')
    ax.plot_wireframe(ones,X,Y, alpha=0.5, color='slategray')
    ax.plot_wireframe(zeros,X,Y, alpha=0.5, color='slategray')

    cmap = ax.scatter3D(x, y, z, c=deflection, cmap=cmap, s=s)
    plt.colorbar(cmap)
    plt.show()



space_x = space_y = space_z = 1.0 # workspace size
link = np.array([0.75, 0.75], dtype=float) # links length
d = np.array([0.15, 0.15], dtype=float) # links diameter

T_base_z = np.eye(4, dtype=float) # since the global coordinate frame coinside with the local frame of the origin of the leg Z
T_base_y = np.linalg.multi_dot([Tz(space_z), Rx(-np.pi/2)])
T_base_x = np.linalg.multi_dot([Ty(space_y), Ry(np.pi/2), Rz(np.pi)])
T_base = [T_base_x, T_base_y, T_base_z]

T_tool_z = np.eye(4, dtype=float)
T_tool_y = np.transpose(Rx(-np.pi/2))
T_tool_x = np.transpose(np.linalg.multi_dot([Ry(np.pi/2), Rz(np.pi)]))
T_tool = [T_tool_x, T_tool_y, T_tool_z]

theta = np.zeros(13, dtype=float)
theta = [theta, theta, theta]

flag = 1 # '+1' elbow-down or '-1' elbow-up

K_active = 1000000 # actuator stiffness
E = 7.0000e+10 # Young's modulus
G = 2.5500e+10 # shear modulus
Ktheta = KThetaLeg(K_active, E, G, d, link)
Ktheta = [Ktheta, Ktheta, Ktheta]
F = np.array([[0], [0], [1000], [0], [0], [0]], dtype=float)


xScatter = np.array([])
yScatter = np.array([])
zScatter = np.array([])
dScatter = np.array([])

start = 0.01
step = 0.1
step_z = 0.1
for z in np.arange(start, space_z + start, step_z):
    xData = np.array([])
    yData = np.array([])
    zData = np.array([])
    dData = np.array([])
    for x in np.arange(start, space_x + start, step):
        for y in np.arange(start, space_y + start, step):
            try:
                print(x, y, z)
            
                p_global = np.array([x, y, z], dtype=float)
                q_active = [[p_global[0]],[p_global[1]],[p_global[2]]]

                q_passive = ikTripteron(T_base, p_global, link, flag)
                #plotTripteron(T_base, p_global, q_passive, theta, link)
                
                Jq = JacobianPassiveTripteron(T_base, T_tool, q_active, q_passive, theta, link)
                Jtheta = JacobianThetaTripteron(T_base, T_tool, q_active, q_passive, theta, link)

                Kc = KcTripteronVJM(Ktheta, Jq, Jtheta)

                dt = dtTripteron(Kc, F)
                deflection = np.sqrt(dt[0]**2 + dt[1]**2 + dt[2]**2)
                
                xData = np.append(xData, x)
                yData = np.append(yData, y)
                zData = np.append(zData, z)
                dData = np.append(dData, deflection)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                print(error)
                pass

    xScatter = np.append(xScatter, xData)
    yScatter = np.append(yScatter, yData)
    zScatter = np.append(zScatter, zData)
    dScatter = np.append(dScatter, dData)

    N = int((dData.shape[0])**.5)
    dData = dData.reshape(N, N)
    
    cmap = plt.cm.get_cmap('RdGy_r', 12)

    plt.imshow(dData, extent=(np.amin(xData), np.amax(xData), np.amin(yData), np.amax(yData)), cmap=cmap, interpolation='lanczos')
    plt.colorbar()
    #plt.clim(0.00018, 0.00022)

    filename = './maps/VJM_z_' + str(z)
    #plt.savefig(filename + '.svg', format="svg")
    plt.savefig(filename + '.jpg', format="jpg")
    plt.close()

plotDeflection(xScatter, yScatter, zScatter, dScatter, 'RdGy_r', 60)


def transformStiffness(T_base, p_global, q_passive, link):
    Q = []
    for i in range(len(T_base)):
        q = q_passive[i]
        toOrigin = T_base[i]
        origin = toOrigin[0:3, 3]

        toLink1 = np.linalg.multi_dot([toOrigin, # T_base transform
                                       Tz(p_global[i]), # active joint
                                       Rz(q[0])]) # passive joint
        rotationLink1 = toLink1[0:3, 0:3]

        toLink2 = np.linalg.multi_dot([toLink1, # transform to the passive joint
                                       Tx(link[0]), # rigid link
                                       Rz(q[1])]) # passive joint
        rotationLink2 = toLink2[0:3, 0:3]

        zeros = np.zeros((3,3), dtype=float)

        Q1 = np.vstack([np.hstack([rotationLink1,         zeros]),
                        np.hstack([        zeros, rotationLink1])])

        Q2 = np.vstack([np.hstack([rotationLink2,         zeros]),
                        np.hstack([        zeros, rotationLink2])])

        Q.append([Q1, Q2])
    return Q

K11 = elementStiffness11(E, G, d[0], link[0])
K12 = elementStiffness12(E, G, d[0], link[0])
K21 = np.transpose(K12)
K22 = elementStiffness22(E, G, d[0], link[0])

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
np.set_printoptions(precision=2, suppress=True, linewidth=200)

lambda_r_12_x = np.array([[0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_e_12_x = np.array([1, 0, 0, 0, 0, 0], dtype=float)

lambda_r_12_y = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_e_12_y = np.array([0, 1, 0, 0, 0, 0], dtype=float)

lambda_r_12_z = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_e_12_z = np.array([0, 0, 1, 0, 0, 0], dtype=float)

lambda_r_12 = [lambda_r_12_x, lambda_r_12_y, lambda_r_12_z]
lambda_e_12 = [lambda_e_12_x, lambda_e_12_y, lambda_e_12_z]

lambda_r_34_x = lambda_r_56_x = lambda_r_78_x  = np.array([[1, 0, 0, 0, 0, 0],
                                                           [0, 1, 0, 0, 0, 0],
                                                           [0, 0, 1, 0, 0, 0],
                                                           [0, 0, 0, 0, 1, 0],
                                                           [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_p_34_x = lambda_p_56_x = lambda_p_78_x = np.array([0, 0, 0, 1, 0, 0], dtype=float)

lambda_r_34_y = lambda_r_56_y = lambda_r_78_y  = np.array([[1, 0, 0, 0, 0, 0],
                                                           [0, 1, 0, 0, 0, 0],
                                                           [0, 0, 1, 0, 0, 0],
                                                           [0, 0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_p_34_y = lambda_p_56_y = lambda_p_78_y = np.array([0, 0, 0, 0, 1, 0], dtype=float)

lambda_r_34_z = lambda_r_56_z = lambda_r_78_z  = np.array([[1, 0, 0, 0, 0, 0],
                                                           [0, 1, 0, 0, 0, 0],
                                                           [0, 0, 1, 0, 0, 0],
                                                           [0, 0, 0, 1, 0, 0],
                                                           [0, 0, 0, 0, 1, 0]], dtype=float)

lambda_p_34_z = lambda_p_56_z = lambda_p_78_z = np.array([0, 0, 0, 0, 0, 1], dtype=float)

lambda_r_34 = [lambda_r_34_x, lambda_r_34_y, lambda_r_34_z]
lambda_r_56 = [lambda_r_56_x, lambda_r_56_y, lambda_r_56_z]
lambda_r_78 = [lambda_r_78_x, lambda_r_78_y, lambda_r_78_z]

lambda_p_34 = [lambda_p_34_x, lambda_p_34_y, lambda_p_34_z]
lambda_p_56 = [lambda_p_56_x, lambda_p_56_y, lambda_p_56_z]
lambda_p_78 = [lambda_p_78_x, lambda_p_78_y, lambda_p_78_z]

def KcTripteronMSA(Q, K11, K12, K21, K22, lambda_e_12, lambda_r_12, lambda_r_34, lambda_r_56, lambda_r_78, lambda_p_34, lambda_p_56, lambda_p_78):
    Kc = []
    for i in range(len(Q)):
        # Equation 1
        eq1 = np.hstack([np.zeros((6, 6*9), dtype=float), np.eye(6, dtype=float), np.zeros((6, 6*8), dtype=float)])

        Q1 = Q[i][0]
        K1_11 = np.linalg.multi_dot([Q1, K11, np.transpose(Q1)])
        K1_12 = np.linalg.multi_dot([Q1, K12, np.transpose(Q1)])
        K1_21 = np.linalg.multi_dot([Q1, K21, np.transpose(Q1)])
        K1_22 = np.linalg.multi_dot([Q1, K22, np.transpose(Q1)])

        # Equation 2
        eq2 = np.hstack([np.zeros((6, 6*3), dtype=float), -np.eye(6, dtype=float), np.zeros((6, 6*8), dtype=float), K1_11, K1_12, np.zeros((6, 6*4), dtype=float)])

        # Equation 3
        eq3 = np.hstack([np.zeros((6, 6*4), dtype=float), -np.eye(6, dtype=float), np.zeros((6, 6*7), dtype=float), K1_21, K1_22, np.zeros((6, 6*4), dtype=float)])

        Q2 = Q[i][1]
        K2_11 = np.linalg.multi_dot([Q2, K11, np.transpose(Q2)])
        K2_12 = np.linalg.multi_dot([Q2, K12, np.transpose(Q2)])
        K2_21 = np.linalg.multi_dot([Q2, K21, np.transpose(Q2)])
        K2_22 = np.linalg.multi_dot([Q2, K22, np.transpose(Q2)])

        # Equation 4
        eq4 = np.hstack([np.zeros((6, 6*5), dtype=float), -np.eye(6, dtype=float), np.zeros((6, 6*8), dtype=float), K2_11, K2_12, np.zeros((6, 6*2), dtype=float)])

        # Equation 5
        eq5 = np.hstack([np.zeros((6, 6*6), dtype=float), -np.eye(6, dtype=float), np.zeros((6, 6*7), dtype=float), K2_21, K2_22, np.zeros((6, 6*2), dtype=float)])

        # Equation 6
        eq6 = np.hstack([np.zeros((6, 6*16), dtype=float), np.eye(6, dtype=float), -np.eye(6, dtype=float)])

        # Equation 7
        eq7 = np.hstack([np.zeros((6, 6*7), dtype=float), np.eye(6, dtype=float), np.eye(6, dtype=float), np.zeros((6, 6*9), dtype=float)])

        # Equation 8
        eq8 = np.hstack([np.zeros((6, 6*10), dtype=float), np.eye(6, dtype=float), -np.eye(6, dtype=float), np.zeros((6, 6*6), dtype=float)])

        # Equation 9
        eq9 = np.hstack([np.zeros((6, 6*1), dtype=float), np.eye(6, dtype=float), np.eye(6, dtype=float), np.zeros((6, 6*15), dtype=float)])

        # Equation 10
        eq10 = np.hstack([np.zeros((5, 6*9), dtype=float), lambda_r_12[i], -lambda_r_12[i], np.zeros((5, 6*7), dtype=float)])

        # Equation 11
        eq11 = np.hstack([np.eye(6, dtype=float), np.eye(6, dtype=float), np.zeros((6, 6*16), dtype=float)])

        # Equation 12
        eq12 = np.hstack([lambda_e_12[i], np.zeros((6*8), dtype=float), K_active*lambda_e_12[i], -K_active*lambda_e_12[i], np.zeros((6*7), dtype=float)])

        # Equation 13
        eq13 = np.hstack([np.zeros((5, 6*11), dtype=float), lambda_r_34[i], -lambda_r_34[i], np.zeros((5, 6*5), dtype=float)])

        # Equation 14
        eq14 = np.hstack([np.zeros((5, 6*2), dtype=float), lambda_r_34[i], lambda_r_34[i], np.zeros((5, 6*14), dtype=float)])

        # Equation 15
        eq15 = np.hstack([np.zeros((6*2), dtype=float), lambda_p_34[i], np.zeros((6*15), dtype=float)])

        # Equation 16
        eq16 = np.hstack([np.zeros((6*3), dtype=float), lambda_p_34[i], np.zeros((6*14), dtype=float)])

        # Equation 17
        eq17 = np.hstack([np.zeros((5, 6*13), dtype=float), lambda_r_56[i], -lambda_r_56[i], np.zeros((5, 6*3), dtype=float)])

        # Equation 18
        eq18 = np.hstack([np.zeros((5, 6*4), dtype=float), lambda_r_56[i], lambda_r_56[i], np.zeros((5, 6*12), dtype=float)])

        # Equation 19
        eq19 = np.hstack([np.zeros((6*4), dtype=float), lambda_p_56[i], np.zeros((6*13), dtype=float)])

        # Equation 20
        eq20 = np.hstack([np.zeros((6*5), dtype=float), lambda_p_56[i], np.zeros((6*12), dtype=float)])

        # Equation 21
        eq21 = np.hstack([np.zeros((5, 6*15), dtype=float), lambda_r_78[i], -lambda_r_78[i], np.zeros((5, 6*1), dtype=float)])

        # Equation 22
        eq22 = np.hstack([np.zeros((5, 6*6), dtype=float), lambda_r_78[i], lambda_r_78[i], np.zeros((5, 6*10), dtype=float)])

        # Equation 23
        eq23 = np.hstack([np.zeros((6*6), dtype=float), lambda_p_78[i], np.zeros((6*11), dtype=float)])

        # Equation 24
        eq24 = np.hstack([np.zeros((6*7), dtype=float), lambda_p_78[i], np.zeros((6*10), dtype=float)])

        # Equation 25
        eq25 = np.hstack([np.zeros((6, 6*8), dtype=float), -np.eye(6, dtype=float), np.zeros((6, 6*9), dtype=float)])

        # Aggregated matrix
        agg = np.vstack([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13, eq14, eq15, eq16, eq17, eq18, eq19, eq20, eq21, eq22, eq23, eq24, eq25])

        A = agg[0:102, 0:102]
        B = agg[0:102, 102:108]
        C = agg[102:108, 0:102]
        D = agg[102:108, 102:108]

        K_leg = D - np.linalg.multi_dot([C, np.linalg.inv(A), B])
        Kc.append(K_leg)
    Kc = Kc[0] + Kc[1] + Kc[2]
    return Kc








xScatter = np.array([])
yScatter = np.array([])
zScatter = np.array([])
dScatter = np.array([])

start = 0.01
step = 0.1
step_z = 0.1
for z in np.arange(start, space_z + start, step_z):
    xData = np.array([])
    yData = np.array([])
    zData = np.array([])
    dData = np.array([])
    for x in np.arange(start, space_x + start, step):
        for y in np.arange(start, space_y + start, step):
            try:
                print(x, y, z)
            
                p_global = np.array([x, y, z], dtype=float)

                q_passive = ikTripteron(T_base, p_global, link, flag)
                #plotTripteron(T_base, p_global, q_passive, theta, link)

                Q = transformStiffness(T_base, p_global, q_passive, link)
            
                Kc = KcTripteronMSA(Q, K11, K12, K21, K22, lambda_e_12, lambda_r_12, lambda_r_34, lambda_r_56, lambda_r_78, lambda_p_34, lambda_p_56, lambda_p_78)

                dt = dtTripteron(Kc, F)
                deflection = np.sqrt(dt[0]**2 + dt[1]**2 + dt[2]**2)
                
                xData = np.append(xData, x)
                yData = np.append(yData, y)
                zData = np.append(zData, z)
                dData = np.append(dData, deflection)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as error:
                print(error)
                pass

    xScatter = np.append(xScatter, xData)
    yScatter = np.append(yScatter, yData)
    zScatter = np.append(zScatter, zData)
    dScatter = np.append(dScatter, dData)

    N = int((dData.shape[0])**.5)
    dData = dData.reshape(N, N)

    cmap = plt.cm.get_cmap('RdGy_r', 12)

    plt.imshow(dData, extent=(np.amin(xData), np.amax(xData), np.amin(yData), np.amax(yData)), cmap=cmap, interpolation='lanczos')
    plt.colorbar()
    #plt.clim(0.00018, 0.00022)

    filename = './maps/MSA_z_' + str(z)
    #plt.savefig(filename + '.svg', format="svg")
    plt.savefig(filename + '.jpg', format="jpg")
    plt.close()

plotDeflection(xScatter, yScatter, zScatter, dScatter, 'RdGy_r', 60)
