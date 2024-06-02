import numpy as np
import line_class
import physics_engine


def path_creator(l1, goal_line):
    """computes the path we want, avoiding to create a line on the arrival"""
    line_list = [line_class.Line(physics_engine.universal_scale_factor * l1[i][0],
                                 physics_engine.universal_scale_factor * l1[i][1],
                                 physics_engine.universal_scale_factor * l1[i + 1][0],
                                 physics_engine.universal_scale_factor * l1[i + 1][1])
                 for i in range(len(l1) - 1) if
                 ((goal_line[0][0] != l1[i:i + 2][0][0] or goal_line[0][1] != l1[i:i + 2][0][1]) or (
                         goal_line[1][0] != l1[i:i + 2][1][0] or goal_line[1][1] != l1[i:i + 2][1][1]))]
    line_list.append(line_class.Line(physics_engine.universal_scale_factor * l1[0][0],
                                     physics_engine.universal_scale_factor * l1[0][1],
                                     physics_engine.universal_scale_factor * l1[-1][0],
                                     physics_engine.universal_scale_factor * l1[-1][1]))
    return line_list


def level1():
    point_a = np.array(([50, 50]))
    point_b = np.array(([250, 50]))
    point_c = np.array(([450, 100]))
    point_d = np.array(([600, 200]))
    point_e = np.array(([650, 350]))
    point_f = np.array(([650, 450]))
    point_g = np.array(([600, 600]))
    point_h = np.array(([450, 750]))
    point_i = np.array(([350, 850]))
    point_j = np.array(([400, 1000]))
    point_k = np.array(([600, 1050]))
    point_l = np.array(([750, 1050]))
    point_m = np.array(([850, 950]))
    point_n = np.array(([900, 850]))
    point_o = np.array(([900, 500]))
    point_p = np.array(([1000, 500]))
    point_q = np.array(([1000, 850]))
    point_r = np.array(([950, 950]))
    point_s = np.array(([800, 1150]))
    point_t = np.array(([600, 1200]))
    point_u = np.array(([400, 1200]))
    point_v = np.array(([250, 1050]))
    point_w = np.array(([200, 900]))
    point_x = np.array(([250, 750]))
    point_y = np.array(([400, 650]))
    point_z = np.array(([450, 550]))
    point_a1 = np.array(([500, 450]))
    point_b1 = np.array(([500, 300]))
    point_c1 = np.array(([400, 200]))
    point_d1 = np.array(([250, 150]))
    point_e1 = np.array(([50, 150]))
    lines = np.array(
        path_creator([point_a, point_b, point_c, point_d, point_e, point_f, point_g, point_h, point_i, point_j, point_k,
                      point_l, point_m, point_n, point_o, point_p, point_q, point_r, point_s, point_t, point_u, point_v,
                      point_w, point_x, point_y, point_z, point_a1, point_b1, point_c1, point_d1, point_e1],
                     [point_o, point_p]))
    return lines
