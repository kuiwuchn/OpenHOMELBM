import numpy as np

def generate_unit_circle_points(n):
    """
    生成单位圆上的n个等距点（首尾不重复）
    
    参数:
        n (int): 点的数量（线段数量 = 点数，因首尾闭合）
    
    返回:
        list: 点的列表 [(x1, y1), (x2, y2), ..., (xn, yn)]，顺序连接形成圆
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)  # 等分角度，不包括终点（避免首尾重复）
    points = [(np.cos(theta), np.sin(theta)) for theta in angles]
    return points

def generate_water_drop_points(n):

    
    points = []
    
    
    # 生成下半部分（抛物线形状）
    theta_lower = np.linspace(0, 2*np.pi, n)
    for theta in theta_lower:
        x =3*(1+np.cos(theta))*(2+np.cos(theta))/(5+4*np.cos(theta))
        y =3*(1+np.cos(theta))* np.sin(theta)/(5+4*np.cos(theta)) 
        points.append((x, y))
        # print(x, y)
    
    # 调整点顺序，使水滴从顶部开始顺时针生成
    # points = points[-n_upper:] + points[:-n_upper]
    
    return points