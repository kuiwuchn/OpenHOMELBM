import numpy as np

def generate_unit_circle_points(n):
    """
    Generate evenly spaced points on a unit circle.
    
    Args:
        n (int): Number of points and closed segments.
    
    Returns:
        list: Ordered points forming the circle.
    """
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)  # Exclude the duplicate endpoint.
    points = [(np.cos(theta), np.sin(theta)) for theta in angles]
    return points

def generate_water_drop_points(n):

    
    points = []
    
    
    # Generate the parabolic lower half.
    theta_lower = np.linspace(0, 2*np.pi, n)
    for theta in theta_lower:
        x =3*(1+np.cos(theta))*(2+np.cos(theta))/(5+4*np.cos(theta))
        y =3*(1+np.cos(theta))* np.sin(theta)/(5+4*np.cos(theta)) 
        points.append((x, y))
        # print(x, y)
    
    # Order points clockwise from the top.
    # points = points[-n_upper:] + points[:-n_upper]
    
    return points
