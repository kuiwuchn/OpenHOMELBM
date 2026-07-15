import warp as wp
from .lbm_core import HomeFlow

cs2 = wp.constant(wp.float32(1 / 9 + 1 / 9 + 1 / 36 + 1 / 36 + 1 / 36 + 1 / 36))

ML_FLUID = wp.constant(wp.int32(0))
ML_SOLID = wp.constant(wp.int32(1))
# wall
ML_WALL = wp.constant(wp.int32(2))
# invalid
ML_INVALID = wp.constant(wp.int32(6))

OBJ_CIRCLE = wp.constant(0)  # Circle geometry type
OBJ_RECTANGLE = wp.constant(1)  # Square geometry type


@wp.kernel
def InitBoundary(flows: wp.array(dtype=HomeFlow)):
    world_idx, x, y = wp.tid()
    flow = flows[world_idx]
    if flow.flag[x, y] != ML_SOLID and flow.flag[x, y] != ML_INVALID:
        if (x == 0 or x == flow.nx - 1) or (y == 0 or y == flow.ny - 1):
            flow.flag[x, y] = ML_WALL


@wp.kernel
def InitFlow(flows: wp.array(dtype=HomeFlow)):
    world_idx, x, y = wp.tid()
    flow = flows[world_idx]
    rho = 1.0
    ux = 0.0
    uy = 0.0

    pop = wp.types.vector(length=9, dtype=wp.float32)
    U_sqr = ux * ux + uy * uy

    for i in range(9):
        cu = flow.cx_d2q9[i] * ux + flow.cy_d2q9[i] * uy
        pop[i] = rho * flow.w_d2q9[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr)

    invRho = 1.0 / rho

    pixx = pop[1] + pop[2] + pop[5] + pop[6] + pop[7] + pop[8]
    piyy = pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]
    pixy = pop[5] + pop[7] - pop[6] - pop[8]
    cs2 = pixx  ## Very Important!!!
    pixx = pixx * invRho - cs2
    piyy = piyy * invRho - cs2
    pixy = pixy * invRho

    flow.rho[x, y] = rho
    flow.rho_post[x, y] = rho
    flow.u[x, y] = wp.vec2(ux, uy)
    flow.u_post[x, y] = wp.vec2(ux, uy)
    flow.Sxx[x, y] = pixx
    flow.Sxx_post[x, y] = pixx
    flow.Syy[x, y] = piyy
    flow.Syy_post[x, y] = piyy
    flow.Sxy[x, y] = pixy
    flow.Sxy_post[x, y] = pixy

    flow.forcex[x, y] = 0.0
    flow.forcey[x, y] = 0.0


@wp.kernel
def ResetSingleWorldFlow(
    flows: wp.array(dtype=HomeFlow), reset_mask: wp.array(dtype=wp.int32)
):
    """
    Reset flow field for specific worlds indicated by reset_mask.
    reset_mask[world_idx] = 1 means reset this world, 0 means skip.
    """
    world_idx, x, y = wp.tid()

    # Skip if this world should not be reset
    if reset_mask[world_idx] == 0:
        return

    flow = flows[world_idx]
    rho = 1.0
    ux = 0.0
    uy = 0.0

    pop = wp.types.vector(length=9, dtype=wp.float32)
    U_sqr = ux * ux + uy * uy

    for i in range(9):
        cu = flow.cx_d2q9[i] * ux + flow.cy_d2q9[i] * uy
        pop[i] = rho * flow.w_d2q9[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr)

    invRho = 1.0 / rho

    pixx = pop[1] + pop[2] + pop[5] + pop[6] + pop[7] + pop[8]
    piyy = pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]
    pixy = pop[5] + pop[7] - pop[6] - pop[8]
    cs2_local = pixx  ## Very Important!!!
    pixx = pixx * invRho - cs2_local
    piyy = piyy * invRho - cs2_local
    pixy = pixy * invRho

    flow.rho[x, y] = rho
    flow.rho_post[x, y] = rho
    flow.u[x, y] = wp.vec2(ux, uy)
    flow.u_post[x, y] = wp.vec2(ux, uy)
    flow.Sxx[x, y] = pixx
    flow.Sxx_post[x, y] = pixx
    flow.Syy[x, y] = piyy
    flow.Syy_post[x, y] = piyy
    flow.Sxy[x, y] = pixy
    flow.Sxy_post[x, y] = pixy

    flow.forcex[x, y] = 0.0
    flow.forcey[x, y] = 0.0


@wp.kernel
def ResetSingleWorldSolidTransform(
    flows: wp.array(dtype=HomeFlow), reset_mask: wp.array(dtype=wp.int32)
):
    """
    Reset solid_line_transformed and solid_line_transformed_last for specific worlds.
    This is needed so that the boundary velocity calculation won't have stale data.
    """
    world_idx, solid_id, idx = wp.tid()

    # Skip if this world should not be reset
    if reset_mask[world_idx] == 0:
        return

    flow = flows[world_idx]

    # Check if this segment is valid for this object
    if idx >= flow.solid_line_num[solid_id]:
        return

    # Reset transformed positions to zero (will be properly set on next step)
    flow.solid_line_transformed[solid_id, idx] = wp.vec2(0.0, 0.0)
    flow.solid_line_transformed_last[solid_id, idx] = wp.vec2(0.0, 0.0)


@wp.kernel
def ResetSingleWorldForces(
    flows: wp.array(dtype=HomeFlow), reset_mask: wp.array(dtype=wp.int32)
):
    """
    Reset solid forces and torques for specific worlds.
    """
    world_idx = wp.tid()

    # Skip if this world should not be reset
    if reset_mask[world_idx] == 0:
        return

    flow = flows[world_idx]

    for solid_id in range(flow.n_objects):
        flow.solid_forcex[solid_id] = 0.0
        flow.solid_forcey[solid_id] = 0.0
        flow.torque[solid_id] = 0.0


@wp.kernel
def get_u_img(flow: HomeFlow):
    x, y = wp.tid()
    flow.u_img[x, y] = wp.sqrt(flow.u[x, y][0] ** 2.0 + flow.u[x, y][1] ** 2.0)


@wp.kernel
def get_velocity_rgb(flow: HomeFlow, max_velocity: float):
    """
    Build a three-channel velocity image:
    - R: velocity magnitude
    - G: normalized x velocity
    - B: normalized y velocity

    Map velocity from [-max, max] to [0, 1].
    """
    x, y = wp.tid()
    ux = flow.u[x, y][0]
    uy = flow.u[x, y][1]

    # R: normalized velocity magnitude.
    magnitude = wp.sqrt(ux * ux + uy * uy)
    r = wp.clamp(magnitude / max_velocity, 0.0, 1.0)

    # G: normalized x velocity.
    g = wp.clamp((ux + max_velocity) / (2.0 * max_velocity), 0.0, 1.0)

    # B: normalized y velocity.
    b = wp.clamp((uy + max_velocity) / (2.0 * max_velocity), 0.0, 1.0)

    flow.u_img_rgb[x, y] = wp.vec3(r, g, b)


@wp.kernel
def get_vorticity_img(flow: HomeFlow):
    """
    Compute vorticity as dv/dx - du/dy.

    Use central differences internally and one-sided differences at boundaries.
    Store the result in ``flow.u_img``.
    """
    x, y = wp.tid()

    # Use one-sided boundary differences.
    dvdx = 0.0
    dudy = 0.0

    # Compute dv/dx.
    if x == 0:
        # Forward difference at the left boundary.
        dvdx = (flow.u[x + 1, y][1] - flow.u[x, y][1]) / flow.grid_length
    elif x == flow.nx - 1:
        # Backward difference at the right boundary.
        dvdx = (flow.u[x, y][1] - flow.u[x - 1, y][1]) / flow.grid_length
    else:
        # Central difference in the interior.
        dvdx = (flow.u[x + 1, y][1] - flow.u[x - 1, y][1]) / (2.0 * flow.grid_length)

    # Compute du/dy.
    if y == 0:
        # Forward difference at the lower boundary.
        dudy = (flow.u[x, y + 1][0] - flow.u[x, y][0]) / flow.grid_length
    elif y == flow.ny - 1:
        # Backward difference at the upper boundary.
        dudy = (flow.u[x, y][0] - flow.u[x, y - 1][0]) / flow.grid_length
    else:
        # Central difference in the interior.
        dudy = (flow.u[x, y + 1][0] - flow.u[x, y - 1][0]) / (2.0 * flow.grid_length)

    # Store vorticity in u_img.
    flow.u_img[x, y] = dvdx - dudy


@wp.kernel
def get_vorticity_with_solid_img(flow: HomeFlow, thickness: float):
    """
    Compute vorticity and mark solid boundaries.

    vorticity = ∂v/∂x - ∂u/∂y
    Use a sentinel value to distinguish solid regions.
    """
    x, y = wp.tid()

    # Detect solid cells.
    pos = wp.vec2(float(x), float(y))
    min_d = float(1.0e6)
    is_inside = int(0)

    # Iterate over all solids
    for obj_idx in range(flow.n_objects):
        scale = flow.solid_scale[obj_idx] * float(flow.nx)
        center = flow.solid_position[obj_idx]
        max_r = flow.solid_max_radius[obj_idx] * scale * scale

        # Bounding box check (optimization)
        if wp.length_sq(pos - center) > max_r + thickness:
            continue

        n_pts = flow.solid_line_num[obj_idx]
        intersections = int(0)

        for k in range(n_pts):
            p1 = flow.solid_line_transformed[obj_idx, k]
            p2 = flow.solid_line_transformed[obj_idx, (k + 1) % n_pts]

            d = point_segment_dist(pos, p1, p2)
            if d < min_d:
                min_d = d

            # Ray casting
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]

            if (y1 > pos[1]) != (y2 > pos[1]):
                intersect_x = x1 + (pos[1] - y1) * (x2 - x1) / (y2 - y1)
                if pos[0] < intersect_x:
                    intersections = intersections + 1

        if (intersections % 2) == 1:
            is_inside = 1

    # Mark solids with a value above normal vorticity.
    if is_inside == 1 or min_d < thickness:
        flow.u_img[x, y] = 1000.0
        return  # Skip fluid calculations.

    # Compute vorticity for fluid cells.
    # Use one-sided boundary differences.
    dvdx = 0.0
    dudy = 0.0

    # Compute dv/dx.
    if x == 0:
        # Forward difference at the left boundary.
        dvdx = (flow.u[x + 1, y][1] - flow.u[x, y][1]) / flow.grid_length
    elif x == flow.nx - 1:
        # Backward difference at the right boundary.
        dvdx = (flow.u[x, y][1] - flow.u[x - 1, y][1]) / flow.grid_length
    else:
        # Central difference in the interior.
        dvdx = (flow.u[x + 1, y][1] - flow.u[x - 1, y][1]) / (2.0 * flow.grid_length)

    # Compute du/dy.
    if y == 0:
        # Forward difference at the lower boundary.
        dudy = (flow.u[x, y + 1][0] - flow.u[x, y][0]) / flow.grid_length
    elif y == flow.ny - 1:
        # Backward difference at the upper boundary.
        dudy = (flow.u[x, y][0] - flow.u[x, y - 1][0]) / flow.grid_length
    else:
        # Central difference in the interior.
        dudy = (flow.u[x, y + 1][0] - flow.u[x, y - 1][0]) / (2.0 * flow.grid_length)

    # Compute vorticity.
    flow.u_img[x, y] = dvdx - dudy


@wp.kernel
def stream_and_collide(flows: wp.array(dtype=HomeFlow)):
    world_idx, x, y = wp.tid()
    flow = flows[world_idx]
    if flow.flag[x, y] == ML_FLUID:
        pop = wp.types.vector(length=9, dtype=wp.float32)
        rhoVar_cur = flow.rho[x, y]
        ux_cur = flow.u[x, y][0]
        uy_cur = flow.u[x, y][1]
        Sxx_cur = flow.Sxx[x, y]
        Syy_cur = flow.Syy[x, y]
        Sxy_cur = flow.Sxy[x, y]

        for i in range(9):
            dx = flow.cx_d2q9[i]
            dy = flow.cy_d2q9[i]
            x1 = x - int(dx)
            y1 = y - int(dy)

            if x1 >= 0 and x1 < flow.nx and y1 >= 0 and y1 < flow.ny:
                # Check for intersection with any solid
                cutcell_result = get_cutcell(flow, float(x), float(y), dx, dy)
                cutcell = cutcell_result[0]
                solid_id = int(cutcell_result[1])
                segment_ratio = cutcell_result[2]
                segment_id = int(cutcell_result[3])

                if cutcell >= 0.0 and solid_id >= 0:
                    # Intersection with solid detected

                    # Calculate point position using segment_ratio and segment_id
                    # Current time position on segment (material point)
                    c_current = flow.solid_line_transformed[solid_id, segment_id]
                    d_current = flow.solid_line_transformed[
                        solid_id, (segment_id + 1) % flow.solid_line_num[solid_id]
                    ]
                    p_on_segment_current = (
                        c_current + (d_current - c_current) * segment_ratio
                    )

                    # Last time position on segment (same material point)
                    c_last = flow.solid_line_transformed_last[solid_id, segment_id]
                    d_last = flow.solid_line_transformed_last[
                        solid_id, (segment_id + 1) % flow.solid_line_num[solid_id]
                    ]
                    p_on_segment_last = c_last + (d_last - c_last) * segment_ratio

                    # Calculate velocity of the material point: u_p = (p_current - p_last) / dt
                    u_p = (p_on_segment_current - p_on_segment_last) / flow.time_step

                    # Use material point position for force calculation
                    rx = p_on_segment_current[0] - flow.solid_position[solid_id].x
                    ry = p_on_segment_current[1] - flow.solid_position[solid_id].y
                    Sxx_s = u_p[0] * u_p[0] + Sxx_cur - ux_cur * ux_cur
                    Syy_s = u_p[1] * u_p[1] + Syy_cur - uy_cur * uy_cur
                    Sxy_s = u_p[0] * u_p[1] + Sxy_cur - ux_cur * uy_cur
                    pop[i] = mlCalDistribution(
                        rhoVar_cur, u_p[0], u_p[1], Sxx_s, Syy_s, Sxy_s, i
                    )
                    if pop[i] < 0.0:
                        pop[i] = 0.0
                    cur_f = mlCalDistribution(
                        rhoVar_cur,
                        ux_cur,
                        uy_cur,
                        Sxx_cur,
                        Syy_cur,
                        Sxy_cur,
                        flow.indexd2q9Inv_gpu[i],
                    )
                    bndForcex = cur_f * (
                        flow.cx_d2q9[flow.indexd2q9Inv_gpu[i]] - u_p[0]
                    ) - pop[i] * (flow.cx_d2q9[i] - u_p[0])
                    bndForcey = cur_f * (
                        flow.cy_d2q9[flow.indexd2q9Inv_gpu[i]] - u_p[1]
                    ) - pop[i] * (flow.cy_d2q9[i] - u_p[1])
                    bndTorque = (rx * bndForcey - ry * bndForcex) * flow.grid_length
                    wp.atomic_add(flow.solid_forcex, solid_id, bndForcex)
                    wp.atomic_add(flow.solid_forcey, solid_id, bndForcey)
                    wp.atomic_add(flow.torque, solid_id, bndTorque)
                else:
                    # No intersection, regular streaming
                    rhoVar = flow.rho[x1, y1]
                    ux = flow.u[x1, y1][0]
                    uy = flow.u[x1, y1][1]
                    Sxx = flow.Sxx[x1, y1]
                    Syy = flow.Syy[x1, y1]
                    Sxy = flow.Sxy[x1, y1]
                    pop[i] = mlCalDistribution(rhoVar, ux, uy, Sxx, Syy, Sxy, i)
                    if pop[i] < 0.0:
                        pop[i] = 0.0

        Fx = flow.forcex[x, y]
        Fy = flow.forcey[x, y]
        rhoVar = (
            pop[0]
            + pop[1]
            + pop[2]
            + pop[3]
            + pop[4]
            + pop[5]
            + pop[6]
            + pop[7]
            + pop[8]
        )
        invRho = 1.0 / rhoVar
        ux = (pop[1] - pop[2] + pop[5] - pop[6] - pop[7] + pop[8] + 0.5 * Fx) * invRho
        uy = (pop[3] - pop[4] + pop[5] + pop[6] - pop[7] - pop[8] + 0.5 * Fy) * invRho
        pixx = pop[1] + pop[2] + pop[5] + pop[6] + pop[7] + pop[8]
        piyy = pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]
        pixy = pop[5] + pop[7] - pop[6] - pop[8]
        Omega = 1.0 / ((flow.vis_shear) * 3.0 + 0.5)

        # mlGetPIAfterCollision 2D: Appendix C
        Ru2 = rhoVar * ux * ux
        Rv2 = rhoVar * uy * uy
        pixx_part = (pixx - piyy) / 2.0
        piyy_part = (piyy - pixx) / 2.0

        pixy = (
            (1.0 - Omega) * pixy
            + ux * uy * rhoVar * Omega
            + (1.0 - 0.5 * Omega) * (Fy * ux + Fx * uy)
        )
        pixx = (
            rhoVar / 3.0
            + pixx_part * (1.0 - Omega)
            + (1.0 + Omega) / 2.0 * Ru2
            + (1.0 - Omega) / 2.0 * Rv2
            + Fx * ux
            + (1.0 - Omega) / 2.0 * (Fx * ux - Fy * uy)
        )
        piyy = (
            rhoVar / 3.0
            + piyy_part * (1.0 - Omega)
            + (1.0 + Omega) / 2.0 * Rv2
            + (1.0 - Omega) / 2.0 * Ru2
            + Fy * uy
            + (1.0 - Omega) / 2.0 * (Fy * uy - Fx * ux)
        )

        flow.rho_post[x, y] = rhoVar
        flow.u_post[x, y] = wp.vec2(ux + Fx * invRho / 2.0, uy + Fy * invRho / 2.0)
        flow.Sxx_post[x, y] = pixx * invRho - cs2
        flow.Syy_post[x, y] = piyy * invRho - cs2
        flow.Sxy_post[x, y] = pixy * invRho


# def Swap_Mom(flow):
#     flow.rho, flow.rho_post,flow.u, flow.u_post,flow.Sxx, flow.Sxx_post,flow.Syy, flow.Syy_post,flow.Sxy, flow.Sxy_post \
#           = flow.rho_post, flow.rho, flow.u_post, flow.u,flow.Sxx_post, flow.Sxx,flow.Syy_post, flow.Syy,flow.Sxy_post, flow.Sxy


@wp.kernel
def Swap_Mom(flows: wp.array(dtype=HomeFlow)):
    tid = wp.tid()
    f = flows[tid]

    tmp_rho = f.rho
    f.rho = f.rho_post
    f.rho_post = tmp_rho

    tmp_u = f.u
    f.u = f.u_post
    f.u_post = tmp_u

    tmp_Sxx = f.Sxx
    f.Sxx = f.Sxx_post
    f.Sxx_post = tmp_Sxx

    tmp_Syy = f.Syy
    f.Syy = f.Syy_post
    f.Syy_post = tmp_Syy

    tmp_Sxy = f.Sxy
    f.Sxy = f.Sxy_post
    f.Sxy_post = tmp_Sxy

    flows[tid] = f


def init_force(flow):
    flow.solid_forcex.zero_()
    flow.solid_forcey.zero_()
    flow.torque.zero_()


@wp.kernel
def precompute_transformed_segments(flows: wp.array(dtype=HomeFlow)):
    world_idx, solid_id, idx = wp.tid()
    flow = flows[world_idx]

    # Check if this segment is valid for this object
    if idx >= flow.solid_line_num[solid_id]:
        return

    scale = flow.solid_scale[solid_id] * float(flow.nx)
    cos_angle = wp.cos(flow.solid_angle[solid_id])
    sin_angle = wp.sin(flow.solid_angle[solid_id])

    # Get original vertex
    p = flow.solid_line[solid_id, idx]

    # Apply transformations: center -> scale -> rotate -> translate
    pp = (p - flow.solid_mass_center[solid_id]) * scale
    new_x = pp.x * cos_angle - pp.y * sin_angle + flow.solid_position[solid_id].x
    new_y = pp.x * sin_angle + pp.y * cos_angle + flow.solid_position[solid_id].y

    # Check if this is first time (current position is zero)
    old_x = flow.solid_line_transformed[solid_id, idx].x
    old_y = flow.solid_line_transformed[solid_id, idx].y

    if old_x == 0.0 and old_y == 0.0:
        # First time: initialize last to current (zero velocity)
        flow.solid_line_transformed_last[solid_id, idx] = wp.vec2(new_x, new_y)
    else:
        # Normal: save old position to last
        flow.solid_line_transformed_last[solid_id, idx] = wp.vec2(old_x, old_y)

    # Update current position
    flow.solid_line_transformed[solid_id, idx] = wp.vec2(new_x, new_y)


@wp.func
def get_cutcell(
    flow: HomeFlow, x: wp.float32, y: wp.float32, dx: wp.float32, dy: wp.float32
) -> wp.vec4:
    """
    Find the closest intersection with any solid object.
    Returns: vec4(cutcell_distance, solid_id, segment_ratio, segment_id)
    where solid_id=-1 means no intersection
    cutcell_distance: distance ratio along ray (a->b) where intersection occurs
    solid_id: ID of the intersecting solid object
    segment_ratio: position ratio along the solid segment (c->d) where intersection occurs
    segment_id: ID of the intersecting segment's first endpoint
    """
    cutcell = float(-1.0)
    intersect_solid_id = int(-1)
    intersect_segment_ratio = float(-1.0)
    intersect_segment_id = int(-1)

    a = wp.vec2(x - dx, y - dy)
    b = wp.vec2(x, y)

    # Loop through all solid objects
    for solid_id in range(flow.n_objects):
        # Early exit: check distance from point (x,y) to solid center
        point_pos = wp.vec2(x, y)
        solid_center = flow.solid_position[solid_id]
        dist_to_center_sq = wp.length_sq(point_pos - solid_center)
        scale = flow.solid_scale[solid_id] * float(flow.nx)
        scaled_max_radius_sq = flow.solid_max_radius[solid_id] * scale * scale

        if dist_to_center_sq > scaled_max_radius_sq:
            continue  # Skip this solid, too far away

        # Check line segments of this solid
        num_segments = flow.solid_line_num[solid_id]
        for n in range(num_segments):
            c = flow.solid_line_transformed[solid_id, n]
            d = flow.solid_line_transformed[solid_id, (n + 1) % num_segments]

            area_abc = (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)
            area_abd = (a.x - d.x) * (b.y - d.y) - (a.y - d.y) * (b.x - d.x)
            if area_abc * area_abd >= 0.0:
                continue
            area_cda = (c.x - a.x) * (d.y - a.y) - (c.y - a.y) * (d.x - a.x)
            area_cdb = area_cda + area_abc - area_abd
            if area_cda * area_cdb > 0.0:
                continue
            t = 1.0 - area_cda / (area_abd - area_abc)
            if cutcell < 0.0 or t < cutcell:
                cutcell = t
                intersect_solid_id = solid_id
                # Calculate segment ratio: position along segment c->d
                # s = area_abc / (area_abc - area_abd) gives the ratio along c->d
                s = area_abc / (area_abc - area_abd)
                intersect_segment_ratio = s
                intersect_segment_id = n

    return wp.vec4(
        cutcell,
        float(intersect_solid_id),
        intersect_segment_ratio,
        float(intersect_segment_id),
    )


@wp.kernel
def apply_bc(flows: wp.array(dtype=HomeFlow)):  # impose boundary conditions
    world_idx, x, y = wp.tid()
    flow = flows[world_idx]
    # if flow.flag[x, y] == ML_SOLID or flow.flag[x, y] == ML_FLUID:
    #     return
    if x == 0:
        apply_bc_core(
            flow, 1, 0, 0, y, 1, y
        )  # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
    elif x == flow.nx - 1:
        apply_bc_core(
            flow, 1, 2, flow.nx - 1, y, flow.nx - 2, y
        )  # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
    if y == flow.ny - 1:
        apply_bc_core(
            flow, 1, 1, x, flow.ny - 1, x, flow.ny - 2
        )  # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
    elif y == 0:
        apply_bc_core(
            flow, 1, 3, x, 0, x, 1
        )  # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1


@wp.func
def apply_bc_core(
    flow: HomeFlow,
    outer: wp.int32,
    dr: wp.int32,
    ibc: wp.int32,
    jbc: wp.int32,
    inb: wp.int32,
    jnb: wp.int32,
):
    if outer == 1:  # handle outer boundary
        if flow.bc_type[dr] == 0:
            flow.u_post[ibc, jbc][0] = flow.bc_value[dr][0]
            flow.u_post[ibc, jbc][1] = flow.bc_value[dr][1]
        elif flow.bc_type[dr] == 1:
            flow.u_post[ibc, jbc][0] = flow.u_post[inb, jnb][0]
            flow.u_post[ibc, jbc][1] = flow.u_post[inb, jnb][1]
    flow.rho_post[ibc, jbc] = flow.rho_post[inb, jnb]

    pixx_cur, piyy_cur, pixy_cur = f_eq_home(flow, ibc, jbc)
    pixx_back, piyy_back, pixy_back = f_eq_home(flow, inb, jnb)
    flow.Sxx_post[ibc, jbc] = pixx_cur - pixx_back + flow.Sxx_post[inb, jnb]
    flow.Syy_post[ibc, jbc] = piyy_cur - piyy_back + flow.Syy_post[inb, jnb]
    flow.Sxy_post[ibc, jbc] = pixy_cur - pixy_back + flow.Sxy_post[inb, jnb]


@wp.func
def f_eq_home(flow: HomeFlow, x1: wp.int32, y1: wp.int32):
    pop = wp.types.vector(length=9, dtype=wp.float32)
    rho = flow.rho_post[x1, y1]
    ux = flow.u_post[x1, y1].x
    uy = flow.u_post[x1, y1].y
    U_sqr = ux * ux + uy * uy
    for i in range(9):
        cu = flow.cx_d2q9[i] * ux + flow.cy_d2q9[i] * uy
        pop[i] = rho * flow.w_d2q9[i] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr)
        if pop[i] < 0.0:
            pop[i] = 0.0
    invRho = 1.0 / rho

    pixx = pop[1] + pop[2] + pop[5] + pop[6] + pop[7] + pop[8]
    piyy = pop[3] + pop[4] + pop[5] + pop[6] + pop[7] + pop[8]
    pixy = pop[5] + pop[7] - pop[6] - pop[8]

    pixx = pixx * invRho - cs2
    piyy = piyy * invRho - cs2
    pixy = pixy * invRho

    return pixx, piyy, pixy


@wp.func
def mlCalDistribution(
    rhoVar: wp.float32,
    ux: wp.float32,
    uy: wp.float32,
    Sxx: wp.float32,
    Syy: wp.float32,
    Sxy: wp.float32,
    i: wp.int32,
) -> wp.float32:
    Axxy = Sxx * uy + 2.0 * Sxy * ux - 2.0 * ux * ux * uy
    Axyy = Syy * ux + 2.0 * Sxy * uy - 2.0 * ux * uy * uy
    if i == 0:
        f_out = 4.0 / 9.0 * rhoVar * (1.0 - 0.5 * 3.0 * Sxx - 0.5 * 3.0 * Syy)
        return f_out

    elif i == 1:
        f_out = (
            1.0
            / 9.0
            * rhoVar
            * (1.0 + 3.0 * ux + 3.0 * Sxx - 0.5 * 3.0 * Syy - 4.5 * Axyy)
        )
        return f_out
    elif i == 2:
        f_out = (
            1.0
            / 9.0
            * rhoVar
            * (1.0 - 3.0 * ux + 3.0 * Sxx - 0.5 * 3.0 * Syy + 4.5 * Axyy)
        )
        return f_out
    elif i == 3:
        f_out = (
            1.0
            / 9.0
            * rhoVar
            * (1.0 + 3.0 * uy - 0.5 * 3.0 * Sxx + 3.0 * Syy - 4.5 * Axxy)
        )
        return f_out
    elif i == 4:
        f_out = (
            1.0
            / 9.0
            * rhoVar
            * (1.0 - 3.0 * uy - 0.5 * 3.0 * Sxx + 3.0 * Syy + 4.5 * Axxy)
        )
        return f_out

    elif i == 5:
        f_out = (
            1.0
            / 36.0
            * rhoVar
            * (
                1.0
                + 3.0 * (ux + uy)
                + 3.0 * (Sxx + Syy + 3.0 * Sxy)
                + 9.0 * (Axyy + Axxy)
            )
        )
        return f_out
    elif i == 6:
        f_out = (
            1.0
            / 36.0
            * rhoVar
            * (
                1.0
                - 3.0 * ux
                + 3.0 * uy
                + 3.0 * (Sxx + Syy - 3.0 * Sxy)
                + 9.0 * (Axxy - Axyy)
            )
        )
        return f_out
    elif i == 7:
        f_out = (
            1.0
            / 36.0
            * rhoVar
            * (
                1.0
                - 3.0 * ux
                - 3.0 * uy
                + 3.0 * (Sxx + Syy + 3.0 * Sxy)
                - 9.0 * (Axxy + Axyy)
            )
        )
        return f_out
    elif i == 8:
        f_out = (
            1.0
            / 36.0
            * rhoVar
            * (
                1.0
                + 3.0 * ux
                - 3.0 * uy
                + 3.0 * (Sxx + Syy - 3.0 * Sxy)
                - 9.0 * (Axxy - Axyy)
            )
        )
        return f_out


@wp.func
def point_segment_dist(p: wp.vec2, a: wp.vec2, b: wp.vec2):
    ab = b - a
    ap = p - a
    len_sq = wp.dot(ab, ab)
    if len_sq == 0.0:
        return wp.length(ap)
    t = wp.dot(ap, ab) / len_sq
    t = wp.clamp(t, 0.0, 1.0)
    closest = a + ab * t
    return wp.length(p - closest)


@wp.kernel
def get_solid_boundary_img(flow: HomeFlow, thickness: float):
    """
    Render solid boundaries to u_img (1.0 for solid, 0.0 for empty)
    """
    x, y = wp.tid()

    pos = wp.vec2(float(x), float(y))

    min_d = float(1.0e6)
    is_inside = int(0)

    # Iterate over all solids
    for obj_idx in range(flow.n_objects):
        scale = flow.solid_scale[obj_idx] * float(flow.nx)
        center = flow.solid_position[obj_idx]
        max_r = flow.solid_max_radius[obj_idx] * scale * scale

        # Bounding box check (optimization)
        if wp.length_sq(pos - center) > max_r + thickness:
            continue

        n_pts = flow.solid_line_num[obj_idx]
        intersections = int(0)

        # Check distance to each segment and perform ray casting
        for k in range(n_pts):
            p1 = flow.solid_line_transformed[obj_idx, k]
            # Connect to next point (closed loop)
            p2 = flow.solid_line_transformed[obj_idx, (k + 1) % n_pts]

            # Distance calculation (for border)
            d = point_segment_dist(pos, p1, p2)
            if d < min_d:
                min_d = d

            # Ray direction: +X
            x1 = p1[0]
            y1 = p1[1]
            x2 = p2[0]
            y2 = p2[1]

            # Check edge intersection with ray
            # Condition: edge crosses the horizontal line y = pos.y
            if (y1 > pos[1]) != (y2 > pos[1]):
                # Calculate x-coordinate of the intersection
                # x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                intersect_x = x1 + (pos[1] - y1) * (x2 - x1) / (y2 - y1)

                # Check if intersection is to the right of the point
                if pos[0] < intersect_x:
                    intersections = intersections + 1

        # Odd number of intersections means point is inside
        if (intersections % 2) == 1:
            is_inside = 1

    if is_inside == 1 or min_d < thickness:
        flow.u_img[x, y] = 1.0
    else:
        flow.u_img[x, y] = 0.0
