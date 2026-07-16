"""
3D LBM Functions with Multi-World Support

Warp kernels for 3D Lattice Boltzmann Method simulation.
"""
import warp as wp
from .lbm_core_3d import (
    HomeFlow3D, cx_d3q27, cy_d3q27, cz_d3q27, w_d3q27, 
    indexd3q27Inv, cs2, ML_FLUID, ML_WALL, ML_SOLID
)


@wp.kernel
def InitBoundary3D(flows: wp.array(dtype=HomeFlow3D)):
    """Initialize boundary flags for all worlds."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    if (x == 0 or x == flow.nx - 1) or (y == 0 or y == flow.ny - 1) or (z == 0 or z == flow.nz - 1):
        flow.flag[x, y, z] = ML_WALL


@wp.kernel
def InitFlow3D(flows: wp.array(dtype=HomeFlow3D)):
    """Initialize flow field for all worlds."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    
    rho = 1.0
    ux = 0.0
    uy = 0.0
    uz = 0.0
    
    pop = wp.types.vector(length=27, dtype=wp.float32)
    U_sqr = ux * ux + uy * uy + uz * uz
    
    for i in range(27):
        cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
        pop[i] = w_d3q27[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr)
    
    invRho = 1.0 / rho

    pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + 
            pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])
    pixy = ((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - 
            (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26]))
    pixz = ((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - 
            (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26]))
    piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + 
            pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])
    piyz = ((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - 
            (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24]))
    pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + 
            pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])

    cs2_local = pixx
    pixx = pixx * invRho - cs2_local
    pixy = pixy * invRho
    pixz = pixz * invRho
    piyy = piyy * invRho - cs2_local
    piyz = piyz * invRho
    pizz = pizz * invRho - cs2_local

    flow.rho[x, y, z] = rho
    flow.u[x, y, z] = wp.vec3(ux, uy, uz)
    flow.Sxx[x, y, z] = pixx
    flow.Syy[x, y, z] = piyy
    flow.Szz[x, y, z] = pizz
    flow.Sxy[x, y, z] = pixy
    flow.Sxz[x, y, z] = pixz
    flow.Syz[x, y, z] = piyz

    flow.rho_post[x, y, z] = rho
    flow.u_post[x, y, z] = wp.vec3(ux, uy, uz)
    flow.Sxx_post[x, y, z] = pixx
    flow.Syy_post[x, y, z] = piyy
    flow.Szz_post[x, y, z] = pizz
    flow.Sxy_post[x, y, z] = pixy
    flow.Sxz_post[x, y, z] = pixz
    flow.Syz_post[x, y, z] = piyz

    flow.forcex[x, y, z] = 0.0
    flow.forcey[x, y, z] = 0.0
    flow.forcez[x, y, z] = 0.0


@wp.kernel
def ResetSingleWorldFlow3D(flows: wp.array(dtype=HomeFlow3D), reset_mask: wp.array(dtype=wp.int32)):
    """Reset flow field for specific worlds indicated by reset_mask."""
    world_idx, x, y, z = wp.tid()
    
    if reset_mask[world_idx] == 0:
        return
    
    flow = flows[world_idx]
    
    rho = 1.0
    ux = 0.0
    uy = 0.0
    uz = 0.0
    
    pop = wp.types.vector(length=27, dtype=wp.float32)
    U_sqr = ux * ux + uy * uy + uz * uz
    
    for i in range(27):
        cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
        pop[i] = w_d3q27[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr)
    
    invRho = 1.0 / rho

    pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + 
            pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])
    pixy = ((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - 
            (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26]))
    pixz = ((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - 
            (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26]))
    piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + 
            pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])
    piyz = ((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - 
            (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24]))
    pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + 
            pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])

    cs2_local = pixx
    pixx = pixx * invRho - cs2_local
    pixy = pixy * invRho
    pixz = pixz * invRho
    piyy = piyy * invRho - cs2_local
    piyz = piyz * invRho
    pizz = pizz * invRho - cs2_local

    flow.rho[x, y, z] = rho
    flow.rho_post[x, y, z] = rho
    flow.u[x, y, z] = wp.vec3(ux, uy, uz)
    flow.u_post[x, y, z] = wp.vec3(ux, uy, uz)
    flow.Sxx[x, y, z] = pixx
    flow.Sxx_post[x, y, z] = pixx
    flow.Syy[x, y, z] = piyy
    flow.Syy_post[x, y, z] = piyy
    flow.Szz[x, y, z] = pizz
    flow.Szz_post[x, y, z] = pizz
    flow.Sxy[x, y, z] = pixy
    flow.Sxy_post[x, y, z] = pixy
    flow.Sxz[x, y, z] = pixz
    flow.Sxz_post[x, y, z] = pixz
    flow.Syz[x, y, z] = piyz
    flow.Syz_post[x, y, z] = piyz

    flow.forcex[x, y, z] = 0.0
    flow.forcey[x, y, z] = 0.0
    flow.forcez[x, y, z] = 0.0


@wp.kernel
def ResetSingleWorldSolidTransform3D(
    flows: wp.array(dtype=HomeFlow3D), reset_mask: wp.array(dtype=wp.int32)
):
    """
    Reset solid mesh transforms for specific worlds.
    This is needed so that the boundary velocity calculation won't have stale data.
    3D launch: (nworld, n_objects, max_segments_per_object)
    """
    world_idx, solid_id, idx = wp.tid()
    
    # Skip if this world should not be reset
    if reset_mask[world_idx] == 0:
        return
    
    flow = flows[world_idx]
    
    # Reset transform to identity
    if idx == 0:
        flow.mesh_transforms[solid_id] = wp.transform_identity()
        flow.mesh_transforms_last[solid_id] = wp.transform_identity()
        flow.mesh_transforms_initialized[solid_id] = 0


@wp.kernel
def ResetSingleWorldForces3D(
    flows: wp.array(dtype=HomeFlow3D), reset_mask: wp.array(dtype=wp.int32)
):
    """
    Reset solid forces and torques for specific worlds.
    1D launch: nworld
    """
    world_idx = wp.tid()
    
    # Skip if this world should not be reset
    if reset_mask[world_idx] == 0:
        return
    
    flow = flows[world_idx]
    
    for solid_id in range(flow.n_objects):
        flow.solid_force[solid_id] = wp.vec3(0.0, 0.0, 0.0)
        flow.solid_torque[solid_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def mlCalDistribution3D(
    rhoVar: wp.float32,
    ux: wp.float32, uy: wp.float32, uz: wp.float32,
    Sxx: wp.float32, Sxy: wp.float32, Sxz: wp.float32,
    Syy: wp.float32, Syz: wp.float32, Szz: wp.float32,
    i: wp.int32
) -> wp.float32:
    """Calculate equilibrium distribution for D3Q27 (indexes vector constants)."""
    cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
    U_sqr = ux * ux + uy * uy + uz * uz

    # Higher order terms
    Qxx = cx_d3q27[i] * cx_d3q27[i] - 1.0/3.0
    Qyy = cy_d3q27[i] * cy_d3q27[i] - 1.0/3.0
    Qzz = cz_d3q27[i] * cz_d3q27[i] - 1.0/3.0
    Qxy = cx_d3q27[i] * cy_d3q27[i]
    Qxz = cx_d3q27[i] * cz_d3q27[i]
    Qyz = cy_d3q27[i] * cz_d3q27[i]

    f_neq = 1.5 * (Qxx * Sxx + Qyy * Syy + Qzz * Szz + 2.0 * (Qxy * Sxy + Qxz * Sxz + Qyz * Syz))

    f_out = w_d3q27[i] * rhoVar * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr + f_neq)
    return f_out


@wp.func
def mlCalEquilibrium3D_s(
    rhoVar: wp.float32,
    ux: wp.float32, uy: wp.float32, uz: wp.float32,
    Sxx: wp.float32, Sxy: wp.float32, Sxz: wp.float32,
    Syy: wp.float32, Syz: wp.float32, Szz: wp.float32,
    dx: wp.float32, dy: wp.float32, dz: wp.float32, wi: wp.float32
) -> wp.float32:
    """Scalar-argument twin of mlCalDistribution3D: takes the lattice velocity
    (dx,dy,dz) and weight wi directly instead of indexing the vector constants.
    With dx=cx[i], dy=cy[i], dz=cz[i], wi=w[i] the arithmetic is identical, so
    the returned value is bit-identical to mlCalDistribution3D(...,i). This lets
    the caller use a runtime loop + global-memory constants (no register-resident
    vector indexing, which spills)."""
    cu = dx * ux + dy * uy + dz * uz
    U_sqr = ux * ux + uy * uy + uz * uz

    Qxx = dx * dx - 1.0/3.0
    Qyy = dy * dy - 1.0/3.0
    Qzz = dz * dz - 1.0/3.0
    Qxy = dx * dy
    Qxz = dx * dz
    Qyz = dy * dz

    f_neq = 1.5 * (Qxx * Sxx + Qyy * Syy + Qzz * Szz + 2.0 * (Qxy * Sxy + Qxz * Sxz + Qyz * Syz))

    f_out = wi * rhoVar * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr + f_neq)
    return f_out


@wp.func
def get_cutcell_multi_3d(
    mesh_transforms: wp.array(dtype=wp.transform),
    mesh_scale_sizes: wp.array(dtype=wp.vec3),
    mesh_ids: wp.array(dtype=wp.uint64),
    solid_position: wp.array(dtype=wp.vec3),
    solid_bound_radius: wp.array(dtype=wp.float32),
    n_objects: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3
) -> wp.vec3:
    """
    Query all objects for intersection.
    Returns: vec3(cutcell_distance, solid_id, 0) or (-1, -1, 0) if no hit.

    Narrow-band culling: solids whose bounding sphere (center=solid_position,
    radius=solid_bound_radius) does not reach within one lattice link of the
    ray origin are skipped. A cut-cell hit lies within |ray_direction|*cutcell
    <= sqrt(3) < 2 of ray_origin, so a margin of 2.0 is conservative and this
    culling cannot change results.
    """
    cutcell = float(-1.0)
    hit_solid_id = int(-1)

    for solid_id in range(n_objects):
        mesh_id = mesh_ids[solid_id]
        if mesh_id == wp.uint64(0):
            continue

        # Skip solids too far to possibly produce a cut-cell along this link.
        if wp.length(ray_origin - solid_position[solid_id]) > solid_bound_radius[solid_id] + 2.0:
            continue

        transform = mesh_transforms[solid_id]
        scale = mesh_scale_sizes[solid_id]
        
        # Transform ray to local coordinates
        inv_transform = wp.transform_inverse(transform)
        local_origin = wp.transform_point(inv_transform, ray_origin)
        local_dir = wp.transform_vector(inv_transform, ray_direction)
        
        # Scale ray for mesh query
        local_origin_scaled = wp.vec3(
            local_origin[0] / scale[0],
            local_origin[1] / scale[1],
            local_origin[2] / scale[2]
        )
        local_dir_scaled = wp.vec3(
            local_dir[0] / scale[0],
            local_dir[1] / scale[1],
            local_dir[2] / scale[2]
        )
        
        # Query mesh
        query = wp.mesh_query_ray(mesh_id, local_origin_scaled, local_dir_scaled, 2.0)
        
        if query.result:
            t = query.t
            if (cutcell < 0.0 or t < cutcell) and t >= 0.0 and t <= 1.0:
                cutcell = t
                hit_solid_id = solid_id
    
    return wp.vec3(cutcell, float(hit_solid_id), 0.0)


@wp.func
def any_mesh_hit_along_ray_3d(
    mesh_transforms: wp.array(dtype=wp.transform),
    mesh_scale_sizes: wp.array(dtype=wp.vec3),
    mesh_ids: wp.array(dtype=wp.uint64),
    n_objects: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    max_t: float,
) -> int:
    """Return 1 if any mesh is hit along the ray, else 0.

    This is more robust for visualization silhouettes than inside/outside sign
    tests on non-watertight meshes.
    """
    for solid_id in range(n_objects):
        mesh_id = mesh_ids[solid_id]
        if mesh_id == wp.uint64(0):
            continue

        transform = mesh_transforms[solid_id]
        scale = mesh_scale_sizes[solid_id]

        inv_transform = wp.transform_inverse(transform)
        local_origin = wp.transform_point(inv_transform, ray_origin)
        local_dir = wp.transform_vector(inv_transform, ray_direction)

        local_origin_scaled = wp.vec3(
            local_origin[0] / scale[0],
            local_origin[1] / scale[1],
            local_origin[2] / scale[2],
        )
        local_dir_scaled = wp.vec3(
            local_dir[0] / scale[0],
            local_dir[1] / scale[1],
            local_dir[2] / scale[2],
        )

        query = wp.mesh_query_ray(mesh_id, local_origin_scaled, local_dir_scaled, max_t)
        if query.result and query.t >= 0.0 and query.t <= max_t:
            return 1

    return 0


@wp.kernel
def stream_and_collide_3d(flows: wp.array(dtype=HomeFlow3D)):
    """Stream and collide kernel with multi-object support for all worlds."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    
    if flow.flag[x, y, z] == ML_FLUID:
        rhoVar_cur = flow.rho[x, y, z]
        ux_cur = flow.u[x, y, z][0]
        uy_cur = flow.u[x, y, z][1]
        uz_cur = flow.u[x, y, z][2]
        Sxx_cur = flow.Sxx[x, y, z]
        Syy_cur = flow.Syy[x, y, z]
        Szz_cur = flow.Szz[x, y, z]
        Sxy_cur = flow.Sxy[x, y, z]
        Sxz_cur = flow.Sxz[x, y, z]
        Syz_cur = flow.Syz[x, y, z]

        # Narrow-band culling: determine once per cell whether ANY solid is close
        # enough that a cut-cell is possible along some lattice link. Cells outside
        # every solid's (radius + margin) sphere can only ever take the plain
        # streaming branch, so we skip all mesh ray queries for them. This is exact
        # (see get_cutcell_multi_3d): a hit lies within sqrt(3) < 2 of the cell.
        cell_pos = wp.vec3(float(x), float(y), float(z))
        near_solid = int(0)
        for sid in range(flow.n_objects):
            if flow.mesh_ids[sid] == wp.uint64(0):
                continue
            if wp.length(cell_pos - flow.solid_position[sid]) <= flow.solid_bound_radius[sid] + 2.0:
                near_solid = 1

        # Moment accumulators (see Fix #3 note / perf.md). The 27 directions are
        # iterated with a RUNTIME loop over flow.n_dirs and the D3Q27 constants are
        # read from global memory (flow.lat_e / lat_w / lat_inv). This avoids Warp's
        # auto-unroll of `range(27)` — which over-pipelines into ~760 B register
        # spills and runs ~14x slower — and avoids register-materialising the
        # 27-element vector constants. Each streamed population is distributed into
        # named scalar accumulators in ascending-i order (group membership = sign
        # pattern of the lattice velocity), matching the original hand-written moment
        # sums up to float summation associativity (~1e-6, below the sim's existing
        # atomic-order noise). Math, equilibrium and collision are unchanged.
        rho_acc = float(0.0)
        uxp = float(0.0); uxm = float(0.0)
        uyp = float(0.0); uym = float(0.0)
        uzp = float(0.0); uzm = float(0.0)
        pixx = float(0.0); piyy = float(0.0); pizz = float(0.0)
        pixy_p = float(0.0); pixy_m = float(0.0)
        pixz_p = float(0.0); pixz_m = float(0.0)
        piyz_p = float(0.0); piyz_m = float(0.0)

        for i in range(flow.n_dirs):
            e_i = flow.lat_e[i]
            dx = e_i[0]
            dy = e_i[1]
            dz = e_i[2]
            wi = flow.lat_w[i]
            x1 = x - int(dx)
            y1 = y - int(dy)
            z1 = z - int(dz)

            pop_i = float(0.0)
            if x1 >= 0 and x1 < flow.nx and y1 >= 0 and y1 < flow.ny and z1 >= 0 and z1 < flow.nz:
                cutcell = float(-1.0)
                solid_id = int(-1)
                if near_solid == 1:
                    ray_origin = wp.vec3(float(x), float(y), float(z))
                    ray_direction = wp.vec3(-dx, -dy, -dz)

                    # Query all objects for intersection
                    result = get_cutcell_multi_3d(
                        flow.mesh_transforms,
                        flow.mesh_scale_sizes,
                        flow.mesh_ids,
                        flow.solid_position,
                        flow.solid_bound_radius,
                        flow.n_objects,
                        ray_origin,
                        ray_direction
                    )
                    cutcell = result[0]
                    solid_id = int(result[1])

                if cutcell >= 0.0 and cutcell <= 1.0 and solid_id >= 0:
                    # Get transforms for current and last frame
                    transform_current = flow.mesh_transforms[solid_id]
                    transform_last = flow.mesh_transforms_last[solid_id]
                    is_initialized = flow.mesh_transforms_initialized[solid_id]

                    # Calculate hit point
                    hit_point = ray_origin + ray_direction * cutcell

                    # Transform to local coordinates
                    inv_transform = wp.transform_inverse(transform_current)
                    hit_point_local = wp.transform_point(inv_transform, hit_point)

                    # Calculate velocity using frame-to-frame position difference
                    if is_initialized > 0:
                        p_world_current = wp.transform_point(transform_current, hit_point_local)
                        p_world_last = wp.transform_point(transform_last, hit_point_local)
                        u_p = p_world_current - p_world_last
                    else:
                        solid_pos = flow.solid_position[solid_id]
                        solid_linear_v = flow.linear_v[solid_id]
                        solid_angle_v = flow.angle_v[solid_id]
                        dv_fallback = hit_point - solid_pos
                        vel_angle_fallback = wp.vec3(
                            solid_angle_v[1]*dv_fallback[2] - solid_angle_v[2]*dv_fallback[1],
                            solid_angle_v[2]*dv_fallback[0] - solid_angle_v[0]*dv_fallback[2],
                            solid_angle_v[0]*dv_fallback[1] - solid_angle_v[1]*dv_fallback[0]
                        )
                        u_p = solid_linear_v + vel_angle_fallback

                    # Calculate dv for torque calculation
                    solid_pos = flow.solid_position[solid_id]
                    dv = hit_point - solid_pos

                    Sxx_s = u_p[0]*u_p[0] + Sxx_cur - ux_cur*ux_cur
                    Sxy_s = u_p[0]*u_p[1] + Sxy_cur - ux_cur*uy_cur
                    Sxz_s = u_p[0]*u_p[2] + Sxz_cur - ux_cur*uz_cur
                    Syy_s = u_p[1]*u_p[1] + Syy_cur - uy_cur*uy_cur
                    Syz_s = u_p[1]*u_p[2] + Syz_cur - uy_cur*uz_cur
                    Szz_s = u_p[2]*u_p[2] + Szz_cur - uz_cur*uz_cur

                    pop_i = mlCalEquilibrium3D_s(rhoVar_cur, u_p[0], u_p[1], u_p[2],
                                                 Sxx_s, Sxy_s, Sxz_s, Syy_s, Syz_s, Szz_s,
                                                 dx, dy, dz, wi)

                    inv_i = flow.lat_inv[i]
                    e_inv = flow.lat_e[inv_i]
                    cur_f = mlCalEquilibrium3D_s(rhoVar_cur, ux_cur, uy_cur, uz_cur,
                                                 Sxx_cur, Sxy_cur, Sxz_cur, Syy_cur, Syz_cur, Szz_cur,
                                                 e_inv[0], e_inv[1], e_inv[2], flow.lat_w[inv_i])

                    # Calculate boundary force
                    bndForcex = cur_f * (e_inv[0] - u_p[0]) - pop_i * (dx - u_p[0])
                    bndForcey = cur_f * (e_inv[1] - u_p[1]) - pop_i * (dy - u_p[1])
                    bndForcez = cur_f * (e_inv[2] - u_p[2]) - pop_i * (dz - u_p[2])

                    # Calculate boundary torque
                    bndTorquex = dv[1] * bndForcez - dv[2] * bndForcey
                    bndTorquey = dv[2] * bndForcex - dv[0] * bndForcez
                    bndTorquez = dv[0] * bndForcey - dv[1] * bndForcex

                    # Atomic add to the correct solid
                    bndForce = wp.vec3(bndForcex, bndForcey, bndForcez)
                    bndTorque = wp.vec3(bndTorquex, bndTorquey, bndTorquez)
                    wp.atomic_add(flow.solid_force, solid_id, bndForce)
                    wp.atomic_add(flow.solid_torque, solid_id, bndTorque)
                else:
                    rhoVar = flow.rho[x1, y1, z1]
                    ux = flow.u[x1, y1, z1][0]
                    uy = flow.u[x1, y1, z1][1]
                    uz = flow.u[x1, y1, z1][2]
                    Sxx = flow.Sxx[x1, y1, z1]
                    Syy = flow.Syy[x1, y1, z1]
                    Szz = flow.Szz[x1, y1, z1]
                    Sxy = flow.Sxy[x1, y1, z1]
                    Sxz = flow.Sxz[x1, y1, z1]
                    Syz = flow.Syz[x1, y1, z1]
                    pop_i = mlCalEquilibrium3D_s(rhoVar, ux, uy, uz, Sxx, Sxy, Sxz, Syy, Syz, Szz,
                                                 dx, dy, dz, wi)
                    if pop_i < 0.0:
                        pop_i = 0.0

            # Distribute pop_i into moment accumulators in ascending-i order.
            rho_acc = rho_acc + pop_i
            if dx > 0.0:
                uxp = uxp + pop_i
                pixx = pixx + pop_i
            elif dx < 0.0:
                uxm = uxm + pop_i
                pixx = pixx + pop_i
            if dy > 0.0:
                uyp = uyp + pop_i
                piyy = piyy + pop_i
            elif dy < 0.0:
                uym = uym + pop_i
                piyy = piyy + pop_i
            if dz > 0.0:
                uzp = uzp + pop_i
                pizz = pizz + pop_i
            elif dz < 0.0:
                uzm = uzm + pop_i
                pizz = pizz + pop_i
            dxy = dx * dy
            if dxy > 0.0:
                pixy_p = pixy_p + pop_i
            elif dxy < 0.0:
                pixy_m = pixy_m + pop_i
            dxz = dx * dz
            if dxz > 0.0:
                pixz_p = pixz_p + pop_i
            elif dxz < 0.0:
                pixz_m = pixz_m + pop_i
            dyz = dy * dz
            if dyz > 0.0:
                piyz_p = piyz_p + pop_i
            elif dyz < 0.0:
                piyz_m = piyz_m + pop_i

        Fx = flow.forcex[x, y, z]
        Fy = flow.forcey[x, y, z]
        Fz = flow.forcez[x, y, z]

        rhoVar = rho_acc
        invRho = 1.0 / rhoVar
        ux = (uxp - uxm + 0.5 * Fx) * invRho
        uy = (uyp - uym + 0.5 * Fy) * invRho
        uz = (uzp - uzm + 0.5 * Fz) * invRho

        pixy = pixy_p - pixy_m
        pixz = pixz_p - pixz_m
        piyz = piyz_p - piyz_m


        Omega = 1.0 / (flow.vis_shear * 3.0 + 0.5)

        # 3D collision
        pixx_part = (2.0 * pixx - piyy - pizz) / 3.0
        piyy_part = (2.0 * piyy - pixx - pizz) / 3.0
        pizz_part = (2.0 * pizz - pixx - piyy) / 3.0
        RU2 = rhoVar * ux * ux
        RV2 = rhoVar * uy * uy
        RW2 = rhoVar * uz * uz
        RUVW2 = (RU2 + RV2 + RW2) / 3.0
        
        pixx = rhoVar / 3.0 + pixx_part * (1.0 - Omega) + RUVW2 + (2.0 * RU2 * Omega) / 3.0 - (RV2 * Omega) / 3.0 - (RW2 * Omega) / 3.0 + Fx * ux
        piyy = rhoVar / 3.0 + piyy_part * (1.0 - Omega) + RUVW2 - (RU2 * Omega) / 3.0 + (2.0 * RV2 * Omega) / 3.0 - (RW2 * Omega) / 3.0 + Fy * uy
        pizz = rhoVar / 3.0 + pizz_part * (1.0 - Omega) + RUVW2 - (RU2 * Omega) / 3.0 - (RV2 * Omega) / 3.0 + (2.0 * RW2 * Omega) / 3.0 + Fz * uz
        pixy = pixy - pixy * Omega + ux * uy * rhoVar * Omega + (Fy * ux + Fx * uy) / 2.0
        pixz = pixz - pixz * Omega + ux * uz * rhoVar * Omega + (Fz * ux + Fx * uz) / 2.0
        piyz = piyz - piyz * Omega + uy * uz * rhoVar * Omega + (Fz * uy + Fy * uz) / 2.0
        
        flow.rho_post[x, y, z] = rhoVar
        flow.u_post[x, y, z] = wp.vec3(ux + Fx * invRho / 2.0, uy + Fy * invRho / 2.0, uz + Fz * invRho / 2.0)
        flow.Sxx_post[x, y, z] = pixx * invRho - cs2
        flow.Sxy_post[x, y, z] = pixy * invRho
        flow.Sxz_post[x, y, z] = pixz * invRho
        flow.Syy_post[x, y, z] = piyy * invRho - cs2
        flow.Syz_post[x, y, z] = piyz * invRho
        flow.Szz_post[x, y, z] = pizz * invRho - cs2


@wp.kernel
def apply_bc_3d(flows: wp.array(dtype=HomeFlow3D)):
    """Apply boundary conditions for all worlds."""
    world_idx, x, y, z = wp.tid()
    flow = flows[world_idx]
    
    if x == 0:
        apply_bc_core_3d(flow, 1, 0, 0, y, z, 1, y, z)
    elif x == flow.nx - 1:
        apply_bc_core_3d(flow, 1, 1, flow.nx - 1, y, z, flow.nx - 2, y, z)
    if y == flow.ny - 1:
        apply_bc_core_3d(flow, 1, 2, x, flow.ny - 1, z, x, flow.ny - 2, z)
    elif y == 0:
        apply_bc_core_3d(flow, 1, 3, x, 0, z, x, 1, z)
    if z == flow.nz - 1:
        apply_bc_core_3d(flow, 1, 4, x, y, flow.nz - 1, x, y, flow.nz - 2)
    elif z == 0:
        apply_bc_core_3d(flow, 1, 5, x, y, 0, x, y, 1)


@wp.func
def apply_bc_core_3d(
    flow: HomeFlow3D, outer: wp.int32, dr: wp.int32,
    ibc: wp.int32, jbc: wp.int32, kbc: wp.int32,
    inb: wp.int32, jnb: wp.int32, knb: wp.int32
):
    if outer == 1:
        if flow.bc_type[dr] == 0:
            flow.u_post[ibc, jbc, kbc] = flow.bc_value[dr]
        elif flow.bc_type[dr] == 1:
            flow.u_post[ibc, jbc, kbc] = flow.u_post[inb, jnb, knb]
    flow.rho_post[ibc, jbc, kbc] = flow.rho_post[inb, jnb, knb]
    
    pixx_cur, pixy_cur, pixz_cur, piyy_cur, piyz_cur, pizz_cur = f_eq_home_3d(flow, ibc, jbc, kbc)
    pixx_back, pixy_back, pixz_back, piyy_back, piyz_back, pizz_back = f_eq_home_3d(flow, inb, jnb, knb)
    
    flow.Sxx_post[ibc, jbc, kbc] = pixx_cur - pixx_back + flow.Sxx_post[inb, jnb, knb]
    flow.Sxy_post[ibc, jbc, kbc] = pixy_cur - pixy_back + flow.Sxy_post[inb, jnb, knb]
    flow.Sxz_post[ibc, jbc, kbc] = pixz_cur - pixz_back + flow.Sxz_post[inb, jnb, knb]
    flow.Syy_post[ibc, jbc, kbc] = piyy_cur - piyy_back + flow.Syy_post[inb, jnb, knb]
    flow.Syz_post[ibc, jbc, kbc] = piyz_cur - piyz_back + flow.Syz_post[inb, jnb, knb]
    flow.Szz_post[ibc, jbc, kbc] = pizz_cur - pizz_back + flow.Szz_post[inb, jnb, knb]


@wp.func
def f_eq_home_3d(flow: HomeFlow3D, x1: wp.int32, y1: wp.int32, z1: wp.int32):
    pop = wp.types.vector(length=27, dtype=wp.float32)
    rho = flow.rho_post[x1, y1, z1]
    ux = flow.u_post[x1, y1, z1].x
    uy = flow.u_post[x1, y1, z1].y
    uz = flow.u_post[x1, y1, z1].z
    U_sqr = ux * ux + uy * uy + uz * uz
    
    for i in range(27):
        cu = cx_d3q27[i] * ux + cy_d3q27[i] * uy + cz_d3q27[i] * uz
        pop[i] = w_d3q27[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * U_sqr)
    
    invRho = 1.0 / rho

    pixx = (pop[1] + pop[2] + pop[7] + pop[8] + pop[9] + pop[10] + pop[13] + pop[14] + 
            pop[15] + pop[16] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])
    pixy = ((pop[7] + pop[8] + pop[19] + pop[20] + pop[21] + pop[22]) - 
            (pop[13] + pop[14] + pop[23] + pop[24] + pop[25] + pop[26]))
    pixz = ((pop[9] + pop[10] + pop[19] + pop[20] + pop[23] + pop[24]) - 
            (pop[15] + pop[16] + pop[21] + pop[22] + pop[25] + pop[26]))
    piyy = (pop[3] + pop[4] + pop[7] + pop[8] + pop[11] + pop[12] + pop[13] + pop[14] + 
            pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])
    piyz = ((pop[11] + pop[12] + pop[19] + pop[20] + pop[25] + pop[26]) - 
            (pop[17] + pop[18] + pop[21] + pop[22] + pop[23] + pop[24]))
    pizz = (pop[5] + pop[6] + pop[9] + pop[10] + pop[11] + pop[12] + pop[15] + pop[16] + 
            pop[17] + pop[18] + pop[19] + pop[20] + pop[21] + pop[22] + pop[23] + pop[24] + pop[25] + pop[26])

    pixx = pixx * invRho - cs2
    pixy = pixy * invRho
    pixz = pixz * invRho
    piyy = piyy * invRho - cs2
    piyz = piyz * invRho
    pizz = pizz * invRho - cs2

    return pixx, pixy, pixz, piyy, piyz, pizz


@wp.kernel
def Swap_Mom_3D(flows: wp.array(dtype=HomeFlow3D)):
    """Swap pre and post arrays for all worlds."""
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

    tmp_Szz = f.Szz
    f.Szz = f.Szz_post
    f.Szz_post = tmp_Szz

    tmp_Sxy = f.Sxy
    f.Sxy = f.Sxy_post
    f.Sxy_post = tmp_Sxy

    tmp_Sxz = f.Sxz
    f.Sxz = f.Sxz_post
    f.Sxz_post = tmp_Sxz

    tmp_Syz = f.Syz
    f.Syz = f.Syz_post
    f.Syz_post = tmp_Syz

    flows[tid] = f


def init_force_3d(flow: HomeFlow3D):
    """Clear forces and torques for all objects in a flow."""
    flow.solid_force.zero_()
    flow.solid_torque.zero_()


@wp.kernel
def init_force_3d_batch(flows: wp.array(dtype=HomeFlow3D)):
    """
    Clear forces and torques for all objects in all worlds.
    1D launch: (nworld,)
    """
    world_idx = wp.tid()
    flow = flows[world_idx]
    
    for solid_id in range(flow.n_objects):
        flow.solid_force[solid_id] = wp.vec3(0.0, 0.0, 0.0)
        flow.solid_torque[solid_id] = wp.vec3(0.0, 0.0, 0.0)


# ============== Rendering Kernels for 3D ==============


@wp.kernel
def get_u_img_xy_3d(flow: HomeFlow3D, z_slice: int):
    """
    Get velocity magnitude on XY plane at z=z_slice.
    Result saved to flow.u_img_xy (shape: nx, ny)
    """
    x, y = wp.tid()
    u_vec = flow.u[x, y, z_slice]
    flow.u_img_xy[x, y] = wp.sqrt(u_vec[0] ** 2.0 + u_vec[1] ** 2.0 + u_vec[2] ** 2.0)


@wp.kernel
def get_u_img_yz_3d(flow: HomeFlow3D, x_slice: int):
    """
    Get velocity magnitude on YZ plane at x=x_slice.
    Result saved to flow.u_img_xz (reusing buffer, shape: nx->y, nz->z)
    Note: We store (y, z) in u_img_xz which has shape (nx, nz), so we use (y, z) indexing
    """
    y, z = wp.tid()
    u_vec = flow.u[x_slice, y, z]
    # Store in u_img_xz buffer - reinterpreted as (ny, nz) for YZ plane
    # But u_img_xz has shape (nx, nz), so we need to be careful
    # Actually we'll create a separate buffer or just use u_img_xy for one and u_img_xz for another
    flow.u_img_xz[y, z] = wp.sqrt(u_vec[0] ** 2.0 + u_vec[1] ** 2.0 + u_vec[2] ** 2.0)


@wp.kernel
def get_vorticity_xy_3d(flow: HomeFlow3D, z_slice: int):
    """
    Compute vorticity (z-component) on XY plane at z=z_slice.
    vorticity_z = ∂v/∂x - ∂u/∂y
    Result saved to flow.u_img_xy
    """
    x, y = wp.tid()
    z = z_slice
    
    dvdx = 0.0
    dudy = 0.0
    
    # Compute ∂v/∂x (v is u[1])
    if x == 0:
        dvdx = (flow.u[x + 1, y, z][1] - flow.u[x, y, z][1]) / flow.grid_length
    elif x == flow.nx - 1:
        dvdx = (flow.u[x, y, z][1] - flow.u[x - 1, y, z][1]) / flow.grid_length
    else:
        dvdx = (flow.u[x + 1, y, z][1] - flow.u[x - 1, y, z][1]) / (2.0 * flow.grid_length)
    
    # Compute ∂u/∂y (u is u[0])
    if y == 0:
        dudy = (flow.u[x, y + 1, z][0] - flow.u[x, y, z][0]) / flow.grid_length
    elif y == flow.ny - 1:
        dudy = (flow.u[x, y, z][0] - flow.u[x, y - 1, z][0]) / flow.grid_length
    else:
        dudy = (flow.u[x, y + 1, z][0] - flow.u[x, y - 1, z][0]) / (2.0 * flow.grid_length)
    
    flow.u_img_xy[x, y] = dvdx - dudy


@wp.kernel
def get_vorticity_yz_3d(flow: HomeFlow3D, x_slice: int):
    """
    Compute vorticity (x-component) on YZ plane at x=x_slice.
    vorticity_x = ∂w/∂y - ∂v/∂z
    Result saved to flow.u_img_xz (reused as ny x nz buffer)
    """
    y, z = wp.tid()
    x = x_slice
    
    dwdy = 0.0
    dvdz = 0.0
    
    # Compute ∂w/∂y (w is u[2])
    if y == 0:
        dwdy = (flow.u[x, y + 1, z][2] - flow.u[x, y, z][2]) / flow.grid_length
    elif y == flow.ny - 1:
        dwdy = (flow.u[x, y, z][2] - flow.u[x, y - 1, z][2]) / flow.grid_length
    else:
        dwdy = (flow.u[x, y + 1, z][2] - flow.u[x, y - 1, z][2]) / (2.0 * flow.grid_length)
    
    # Compute ∂v/∂z (v is u[1])
    if z == 0:
        dvdz = (flow.u[x, y, z + 1][1] - flow.u[x, y, z][1]) / flow.grid_length
    elif z == flow.nz - 1:
        dvdz = (flow.u[x, y, z][1] - flow.u[x, y, z - 1][1]) / flow.grid_length
    else:
        dvdz = (flow.u[x, y, z + 1][1] - flow.u[x, y, z - 1][1]) / (2.0 * flow.grid_length)
    
    flow.u_img_xz[y, z] = dwdy - dvdz


@wp.kernel
def get_vorticity_xy_with_solid_3d(flow: HomeFlow3D, z_slice: int, thickness: float):
    """
    Compute vorticity on XY plane with solid overlay.
    Solid regions are marked with value >= 1000.0
    """
    x, y = wp.tid()
    z = z_slice
    
    # Check if inside solid using flag
    if flow.flag[x, y, z] == ML_SOLID:
        flow.u_img_xy[x, y] = 1000.0
        return
    
    # Check mesh distance for each object
    pos = wp.vec3(float(x), float(y), float(z))
    for obj_idx in range(flow.n_objects):
        mesh_id = flow.mesh_ids[obj_idx]
        if mesh_id == wp.uint64(0):
            continue
        
        transform = flow.mesh_transforms[obj_idx]
        scale = flow.mesh_scale_sizes[obj_idx]
        
        # Transform point to mesh local space
        local_pos = wp.transform_point(wp.transform_inverse(transform), pos)
        local_pos = wp.vec3(local_pos[0] / scale[0], local_pos[1] / scale[1], local_pos[2] / scale[2])
        
        # Query distance
        face_idx = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        max_dist = 100.0
        res = wp.mesh_query_point_sign_normal(mesh_id, local_pos, max_dist, sign, face_idx, face_u, face_v)
        
        if res:
            if sign < 0.0:
                # Inside mesh
                flow.u_img_xy[x, y] = 1000.0
                return
            else:
                # Outside mesh, check if within thickness
                closest = wp.mesh_eval_position(mesh_id, face_idx, face_u, face_v)
                diff = local_pos - closest
                dist = wp.length(diff)
                if dist < thickness:
                    flow.u_img_xy[x, y] = 1000.0
                    return
    
    # Compute vorticity
    dvdx = 0.0
    dudy = 0.0
    
    if x == 0:
        dvdx = (flow.u[x + 1, y, z][1] - flow.u[x, y, z][1]) / flow.grid_length
    elif x == flow.nx - 1:
        dvdx = (flow.u[x, y, z][1] - flow.u[x - 1, y, z][1]) / flow.grid_length
    else:
        dvdx = (flow.u[x + 1, y, z][1] - flow.u[x - 1, y, z][1]) / (2.0 * flow.grid_length)
    
    if y == 0:
        dudy = (flow.u[x, y + 1, z][0] - flow.u[x, y, z][0]) / flow.grid_length
    elif y == flow.ny - 1:
        dudy = (flow.u[x, y, z][0] - flow.u[x, y - 1, z][0]) / flow.grid_length
    else:
        dudy = (flow.u[x, y + 1, z][0] - flow.u[x, y - 1, z][0]) / (2.0 * flow.grid_length)
    
    flow.u_img_xy[x, y] = dvdx - dudy


@wp.kernel
def get_vorticity_yz_with_solid_3d(flow: HomeFlow3D, x_slice: int, thickness: float):
    """
    Compute vorticity on YZ plane with solid overlay.
    Solid regions are marked with value >= 1000.0
    """
    y, z = wp.tid()
    x = x_slice
    
    # Check if inside solid using flag
    if flow.flag[x, y, z] == ML_SOLID:
        flow.u_img_xz[y, z] = 1000.0
        return
    
    # Check mesh distance for each object
    pos = wp.vec3(float(x), float(y), float(z))
    for obj_idx in range(flow.n_objects):
        mesh_id = flow.mesh_ids[obj_idx]
        if mesh_id == wp.uint64(0):
            continue
        
        transform = flow.mesh_transforms[obj_idx]
        scale = flow.mesh_scale_sizes[obj_idx]
        
        # Transform point to mesh local space
        local_pos = wp.transform_point(wp.transform_inverse(transform), pos)
        local_pos = wp.vec3(local_pos[0] / scale[0], local_pos[1] / scale[1], local_pos[2] / scale[2])
        
        # Query distance
        face_idx = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        max_dist = 100.0
        res = wp.mesh_query_point_sign_normal(mesh_id, local_pos, max_dist, sign, face_idx, face_u, face_v)
        
        if res:
            if sign < 0.0:
                flow.u_img_xz[y, z] = 1000.0
                return
            else:
                # Outside mesh, check if within thickness
                closest = wp.mesh_eval_position(mesh_id, face_idx, face_u, face_v)
                diff = local_pos - closest
                dist = wp.length(diff)
                if dist < thickness:
                    flow.u_img_xz[y, z] = 1000.0
                    return
    
    # Compute vorticity
    dwdy = 0.0
    dvdz = 0.0
    
    if y == 0:
        dwdy = (flow.u[x, y + 1, z][2] - flow.u[x, y, z][2]) / flow.grid_length
    elif y == flow.ny - 1:
        dwdy = (flow.u[x, y, z][2] - flow.u[x, y - 1, z][2]) / flow.grid_length
    else:
        dwdy = (flow.u[x, y + 1, z][2] - flow.u[x, y - 1, z][2]) / (2.0 * flow.grid_length)
    
    if z == 0:
        dvdz = (flow.u[x, y, z + 1][1] - flow.u[x, y, z][1]) / flow.grid_length
    elif z == flow.nz - 1:
        dvdz = (flow.u[x, y, z][1] - flow.u[x, y, z - 1][1]) / flow.grid_length
    else:
        dvdz = (flow.u[x, y, z + 1][1] - flow.u[x, y, z - 1][1]) / (2.0 * flow.grid_length)
    
    flow.u_img_xz[y, z] = dwdy - dvdz


# ============== 3D Projection Rendering Kernels ==============
# These kernels project the entire 3D flow field onto a 2D plane
# by looking from a specific direction (top-down, side, front)
# Upper/front layers occlude lower/back layers


@wp.kernel
def get_u_projection_topdown_3d(flow: HomeFlow3D):
    """
    Project velocity magnitude from top-down view (looking along -Z axis).
    Iterates from top (high z) to bottom (low z), first non-zero velocity wins.
    For visualization: shows the "surface" velocity field as seen from above.
    Result saved to flow.u_img_xy (shape: nx, ny)
    """
    x, y = wp.tid()
    
    # Start from top (high z), scan downward
    result = float(0.0)
    found = int(0)  # Use int instead of bool for Warp compatibility
    
    # Robust silhouette ray from just above the domain downward.
    if any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(x), float(y), float(flow.nz) + 1.0),
        wp.vec3(0.0, 0.0, -1.0),
        float(flow.nz) + 2.0,
    ) == 1:
        flow.u_img_xy[x, y] = 1000.0
        return

    for z in range(flow.nz - 1, -1, -1):
        if found == 0:
            # Skip wall/boundary cells
            if flow.flag[x, y, z] == ML_WALL:
                continue
            
            u_vec = flow.u[x, y, z]
            u_mag = wp.sqrt(u_vec[0] ** 2.0 + u_vec[1] ** 2.0 + u_vec[2] ** 2.0)
            
            if u_mag > 1e-8:
                result = u_mag
                found = 1
    
    flow.u_img_xy[x, y] = result


@wp.kernel
def get_u_projection_max_topdown_3d(flow: HomeFlow3D):
    """
    Maximum intensity projection from top-down view (looking along -Z axis).
    Takes the maximum velocity magnitude along each vertical column.
    Better for visualizing the overall flow intensity.
    Result saved to flow.u_img_xy (shape: nx, ny)
    """
    x, y = wp.tid()
    
    max_u = float(0.0)
    has_solid = int(0)  # Use int instead of bool
    
    for z in range(flow.nz):
        if flow.flag[x, y, z] == ML_WALL:
            continue
        
        u_vec = flow.u[x, y, z]
        u_mag = wp.sqrt(u_vec[0] ** 2.0 + u_vec[1] ** 2.0 + u_vec[2] ** 2.0)
        
        if u_mag > max_u:
            max_u = u_mag

    has_solid = any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(x), float(y), float(flow.nz) + 1.0),
        wp.vec3(0.0, 0.0, -1.0),
        float(flow.nz) + 2.0,
    )
    
    if has_solid == 1:
        flow.u_img_xy[x, y] = 1000.0
    else:
        flow.u_img_xy[x, y] = max_u


@wp.kernel
def get_vorticity_projection_topdown_3d(flow: HomeFlow3D):
    """
    Maximum intensity projection of vorticity magnitude from top-down view.
    Projects the z-component vorticity (curl_z = dv/dx - du/dy) along Z axis.
    Result saved to flow.u_img_xy (shape: nx, ny)
    """
    x, y = wp.tid()
    
    max_vort = float(0.0)
    has_solid = int(0)  # Use int instead of bool
    
    for z in range(1, flow.nz - 1):  # Skip boundaries
        if flow.flag[x, y, z] == ML_WALL:
            continue
        
        # Compute vorticity z-component
        dvdx = 0.0
        dudy = 0.0
        
        if x > 0 and x < flow.nx - 1:
            dvdx = (flow.u[x + 1, y, z][1] - flow.u[x - 1, y, z][1]) / (2.0 * flow.grid_length)
        elif x == 0:
            dvdx = (flow.u[x + 1, y, z][1] - flow.u[x, y, z][1]) / flow.grid_length
        else:
            dvdx = (flow.u[x, y, z][1] - flow.u[x - 1, y, z][1]) / flow.grid_length
        
        if y > 0 and y < flow.ny - 1:
            dudy = (flow.u[x, y + 1, z][0] - flow.u[x, y - 1, z][0]) / (2.0 * flow.grid_length)
        elif y == 0:
            dudy = (flow.u[x, y + 1, z][0] - flow.u[x, y, z][0]) / flow.grid_length
        else:
            dudy = (flow.u[x, y, z][0] - flow.u[x, y - 1, z][0]) / flow.grid_length
        
        vort_z = dvdx - dudy
        vort_abs = wp.abs(vort_z)
        
        if vort_abs > wp.abs(max_vort):
            max_vort = vort_z  # Keep sign for visualization

    has_solid = any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(x), float(y), float(flow.nz) + 1.0),
        wp.vec3(0.0, 0.0, -1.0),
        float(flow.nz) + 2.0,
    )
    
    if has_solid == 1:
        flow.u_img_xy[x, y] = 1000.0
    else:
        flow.u_img_xy[x, y] = max_vort


@wp.kernel
def get_u_projection_side_3d(flow: HomeFlow3D):
    """
    Maximum intensity projection from side view (looking along -X axis, from right).
    Projects onto YZ plane.
    Result saved to flow.u_img_xz (shape: ny, nz)
    """
    y, z = wp.tid()
    
    max_u = float(0.0)
    has_solid = int(0)  # Use int instead of bool
    
    for x in range(flow.nx - 1, -1, -1):  # Right to left
        if flow.flag[x, y, z] == ML_WALL:
            continue
        
        u_vec = flow.u[x, y, z]
        u_mag = wp.sqrt(u_vec[0] ** 2.0 + u_vec[1] ** 2.0 + u_vec[2] ** 2.0)
        
        if u_mag > max_u:
            max_u = u_mag

    has_solid = any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(flow.nx) + 1.0, float(y), float(z)),
        wp.vec3(-1.0, 0.0, 0.0),
        float(flow.nx) + 2.0,
    )
    
    if has_solid == 1:
        flow.u_img_xz[y, z] = 1000.0
    else:
        flow.u_img_xz[y, z] = max_u


@wp.kernel
def get_u_projection_front_3d(flow: HomeFlow3D):
    """
    Maximum intensity projection from front view (looking along -Y axis).
    Projects onto XZ plane.
    Result saved to flow.u_img_xz_front (shape: nx, nz)
    """
    x, z = wp.tid()
    
    max_u = float(0.0)
    has_solid = int(0)  # Use int instead of bool
    
    for y in range(flow.ny - 1, -1, -1):  # Front to back
        if flow.flag[x, y, z] == ML_WALL:
            continue
        
        u_vec = flow.u[x, y, z]
        u_mag = wp.sqrt(u_vec[0] ** 2.0 + u_vec[1] ** 2.0 + u_vec[2] ** 2.0)
        
        if u_mag > max_u:
            max_u = u_mag

    has_solid = any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(x), float(flow.ny) + 1.0, float(z)),
        wp.vec3(0.0, -1.0, 0.0),
        float(flow.ny) + 2.0,
    )
    
    if has_solid == 1:
        flow.u_img_xz_front[x, z] = 1000.0
    else:
        flow.u_img_xz_front[x, z] = max_u


@wp.kernel
def get_vorticity_projection_side_3d(flow: HomeFlow3D):
    """
    Maximum intensity projection of vorticity from side view (looking along -X axis).
    Projects the x-component vorticity (curl_x = dw/dy - dv/dz) along X axis.
    Result saved to flow.u_img_xz (shape: ny, nz)
    """
    y, z = wp.tid()
    
    max_vort = float(0.0)
    has_solid = int(0)  # Use int instead of bool
    
    for x in range(1, flow.nx - 1):  # Skip boundaries
        if flow.flag[x, y, z] == ML_WALL:
            continue
        
        # Compute vorticity x-component
        dwdy = 0.0
        dvdz = 0.0
        
        if y > 0 and y < flow.ny - 1:
            dwdy = (flow.u[x, y + 1, z][2] - flow.u[x, y - 1, z][2]) / (2.0 * flow.grid_length)
        elif y == 0:
            dwdy = (flow.u[x, y + 1, z][2] - flow.u[x, y, z][2]) / flow.grid_length
        else:
            dwdy = (flow.u[x, y, z][2] - flow.u[x, y - 1, z][2]) / flow.grid_length
        
        if z > 0 and z < flow.nz - 1:
            dvdz = (flow.u[x, y, z + 1][1] - flow.u[x, y, z - 1][1]) / (2.0 * flow.grid_length)
        elif z == 0:
            dvdz = (flow.u[x, y, z + 1][1] - flow.u[x, y, z][1]) / flow.grid_length
        else:
            dvdz = (flow.u[x, y, z][1] - flow.u[x, y, z - 1][1]) / flow.grid_length
        
        vort_x = dwdy - dvdz
        vort_abs = wp.abs(vort_x)
        
        if vort_abs > wp.abs(max_vort):
            max_vort = vort_x

    has_solid = any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(flow.nx) + 1.0, float(y), float(z)),
        wp.vec3(-1.0, 0.0, 0.0),
        float(flow.nx) + 2.0,
    )
    
    if has_solid == 1:
        flow.u_img_xz[y, z] = 1000.0
    else:
        flow.u_img_xz[y, z] = max_vort


@wp.kernel
def get_vorticity_projection_front_3d(flow: HomeFlow3D):
    """
    Maximum intensity projection of vorticity from front view (looking along -Y axis).
    Projects the y-component vorticity (curl_y = du/dz - dw/dx) along Y axis.
    Result saved to flow.u_img_xz_front (shape: nx, nz)
    """
    x, z = wp.tid()
    
    max_vort = float(0.0)
    has_solid = int(0)  # Use int instead of bool
    
    for y in range(1, flow.ny - 1):  # Skip boundaries
        if flow.flag[x, y, z] == ML_WALL:
            continue
        
        # Compute vorticity y-component
        dudz = 0.0
        dwdx = 0.0
        
        if z > 0 and z < flow.nz - 1:
            dudz = (flow.u[x, y, z + 1][0] - flow.u[x, y, z - 1][0]) / (2.0 * flow.grid_length)
        elif z == 0:
            dudz = (flow.u[x, y, z + 1][0] - flow.u[x, y, z][0]) / flow.grid_length
        else:
            dudz = (flow.u[x, y, z][0] - flow.u[x, y, z - 1][0]) / flow.grid_length
        
        if x > 0 and x < flow.nx - 1:
            dwdx = (flow.u[x + 1, y, z][2] - flow.u[x - 1, y, z][2]) / (2.0 * flow.grid_length)
        elif x == 0:
            dwdx = (flow.u[x + 1, y, z][2] - flow.u[x, y, z][2]) / flow.grid_length
        else:
            dwdx = (flow.u[x, y, z][2] - flow.u[x - 1, y, z][2]) / flow.grid_length
        
        vort_y = dudz - dwdx
        vort_abs = wp.abs(vort_y)
        
        if vort_abs > wp.abs(max_vort):
            max_vort = vort_y

    has_solid = any_mesh_hit_along_ray_3d(
        flow.mesh_transforms,
        flow.mesh_scale_sizes,
        flow.mesh_ids,
        flow.n_objects,
        wp.vec3(float(x), float(flow.ny) + 1.0, float(z)),
        wp.vec3(0.0, -1.0, 0.0),
        float(flow.ny) + 2.0,
    )
    
    if has_solid == 1:
        flow.u_img_xz_front[x, z] = 1000.0
    else:
        flow.u_img_xz_front[x, z] = max_vort
