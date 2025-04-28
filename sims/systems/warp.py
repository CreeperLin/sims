def init_fn(
    urdf_path=None,
    fix_base_link=False,
    disable_gravity=False,
    dt=1.0 / 50.0,
    Kp=None,
    Kd=None,
    torque_limits=None,
    headless=False,
    default_dof_pos=None,
    default_root_states=None,
    ctrl_joints_pos='*',
    ctrl_joints_vel=None,
    ctrl_joints_tau=None,
    clip_q_ctrl=False,
    self_collisions=False,
    decimation=1,
    compute_torque=False,
    sim_device='cpu',
    num_envs=1,
    seed=1,
    verbose=False,
    stage_path="example.usd",
    use_cuda_graph=True,
    # use_cuda_graph=False,
):
    import torch
    import numpy as np
    from sims.utils import get_ctrl_inds, dict2list, update_obj, set_seed, list2slice
    from sims.torch_utils import get_euler_xyz

    dtype = torch.float
    np_dtype = np.float32
    device = sim_device
    sim_dt = dt / decimation

    import warp as wp
    # import warp.examples
    import warp.sim
    import warp.sim.render
    from warp.torch import to_torch, from_torch

    if seed is not None:
        set_seed(seed)

    init_pos = np.array([0, 1, 0])
    init_rot = np.array([0.0, 0.0, 0.0, 1.0])
    if default_root_states is not None:
        init_pos[0] = default_root_states[0]
        init_pos[1] = default_root_states[2]
        init_pos[2] = default_root_states[1]
        init_rot[:] = default_root_states[3:7]

    articulation_builder = wp.sim.ModelBuilder()
    floating = not fix_base_link
    xform = wp.transform(init_pos, init_rot)
    asset_kwds = dict(
        density=1000,
        armature=0.01,
        stiffness=200,
        damping=1,
        contact_ke=1.0e4,
        contact_kd=1.0e2,
        contact_kf=1.0e2,
        contact_mu=1.0,
        limit_ke=1.0e4,
        limit_kd=1.0e1,
        collapse_fixed_joints=True,
    )
    wp.sim.parse_urdf(
        urdf_path,
        articulation_builder,
        xform=xform,
        floating=floating,
        **asset_kwds,
    )
    dof_names = articulation_builder.joint_name
    num_dofs = articulation_builder.joint_count
    print(num_dofs)
    print(dof_names)
    if not fix_base_link:
        num_dofs = num_dofs - 1
        dof_names = dof_names[1:]
    builder = wp.sim.ModelBuilder()
    # fps = 100
    # frame_dt = 1.0 / fps
    init_joint_q = np.zeros(7 + num_dofs)
    init_joint_qd = np.zeros(6 + num_dofs)
    init_joint_q[:3] = init_pos
    init_joint_q[3:7] = init_rot

    if compute_torque:
        mode = wp.sim.JOINT_MODE_FORCE
    else:
        mode = wp.sim.JOINT_MODE_TARGET_POSITION
    for i in range(num_envs):
        xform = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())
        builder.add_builder(articulation_builder, xform=xform)
        builder.joint_q = init_joint_q.tolist()
        builder.joint_axis_mode = [mode] * len(builder.joint_axis_mode)
        builder.joint_act = [0] * len(builder.joint_act)
    np.set_printoptions(suppress=True)
    # finalize model
    model = builder.finalize()
    model.joint_q.assign(init_joint_q)
    model.joint_qd.assign(init_joint_qd)
    model.ground = True
    print('model.gravity', model.gravity)
    if disable_gravity:
        model.gravity[:] = 0.0
    model.joint_attach_ke = 16000.0
    model.joint_attach_kd = 200.0
    # use_tile_gemm = False
    control = model.control()
    print('control.joint_act', len(control.joint_act))
    # integrator = wp.sim.XPBDIntegrator()
    # integrator = wp.sim.SemiImplicitIntegrator()
    integrator = wp.sim.FeatherstoneIntegrator(model)
    if not headless and stage_path:
        renderer = wp.sim.render.SimRenderer(model, stage_path)
    else:
        renderer = None
    state_0 = model.state()
    state_1 = model.state()
    wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state_0)
    cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
    use_cuda_graph = use_cuda_graph and cuda_graph

    def simulate():
        nonlocal state_0, state_1
        state_0.clear_forces()
        wp.sim.collide(model, state_0)
        # integrator.simulate(model, state_0, state_1, sim_dt, control)
        integrator.simulate(model, state_0, state_1, sim_dt)
        state_0, state_1 = state_1, state_0

    def sim_step():
        # with wp.ScopedTimer("step"):
        if use_cuda_graph:
            wp.capture_launch(graph)
        else:
            simulate()

    if use_cuda_graph:
        with wp.ScopedCapture() as capture:
            simulate()
        graph = capture.graph
    else:
        graph = None

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    if isinstance(Kp, dict):
        kp_inds, Kp = dict2list(Kp, dof_names, Kp.pop('_default', None))
    if isinstance(Kd, dict):
        kd_inds, Kd = dict2list(Kd, dof_names, Kd.pop('_default', None))
    if isinstance(torque_limits, dict):
        tl_inds, torque_limits = dict2list(torque_limits, dof_names, torque_limits.pop('_default', None))
    kps = torch.zeros(num_dofs, dtype=dtype, device=device)
    kds = torch.zeros(num_dofs, dtype=dtype, device=device)
    dof_torque_limits = torch.zeros(num_dofs, dtype=dtype, device=device)
    if Kp is not None:
        kps[kp_inds] = torch.tensor(Kp, dtype=dtype, device=device)
        print('kps', kps.tolist())
    if Kd is not None:
        kds[kd_inds] = torch.tensor(Kd, dtype=dtype, device=device)
        print('kds', kds.tolist())
    if dof_torque_limits is not None:
        dof_torque_limits[tl_inds] = torch.tensor(torque_limits, dtype=dtype, device=device)
        print('dof_torque_limits', dof_torque_limits.tolist())
    if isinstance(default_dof_pos, dict):
        ddp_inds, default_dof_pos = dict2list(default_dof_pos, dof_names, default_dof_pos.pop('_default', None))
    if default_dof_pos is not None:
        def_dof_pos = np.zeros(num_dofs, dtype=np_dtype)
        def_dof_pos[ddp_inds] = default_dof_pos
        default_dof_pos = def_dof_pos

    # get dof state tensor
    dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    dof_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_pos = np.zeros(num_dofs, dtype=np_dtype)
    action_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_tau = np.zeros(num_dofs, dtype=np_dtype)

    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    default_dof_lower_tensor = torch.zeros(num_dofs, device=device)
    default_dof_upper_tensor = torch.zeros(num_dofs, device=device)
    default_dof_lower_tensor[:] = torch.tensor(articulation_builder.joint_limit_lower, device=device)
    default_dof_upper_tensor[:] = torch.tensor(articulation_builder.joint_limit_upper, device=device)
    default_dof_pos_tensor = torch.zeros(num_dofs, device=device)
    default_dof_vel_tensor = torch.zeros(num_dofs, device=device)
    if default_dof_pos is not None:
        default_dof_pos_tensor[:] = torch.from_numpy(default_dof_pos).to(device=device)
    sim_time = 0.0

    def reset_fn():
        nonlocal state_0, state_1
        state_0 = model.state()
        state_1 = model.state()
        wp.sim.eval_fk(model, model.joint_q, model.joint_qd, None, state_0)
        print(state_0.joint_q)
        print(state_0.joint_qd)
        print(state_1.joint_q)
        print(state_1.joint_qd)

    def step_fn():
        for i in range(decimation):
            joint_q, joint_qd = state_0.joint_q, state_0.joint_qd
            joint_q, joint_qd = to_torch(joint_q), to_torch(joint_qd)
            _dof_pos, _dof_vel = joint_q[-num_dofs:], joint_qd[-num_dofs:]
            _action_pos = torch.from_numpy(action_pos).view(num_dofs).to(device=device)
            _action_vel = torch.from_numpy(action_vel).view(num_dofs).to(device=device)
            _action_tau = torch.from_numpy(action_tau).view(num_dofs).to(device=device)
            tau = None
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                tau = torch.zeros(num_dofs, device=device) if not ctrl_inds_tau else _action_tau
                if ctrl_inds_pos:
                    q_ctrl = _action_pos
                    if clip_q_ctrl:
                        q_ctrl = torch.clamp(q_ctrl, default_dof_lower_tensor, default_dof_upper_tensor)
                    torques = kps * (q_ctrl - _dof_pos) - kds * _dof_vel
                    tau[..., ctrl_inds_pos] = torques[..., ctrl_inds_pos]
                if ctrl_inds_vel:
                    qd_ctrl = _action_vel
                    torques = kds * (qd_ctrl - _dof_vel)
                    tau[..., ctrl_inds_vel] = torques[..., ctrl_inds_vel]
                tau = torch.clip(tau, -dof_torque_limits, dof_torque_limits)
                if verbose:
                    print('ctrl', q_ctrl.cpu().tolist())
                    # print('x', _root_states.cpu().tolist())
                    print('q', _dof_pos.cpu().tolist())
                    print('qd', _dof_vel.cpu().tolist())
                    print('tau', tau.cpu().tolist())
            else:
                if ctrl_inds_pos:
                    control.joint_act.assign(from_torch(_action_pos))
                if ctrl_inds_vel:
                    control.joint_act.assign(from_torch(_action_vel))
            if ctrl_inds_tau:
                tau = _action_tau if tau is None else tau
            if tau is not None:
                control.joint_act.assign(from_torch(tau))
            # print(control.joint_act)
            sim_step()

        joint_q, joint_qd = state_0.joint_q, state_0.joint_qd
        # print('joint_q', joint_q)
        joint_q, joint_qd = to_torch(joint_q), to_torch(joint_qd)
        _dof_pos, _dof_vel = joint_q[-num_dofs:], joint_qd[-num_dofs:]
        # _root_states = data.root_state_w
        _root_pose = joint_q[:7]
        _root_vel = joint_qd[:6]
        print('_root_pose', _root_pose)
        print('_dof_pos', _dof_pos)
        # print(_dof_vel)

        _rpy = get_euler_xyz(_root_pose[3:7])
        dof_pos[:] = _dof_pos.flatten().cpu().numpy()
        dof_vel[:] = _dof_vel.flatten().cpu().numpy()
        root_states[:7] = _root_pose.flatten().cpu().numpy()
        root_states[7:] = _root_vel.flatten().cpu().numpy()
        imu[0:3] = _rpy.flatten().cpu().numpy()
        if renderer is None:
            return
        # with wp.ScopedTimer("render"):
        renderer.begin_frame(sim_time)
        renderer.render(state_0)
        renderer.end_frame()
        sim_time += dt

    def close_fn():
        pass

    return locals()
