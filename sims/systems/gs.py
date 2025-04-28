def init_fn(
    urdf_path=None,
    dt=0.02,
    Kp=None,
    Kd=None,
    torque_limits=None,
    decimation=1,
    fix_base_link=False,
    default_dof_pos=None,
    default_root_states=None,
    ctrl_joints_pos='*',
    ctrl_joints_vel=None,
    ctrl_joints_tau=None,
    compute_torque=False,
    clip_q_ctrl=True,
    headless=False,
    verbose=False,
    device='cuda',
    num_envs=1,
    substeps=2,
    enable_self_collision=False,
):
    import torch
    import numpy as np
    from sims.utils import update_array, update_obj, list2slice, get_ctrl_inds
    from sims.torch_utils import get_euler_xyz
    import genesis as gs
    from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

    backend = gs.cpu if device == 'cpu' else gs.gpu
    gs.init(logging_level="warning", backend=backend)
    np_dtype = np.float32
    tc_dtype = gs.tc_float
    sim_dt = dt / decimation

    init_pos = [0, 0, 1]
    init_rot = [0, 0, 0, 1]
    init_lin_vel = [0, 0, 0]
    init_ang_vel = [0, 0, 0]
    if default_root_states is not None:
        init_pos = default_root_states[:3]
        init_rot = default_root_states[3:7]
        init_lin_vel = default_root_states[7:10]
        init_ang_vel = default_root_states[10:13]
    init_pos = np.array(init_pos)
    init_rot = np.array(init_rot)
    base_init_quat = torch.from_numpy(init_rot).to(
        device=device,
        dtype=tc_dtype,
    ).unsqueeze(0)
    init_pos_tensor = torch.from_numpy(init_pos).to(
        device=device,
        dtype=tc_dtype,
    ).unsqueeze(0)
    init_rot_tensor = torch.from_numpy(init_rot).to(
        device=device,
        dtype=tc_dtype,
    ).unsqueeze(0)
    inv_base_init_quat = inv_quat(base_init_quat)
    print('inv_base_init_quat', inv_base_init_quat)

    show_viewer = not headless
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim_dt, substeps=substeps),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(1 / sim_dt),
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            n_rendered_envs=1,
            shadow=False,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=sim_dt,
            constraint_solver=gs.constraint_solver.Newton,
            # integrator=gs.integrator.Euler,
            # integrator=gs.integrator.implicitfast,
            enable_collision=True,
            enable_joint_limit=True,
            enable_self_collision=enable_self_collision,
            iterations=200,
            ls_iterations=100,
            use_contact_island=True,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane(),)
    # add robot
    robot = scene.add_entity(gs.morphs.URDF(
        file=urdf_path,
        pos=init_pos,
        quat=init_rot,
        fixed=fix_base_link,
    ),)
    # build
    scene.build(n_envs=num_envs)
    # names to indices
    dof_names = list(map(lambda j: j.name, robot._joints))
    # dof_names = [n for n in dof_names if n != 'joint_base_link']
    dof_names = dof_names[1:]
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in dof_names]
    assert all(map(lambda x: isinstance(x, int), motor_dofs))
    print('motor_dofs', motor_dofs)

    print('dof_names', dof_names)

    num_dofs = len(dof_names)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    kps = np.zeros(num_dofs, dtype=np_dtype)
    kds = np.zeros(num_dofs, dtype=np_dtype)
    dof_torque_limits = torch.zeros(num_dofs, dtype=tc_dtype, device=device)
    robot_dof_effort = robot.get_dofs_force_range(motor_dofs)[1]
    # print(robot_dof_effort)
    dof_torque_limits[:] = torch.tensor(robot_dof_effort, dtype=tc_dtype, device=device)
    update_array(kps, Kp, dof_names)
    update_array(kds, Kd, dof_names)
    update_array(dof_torque_limits, torque_limits, dof_names)
    def_dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    update_array(def_dof_pos, default_dof_pos, dof_names)
    action_pos = np.zeros(num_dofs, dtype=np_dtype)
    action_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_tau = np.zeros(num_dofs, dtype=np_dtype)
    dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    dof_vel = np.zeros(num_dofs, dtype=np_dtype)
    dof_tau = np.zeros(num_dofs, dtype=np_dtype)
    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    # PD control parameters
    robot.set_dofs_kp(kps, motor_dofs)
    robot.set_dofs_kv(kds, motor_dofs)
    kps = torch.from_numpy(kps).to(
        device=device,
        dtype=tc_dtype,
    )
    kds = torch.from_numpy(kds).to(
        device=device,
        dtype=tc_dtype,
    )
    print('dof_torque_limits', dof_torque_limits.cpu().numpy().tolist())
    print('kps', kps.cpu().numpy().tolist())
    print('kds', kds.cpu().numpy().tolist())
    default_dof_pos_tensor = torch.from_numpy(def_dof_pos).to(
        device=device,
        dtype=tc_dtype,
    ).unsqueeze(0)
    default_dof_lower_tensor, default_dof_upper_tensor = robot.get_dofs_limit(motor_dofs)
    print('default_dof_lower_tensor', default_dof_lower_tensor)
    print('default_dof_upper_tensor', default_dof_upper_tensor)

    def step_fn():

        for i in range(decimation):
            # joint_q, joint_qd = to_torch(joint_q), to_torch(joint_qd)
            # _dof_pos, _dof_vel = joint_q[-num_dofs:], joint_qd[-num_dofs:]
            _action_pos = torch.from_numpy(action_pos).view(num_dofs).to(device=device)
            _action_vel = torch.from_numpy(action_vel).view(num_dofs).to(device=device)
            _action_tau = torch.from_numpy(action_tau).view(num_dofs).to(device=device)
            tau = None
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                _dof_pos = robot.get_dofs_position(motor_dofs)
                _dof_vel = robot.get_dofs_velocity(motor_dofs)
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
                    robot.control_dofs_position(_action_pos.unsqueeze(0), motor_dofs)
                if ctrl_inds_vel:
                    robot.control_dofs_velocity(_action_vel.unsqueeze(0), motor_dofs)
            if ctrl_inds_tau:
                tau = _action_tau if tau is None else tau
            if tau is not None:
                dof_tau[:] = tau
                robot.control_dofs_force(tau.unsqueeze(0), motor_dofs)
            scene.step()
            # print(control.joint_act)

        base_pos = robot.get_pos()
        base_quat = robot.get_quat()
        base_euler = quat_to_xyz(transform_quat_by_quat(torch.ones_like(base_quat) * inv_base_init_quat, base_quat))
        inv_base_quat = inv_quat(base_quat)
        # print('base_euler', base_euler)
        # print('base_quat', base_quat)
        # print('inv_base_quat', inv_base_quat)
        root_lin_vel = robot.get_vel()
        base_lin_vel = transform_by_quat(root_lin_vel, inv_base_quat)
        root_ang_vel = robot.get_ang()
        base_ang_vel = transform_by_quat(root_ang_vel, inv_base_quat)
        # base_ang_vel[:, 1] = -base_ang_vel[:, 1]
        _dof_pos = robot.get_dofs_position(motor_dofs)
        _dof_vel = robot.get_dofs_velocity(motor_dofs)
        # joint_q, joint_qd = state_0.joint_q, state_0.joint_qd
        # _dof_pos, _dof_vel = _dof_pos[-num_dofs:], _dof_vel[-num_dofs:]
        # _root_states = data.root_state_w
        # _root_pose = joint_q[:7]
        # _root_vel = joint_qd[:6]
        # print('_root_pose', _root_pose)
        # print('_dof_pos', _dof_pos)
        # print(_dof_vel)

        # _rpy = get_euler_xyz(base_quat)
        # _rpy[:, 1] = -_rpy[:, 1]
        # _rpy = -_rpy
        # _rpy = np.deg2rad(base_euler)
        _rpy = -np.deg2rad(base_euler)
        # print('_rpy', _rpy)
        # print('root_ang_vel', root_ang_vel)
        # print('base_ang_vel', base_ang_vel)
        # _rpy = base_euler
        dof_pos[:] = _dof_pos.flatten().cpu().numpy()
        dof_vel[:] = _dof_vel.flatten().cpu().numpy()
        # print('base_ang_vel', base_ang_vel)
        # print('_rpy', _rpy)
        # print('dof_pos', dof_pos)
        # print('dof_vel', dof_vel)
        root_states[0:3] = base_pos.flatten().cpu().numpy()
        root_states[3:7] = base_quat.flatten().cpu().numpy()
        root_states[7:10] = root_lin_vel.flatten().cpu().numpy()
        root_states[10:13] = root_ang_vel.flatten().cpu().numpy()
        imu[0:3] = _rpy.flatten().cpu().numpy()
        # imu[3:6] = lin_acc
        imu[6:9] = base_ang_vel.flatten().cpu().numpy()
        imu[9:12] = base_lin_vel.flatten().cpu().numpy()
        # imu[6:9] = root_ang_vel.flatten().cpu().numpy()
        if headless:
            return

    def close_fn():
        print('close')

    def reset_fn():
        envs_idx = [0]
        robot.set_dofs_position(
            position=default_dof_pos_tensor,
            dofs_idx_local=motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        # base_pos[envs_idx] = base_init_pos
        # base_quat[envs_idx] = base_init_quat.reshape(1, -1)
        robot.set_pos(
            # base_pos[envs_idx],
            init_pos_tensor,
            zero_velocity=False,
            envs_idx=envs_idx)
        robot.set_quat(
            # base_quat[envs_idx],
            init_rot_tensor,
            zero_velocity=False,
            envs_idx=envs_idx)
        # base_lin_vel[envs_idx] = 0
        # base_ang_vel[envs_idx] = 0
        robot.zero_all_dofs_velocity(envs_idx)
        _dof_pos = robot.get_dofs_position(motor_dofs)
        _dof_vel = robot.get_dofs_velocity(motor_dofs)
        dof_pos[:] = _dof_pos.flatten().cpu().numpy()
        dof_vel[:] = _dof_vel.flatten().cpu().numpy()

    return {
        'dt': dt,
        'step_fn': step_fn,
        'close_fn': close_fn,
        'reset_fn': reset_fn,
        'dof_names': dof_names,
        'dof_pos': dof_pos,
        'dof_vel': dof_vel,
        'dof_tau': dof_tau,
        'action_pos': action_pos,
        'action_vel': action_vel,
        'action_tau': action_tau,
        'root_states': root_states,
        'imu': imu,
    }
