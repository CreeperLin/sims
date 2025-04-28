def init_fn(
    urdf_path=None,
    asset_root='/',
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
    device=None,
    sim_device='cpu',
    pipeline='cpu',
    num_envs=1,
    asset_options=None,
    collapse_fixed_joints=True,
    sim_params=None,
    plane_params=None,
    rs_props=None,
    rb_props=None,
    seed=1,
    cam_pos=[3, 2, 1],
    cam_target=[0, 0, 1],
    sim=None,
    viewer=None,
    env=None,
    verbose=False,
    physics_engine='physx',
    lat=0,
    dc=0,
    imu_name='base',
    use_imu_link=False,
    use_force_sensor=False,
    tau_stiffness=0.,
    tau_damping=0.,
    tau_idx=-1,
):
    import os
    from isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
    from isaacgym.torch_utils import to_torch
    import torch
    import math
    import numpy as np
    from sims.utils import update_array, update_obj, set_seed, list2slice, get_ctrl_inds
    from sims.torch_utils import get_euler_xyz
    from isaacgym.torch_utils import quat_rotate_inverse

    if seed is not None:
        set_seed(seed)

    dtype = torch.float
    np_dtype = np.float32

    gym = gymapi.acquire_gym()

    sim_device = device if device is not None else sim_device
    pipeline = pipeline.lower()
    use_gpu_pipeline = (pipeline in ('gpu', 'cuda') and sim_device != 'cpu')
    if sim is None:
        args = type(
            'args',
            (),
            {
                'headless': headless,
                'nographics': False,
                'sim_device': sim_device,
                'pipeline': pipeline,
                # 'graphics_device_id': 0,
                'graphics_device_id': (-1 if headless else 0),
                'num_threads': 0,
                'subscenes': 0,
                'slices': None,
            })
        args.num_envs = num_envs
        args.sim_device_type, args.compute_device_id = gymutil.parse_device_str(sim_device)
        if physics_engine == 'physx':
            args.physics_engine = gymapi.SIM_PHYSX
        elif physics_engine == 'flex':
            args.physics_engine = gymapi.SIM_FLEX
        args.use_gpu_pipeline = use_gpu_pipeline
        args.use_gpu = (args.sim_device_type == 'cuda')
        args.slices = args.subscenes
        sim_params_updates = {} if sim_params is None else sim_params
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        val = sim_params_updates.get('gravity')
        if val is not None:
            sim_params_updates['gravity'] = gymapi.Vec3(*val)
        val = sim_params_updates.get('physx', {}).get('contact_collection')
        if val is not None:
            sim_params_updates['physx']['contact_collection'] = gymapi.ContactCollection(val)
        update_obj(sim_params, sim_params_updates)
        sim_dt = dt / decimation
        print('sim_dt', sim_dt)
        sim_params.dt = sim_dt
        sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        if sim is None:
            raise Exception('Failed to create sim')
    device = sim_device if use_gpu_pipeline else 'cpu'

    opts = gymapi.AssetOptions()
    opts.fix_base_link = fix_base_link
    opts.disable_gravity = disable_gravity
    if compute_torque:
        opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
    if collapse_fixed_joints:
        opts.collapse_fixed_joints = collapse_fixed_joints
    update_obj(opts, ({} if asset_options is None else asset_options))
    print('collapse_fixed_joints', opts.collapse_fixed_joints)
    asset_root = os.path.abspath(asset_root)
    urdf_path = os.path.abspath(urdf_path) if asset_root == '/' else urdf_path
    robot_asset = gym.load_asset(sim, asset_root, urdf_path, opts)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    robot_dof_props = gym.get_asset_dof_properties(robot_asset)
    rs_props_updates = {} if rs_props is None else rs_props
    rigid_shape_props = gym.get_asset_rigid_shape_properties(robot_asset)
    for s in range(len(rigid_shape_props)):
        update_obj(rigid_shape_props[s], rs_props_updates)
    gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
    dof_names = gym.get_asset_dof_names(robot_asset)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    rest_inds = list(set(range(num_dofs)) - set(sum(ctrl_inds_all, [])))
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    # print('robot_dof_props', robot_dof_props)
    robot_dof_drive_mode = robot_dof_props['driveMode']
    robot_dof_drive_mode.fill(gymapi.DOF_MODE_NONE)
    robot_dof_lower_limits = robot_dof_props['lower']
    robot_dof_upper_limits = robot_dof_props['upper']
    robot_dof_stiffness = robot_dof_props['stiffness']
    robot_dof_damping = robot_dof_props['damping']
    robot_dof_effort = robot_dof_props['effort']
    robot_dof_velocity = robot_dof_props['velocity']
    if compute_torque:
        robot_dof_drive_mode[ctrl_inds_pos] = gymapi.DOF_MODE_EFFORT
        robot_dof_drive_mode[ctrl_inds_vel] = gymapi.DOF_MODE_EFFORT
    else:
        robot_dof_drive_mode[ctrl_inds_pos] = gymapi.DOF_MODE_POS
        robot_dof_drive_mode[ctrl_inds_vel] = gymapi.DOF_MODE_VEL
        print('robot_dof_stiffness', robot_dof_stiffness)
        print('robot_dof_damping', robot_dof_damping)
    robot_dof_drive_mode[ctrl_inds_tau] = gymapi.DOF_MODE_EFFORT
    dof_torque_limits = torch.zeros(num_dofs, dtype=dtype, device=device)
    dof_torque_limits[:] = torch.tensor(robot_dof_effort, dtype=dtype, device=device)
    update_array(dof_torque_limits, torque_limits, dof_names)
    kps = torch.zeros(num_dofs, dtype=dtype, device=device)
    kps[:] = torch.tensor(robot_dof_stiffness, dtype=dtype, device=device)
    update_array(kps, Kp, dof_names)
    kds = torch.zeros(num_dofs, dtype=dtype, device=device)
    kds[:] = torch.tensor(robot_dof_damping, dtype=dtype, device=device)
    update_array(kds, Kd, dof_names)
    print('dof_torque_limits', dof_torque_limits.cpu().numpy().tolist())
    print('kps', kps.cpu().numpy().tolist())
    print('kds', kds.cpu().numpy().tolist())
    if not compute_torque:
        robot_dof_stiffness[:] = kps.cpu().numpy()
        robot_dof_damping[:] = kds.cpu().numpy()
    else:
        robot_dof_stiffness[:] = tau_stiffness
        robot_dof_damping[:] = tau_damping
    if torque_limits is not None:
        robot_dof_effort[:] = dof_torque_limits.cpu().numpy()
    def_dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    update_array(def_dof_pos, default_dof_pos, dof_names)
    if len(rest_inds):
        robot_dof_stiffness[rest_inds] = 0.0
        robot_dof_damping[rest_inds] = 0.0
    print('num_dofs', num_dofs)
    print('dof_names', dof_names)
    # print('default_dof_pos', default_dof_pos)
    print('def_dof_pos', def_dof_pos)
    print('robot_dof_drive_mode', robot_dof_drive_mode)
    print('robot_dof_stiffness', robot_dof_stiffness)
    print('robot_dof_damping', robot_dof_damping)
    print('robot_dof_effort', robot_dof_effort)
    print('robot_dof_velocity', robot_dof_velocity)
    print('robot_dof_lower_limits', robot_dof_lower_limits)
    print('robot_dof_upper_limits', robot_dof_upper_limits)

    default_dof_pos_tensor = to_torch(def_dof_pos, device=device)
    default_dof_lower_tensor = to_torch(robot_dof_lower_limits, device=device)
    default_dof_upper_tensor = to_torch(robot_dof_upper_limits, device=device)
    dof_velocity_limits = to_torch(robot_dof_velocity, device=device)
    rb_names = gym.get_asset_rigid_body_names(robot_asset)
    print('rb_names', len(rb_names), rb_names)
    rb_dict = gym.get_asset_rigid_body_dict(robot_asset)
    print('rb_dict', rb_dict)
    imu_idx = rb_dict[([n for n in rb_dict if imu_name in n] or [rb_names[0]])[0]]
    print('imu_idx', imu_idx, use_imu_link)

    if use_force_sensor:
        sensor_pose = gymapi.Transform()
        sensor_options = gymapi.ForceSensorProperties()
        # sensor_options.enable_forward_dynamics_forces = False  # for example gravity
        sensor_options.enable_forward_dynamics_forces = True
        sensor_options.enable_constraint_solver_forces = True  # for example contacts
        # sensor_options.use_world_frame = True
        sensor_options.use_world_frame = False
        gym.create_asset_force_sensor(robot_asset, imu_idx, sensor_pose, sensor_options)

    if env is None:
        # add ground plane
        plane_params_updates = {} if plane_params is None else plane_params
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        update_obj(plane_params, plane_params_updates)
        gym.add_ground(sim, plane_params)
        # configure env grid
        num_per_row = int(math.sqrt(num_envs))
        spacing = 1.0
        if num_envs <= 1:
            spacing = 0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print('Creating %d environments' % num_envs)
        envs = []
        actors = []
        for i in range(num_envs):
            robot_pose = gymapi.Transform()
            mask = 0 if self_collisions else 1
            env = gym.create_env(sim, env_lower, env_upper, num_per_row)
            envs.append(env)
            actor = gym.create_actor(env, robot_asset, robot_pose, 'actor', i, mask, 0)
            actors.append(actor)
            gym.set_actor_dof_properties(env, actor, robot_dof_props)
            body_props = gym.get_actor_rigid_body_properties(env, actor)
            rb_props_updates = {} if rb_props is None else rb_props
            update_obj(body_props, rb_props_updates)
            gym.set_actor_rigid_body_properties(env, actor, body_props, recomputeInertia=True)
        print('self_collisions', self_collisions)

        gym.prepare_sim(sim)

    mass = 0
    for i, p in enumerate(body_props):
        mass += p.mass
    print('mass', mass)

    # create viewer
    if viewer is None and not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print('Failed to create viewer')
    if viewer is not None:
        cam_pos = gymapi.Vec3(*cam_pos)
        cam_target = gymapi.Vec3(*cam_target)
        gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    _root_states_tensor = gym.acquire_actor_root_state_tensor(sim)
    with torch.inference_mode(False):
        _root_states = gymtorch.wrap_tensor(_root_states_tensor).view(num_envs, -1, 13)

    # get rigid body state tensor
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states).view(num_envs, -1, 13)

    _rb_forces = gym.acquire_net_contact_force_tensor(sim)
    rb_forces = gymtorch.wrap_tensor(_rb_forces).view(num_envs, -1, 3)

    if use_force_sensor:
        _force_sensor = gym.acquire_force_sensor_tensor(sim)
        force_sensor = gymtorch.wrap_tensor(_force_sensor)

    # get dof state tensor
    _dof_states = gym.acquire_dof_state_tensor(sim)
    with torch.inference_mode(False):
        dof_states = gymtorch.wrap_tensor(_dof_states)
        _dof_pos = dof_states[:, 0].view(num_envs, -1)
        _dof_vel = dof_states[:, 1].view(num_envs, -1)
    dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    dof_vel = np.zeros(num_dofs, dtype=np_dtype)
    dof_tau = np.zeros(num_dofs, dtype=np_dtype)
    action_pos = np.zeros(num_dofs, dtype=np_dtype)
    action_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_tau = np.zeros(num_dofs, dtype=np_dtype)

    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)
    if default_root_states is not None:
        default_root_states_tensor = torch.tensor(default_root_states, device=device, dtype=dtype)

    def reset_fn():
        _dof_pos[:] = default_dof_pos_tensor.view(*_dof_pos.shape)
        _dof_vel[:] = 0.
        gym.set_dof_state_tensor(sim, _dof_states)
        if default_root_states is not None:
            _root_states[:] = default_root_states_tensor.view(*_root_states.shape)
            # gym.set_actor_root_state_tensor(sim, _root_states_tensor)
            gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(_root_states))
        # gym.fetch_results(sim, True)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        # print(_root_states.cpu().tolist())
        dof_pos[:] = _dof_pos.flatten().cpu().numpy()
        dof_vel[:] = _dof_vel.flatten().cpu().numpy()
        dof_tau[:] = 0

    if lat:
        lat = int(decimation * lat) if isinstance(lat, float) else lat
        print('lat', lat)
        last_action_pos = torch.zeros(num_envs, num_dofs, dtype=dtype, device=device)
        last_action_vel = torch.zeros(num_envs, num_dofs, dtype=dtype, device=device)
        last_action_tau = torch.zeros(num_envs, num_dofs, dtype=dtype, device=device)
        last_action_pos[:] = default_dof_pos_tensor.view(*last_action_pos.shape)

    tau_idx = decimation + tau_idx if tau_idx < 0 else tau_idx

    def step_fn():
        if viewer is not None:
            # if device != 'cpu':
            # gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
        # gym.fetch_results(sim, True)
        # gym.refresh_dof_state_tensor(sim)
        _action_pos = torch.from_numpy(action_pos).view(num_envs, -1).to(device=device)
        _action_vel = torch.from_numpy(action_vel).view(num_envs, -1).to(device=device)
        _action_tau = torch.from_numpy(action_tau).view(num_envs, -1).to(device=device)
        # _action_pos += default_dof_pos_tensor
        if verbose:
            # if True:
            print('q', _dof_pos.cpu().tolist())
            print('x', _root_states.cpu().tolist())
            print('qc', _action_pos.cpu().tolist())
        tau = None
        for i in range(decimation):
            if dc:
                torque_sat = dof_torque_limits
                torque_limits = torch.clamp(torque_sat * (1 - torch.abs(_dof_vel / dof_velocity_limits)), 0,
                                            dof_torque_limits)
            else:
                torque_limits = dof_torque_limits
            ctrl_pos = _action_pos
            if lat and lat > i:
                # print('lat', lat, i)
                ctrl_pos = last_action_pos
            if (ctrl_inds_pos or ctrl_inds_vel):
                tau = torch.zeros_like(_dof_pos) if not ctrl_inds_tau else _action_tau
                if ctrl_inds_pos:
                    ctrl = ctrl_pos
                    if clip_q_ctrl:
                        # print(ctrl, default_dof_lower_tensor)
                        ctrl = torch.clamp(ctrl, default_dof_lower_tensor, default_dof_upper_tensor)
                        # print(ctrl)
                    torques = kps * (ctrl - _dof_pos) - kds * _dof_vel
                    tau[:, ctrl_inds_pos] = torques[:, ctrl_inds_pos]
                if ctrl_inds_vel:
                    dof_vel_ctrl = _action_vel
                    torques = kds * (dof_vel_ctrl - _dof_vel)
                    tau[:, ctrl_inds_vel] = torques[:, ctrl_inds_vel]
                if compute_torque:
                    if not ctrl_inds_tau:
                        tau = torch.clip(tau, -torque_limits, torque_limits)
                        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau))
                else:
                    if ctrl_inds_pos:
                        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(ctrl_pos))
                    if ctrl_inds_vel:
                        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(_action_vel))
                # if verbose:
                #     print('q', _dof_pos.cpu().tolist())
                #     print('qd', _dof_vel.cpu().tolist())
                #     print('tau', tau.cpu().tolist())
            if ctrl_inds_tau:
                tau = torch.clip(_action_tau, -torque_limits, torque_limits)
                gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(tau))
            gym.simulate(sim)
            # if device == 'cpu':
            # gym.fetch_results(sim, True)
            gym.fetch_results(sim, True)
            gym.refresh_dof_state_tensor(sim)
            if tau is not None and tau_idx == i:
                dof_tau[:] = tau.flatten().cpu().numpy()

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_force_tensor(sim)
        # if use_force_sensor:
        # gym.refresh_force_sensor_tensor(sim)
        # gym.refresh_jacobian_tensors(sim)
        # gym.refresh_mass_matrix_tensors(sim)
        dof_pos[:] = _dof_pos.flatten().cpu().numpy()
        dof_vel[:] = _dof_vel.flatten().cpu().numpy()
        rsnp = _root_states.flatten().cpu().numpy()
        root_states[0:13] = rsnp[0:13]
        if use_imu_link:
            imu_states = rb_states[:, imu_idx]
        else:
            imu_states = _root_states.view(-1, 13)
        imu_quat = imu_states[:, 3:7]
        imu_lin_vel = quat_rotate_inverse(imu_quat, imu_states[:, 7:10])
        imu_ang_vel = quat_rotate_inverse(imu_quat, imu_states[:, 10:13])
        _rpy = get_euler_xyz(imu_quat)
        imu[0:3] = _rpy.flatten().cpu().numpy()
        imu[6:9] = imu_ang_vel.flatten().cpu().numpy()
        imu[9:12] = imu_lin_vel.flatten().cpu().numpy()
        if use_force_sensor:
            imu_acc = force_sensor[:, :3] / mass
        else:
            # print(rb_forces)
            imu_acc = rb_forces[:, imu_idx, :3] / mass
        imu[3:6] = imu_acc.flatten().cpu().numpy()

        if lat:
            last_action_pos[:] = _action_pos
            last_action_vel[:] = _action_vel
            last_action_tau[:] = _action_tau

    def close_fn():
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

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
