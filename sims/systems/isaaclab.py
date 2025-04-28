def init_fn(
    urdf_path=None,
    usd_path=None,
    fix_base_link=False,
    disable_gravity=False,
    dt=0.02,
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
    device='cpu',
    num_envs=1,
    terrain_props=None,
    robot_props=None,
    sim_props=None,
    seed=1,
    sim=None,
    verbose=False,
    # ground_usd_path=None,
    ground_usd_path='default_environment.usd',
):
    import torch
    import numpy as np
    from sims.utils import update_obj, update_array, set_seed, list2slice, get_ctrl_inds, obj2dict
    from sims.torch_utils import get_euler_xyz

    try:
        import omni.isaac.lab
        lab_mod = 'omni.isaac.lab'
    except ImportError:
        try:
            import isaaclab
            lab_mod = 'isaaclab'
        except ImportError:
            raise
    # from omni.isaac.lab.app import AppLauncher
    app = __import__(f'{lab_mod}.app', fromlist=[''])
    import sys
    sys.argv = sys.argv[:1]
    render = not headless
    # render = False
    # headless = True
    args_cli = {
        'num_envs': num_envs,
        # 'disable_fabric': True,
        'enable_cameras': render,
        'headless': headless,
        'device': device,
    }
    app_launcher = app.AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # import omni.isaac.lab.sim as sim_utils
    sim_utils = __import__(f'{lab_mod}.sim', fromlist=[''])
    assets = __import__(f'{lab_mod}.assets', fromlist=[''])
    terrains = __import__(f'{lab_mod}.terrains', fromlist=[''])
    actuators = __import__(f'{lab_mod}.actuators', fromlist=[''])
    # from omni.isaac.lab.assets import ArticulationCfg
    # from omni.isaac.lab.terrains import TerrainImporterCfg
    # from omni.isaac.lab.assets import Articulation
    # from omni.isaac.lab.actuators import ImplicitActuatorCfg

    dtype = torch.float
    np_dtype = np.float32
    sim_dt = dt / decimation
    print(sim_dt, decimation)

    if seed is not None:
        set_seed(seed)

    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt,)
    update_obj(sim_cfg, sim_props)
    print('sim_cfg', obj2dict(sim_cfg))
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)

    terrain_cfg = terrains.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    update_obj(terrain_cfg, terrain_props)
    # size: tuple[float, float] = (2.0e6, 2.0e6)
    ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=terrain_cfg.physics_material,
                                                # size=size,
                                               )
    if ground_usd_path is not None:
        ground_plane_cfg.usd_path = ground_usd_path
    ground_plane_cfg.func(terrain_cfg.prim_path, ground_plane_cfg)

    init_pos = np.array([0, 0, 1])
    init_rot = np.array([1.0, 0.0, 0.0, 0.0])
    init_joint_pos = {}
    if default_root_states is not None:
        init_pos[:] = default_root_states[0:3]
        init_rot[1:] = default_root_states[3:6]
        init_rot[0] = default_root_states[6]
        default_root_states_w_first = torch.tensor((init_pos.tolist() + init_rot.tolist()), device=device)
    init_state = assets.ArticulationCfg.InitialStateCfg(
        pos=init_pos,
        rot=init_rot,
        joint_pos=init_joint_pos,
    )
    actuator_cfgs = {}
    actuator_cfgs["default"] = actuators.ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        stiffness=0.0,
        damping=0.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=self_collisions,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        rigid_body_enabled=True,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=100.0,
        enable_gyroscopic_forces=True,
    )
    if urdf_path is not None:
        spawn_cfg = sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            fix_base=fix_base_link,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)),
        )
    if usd_path is not None:
        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            fix_base=fix_base_link,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )
    robot_cfg = assets.ArticulationCfg(
        prim_path="/World/Robot",
        spawn=spawn_cfg,
        init_state=init_state,
        actuators=actuator_cfgs,
    )
    print('robot_cfg', obj2dict(robot_cfg))
    update_obj(robot_cfg, robot_props)

    robot = assets.Articulation(robot_cfg)

    sim.reset()
    dof_names = robot.joint_names
    num_dofs = robot.num_joints
    print('dof_names', num_dofs, dof_names)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    kps = torch.zeros(num_dofs, dtype=dtype, device=device)
    kds = torch.zeros(num_dofs, dtype=dtype, device=device)
    dof_torque_limits = torch.zeros(num_dofs, dtype=dtype, device=device)
    update_array(kps, Kp, dof_names)
    update_array(kds, Kd, dof_names)
    update_array(dof_torque_limits, torque_limits, dof_names)
    print('dof_torque_limits', dof_torque_limits.cpu().numpy().tolist())
    print('kps', kps.cpu().numpy().tolist())
    print('kds', kds.cpu().numpy().tolist())
    def_dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    update_array(def_dof_pos, default_dof_pos, dof_names)

    # get dof state tensor
    dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    dof_vel = np.zeros(num_dofs, dtype=np_dtype)
    dof_tau = np.zeros(num_dofs, dtype=np_dtype)
    action_pos = np.zeros(num_dofs, dtype=np_dtype)
    action_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_tau = np.zeros(num_dofs, dtype=np_dtype)

    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    # default_joint_limit = robot.data.default_joint_limits.squeeze()
    default_joint_pos_limits = robot.data.default_joint_pos_limits.squeeze()
    default_dof_lower_tensor = default_joint_pos_limits[..., 0]
    default_dof_upper_tensor = default_joint_pos_limits[..., 1]
    default_dof_pos_tensor = torch.zeros(num_dofs, device=device)
    default_dof_vel_tensor = torch.zeros(num_dofs, device=device)
    default_dof_pos_tensor[:] = torch.from_numpy(def_dof_pos).to(device=device)

    if render:
        sim.render()
    actuator = next(iter(robot.actuators.values()))

    if not compute_torque:
        actuator.stiffness[:] = kps
        actuator.damping[:] = kds
        actuator.effort_limit[:] = dof_torque_limits
        robot.write_joint_stiffness_to_sim(actuator.stiffness, joint_ids=actuator.joint_indices)
        robot.write_joint_damping_to_sim(actuator.damping, joint_ids=actuator.joint_indices)
        robot.write_joint_effort_limit_to_sim(actuator.effort_limit, joint_ids=actuator.joint_indices)

    print('actuator.stiffness', actuator.stiffness)
    print('actuator.damping', actuator.damping)
    print('actuator.effort_limit', actuator.effort_limit)

    def reset_fn():
        robot.reset()
        if default_root_states is not None:
            robot.write_root_pose_to_sim(default_root_states_w_first)
        robot.write_joint_state_to_sim(default_dof_pos_tensor, default_dof_vel_tensor)

    def step_fn():
        if not simulation_app.is_running():
            return
        for i in range(decimation):
            _dof_pos, _dof_vel = robot.data.joint_pos, robot.data.joint_vel
            _action_pos = torch.from_numpy(action_pos).view(-1, num_dofs).to(device=device)
            _action_vel = torch.from_numpy(action_vel).view(-1, num_dofs).to(device=device)
            _action_tau = torch.from_numpy(action_tau).view(-1, num_dofs).to(device=device)
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                tau = torch.zeros_like(_dof_pos) if not ctrl_inds_tau else _action_tau
                if ctrl_inds_pos:
                    q_ctrl = _action_pos
                    if clip_q_ctrl:
                        q_ctrl = torch.clamp(q_ctrl, default_dof_lower_tensor, default_dof_upper_tensor)
                    torques = kps * (q_ctrl - _dof_pos) - kds * _dof_vel
                    tau[:, ctrl_inds_pos] = torques[:, ctrl_inds_pos]
                if ctrl_inds_vel:
                    qd_ctrl = _action_vel
                    torques = kds * (qd_ctrl - _dof_vel)
                    tau[:, ctrl_inds_vel] = torques[:, ctrl_inds_vel]
                if not ctrl_inds_tau:
                    tau = torch.clip(tau, -dof_torque_limits, dof_torque_limits)
                    robot.set_joint_effort_target(tau)
                if verbose:
                    _root_states = robot.data.root_state_w
                    print('ctrl', q_ctrl.cpu().tolist())
                    print('x', _root_states.cpu().tolist())
                    print('q', _dof_pos.cpu().tolist())
                    print('qd', _dof_vel.cpu().tolist())
                    print('tau', tau.cpu().tolist())
            else:
                if ctrl_inds_pos:
                    robot.set_joint_position_target(_action_pos)
                if ctrl_inds_vel:
                    robot.set_joint_velocity_target(_action_vel)
            if ctrl_inds_tau:
                tau = torch.clip(_action_tau, -dof_torque_limits, dof_torque_limits)
                robot.set_joint_effort_target(tau)
            robot.write_data_to_sim()
            sim.step(render=render)
            # sim.render()
            robot.update(dt=sim_dt)

        _dof_pos, _dof_vel = robot.data.joint_pos, robot.data.joint_vel
        # _root_states = robot.data.root_state_w
        _root_pose = robot.data._root_physx_view.get_root_transforms()
        _root_vel = robot.data._root_physx_view.get_root_velocities()
        # print(_root_pose)

        _rpy = get_euler_xyz(_root_pose[..., 3:7])
        dof_pos[:] = _dof_pos.flatten().cpu().numpy()
        dof_vel[:] = _dof_vel.flatten().cpu().numpy()
        root_states[:7] = _root_pose.flatten().cpu().numpy()
        root_states[7:] = _root_vel.flatten().cpu().numpy()
        imu[0:3] = _rpy.flatten().cpu().numpy()

    def close_fn():
        if not sim.has_gui():
            sim.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()
        simulation_app.close()

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
