def init_fn(
    urdf_path=None,
    asset_root=None,
    plane_path='plane.urdf',
    dt=None,
    freq=60,
    Kp=None,
    Kd=None,
    torque_limits=None,
    render=True,
    headless=False,
    decimation=1,
    fixed_timestep=None,
    fix_base_link=False,
    urdf_flags=None,
    default_dof_pos=None,
    default_root_states=None,
    lateral_friction=None,
    spinning_friction=None,
    rolling_friction=None,
    ctrl_joints_pos='*',
    ctrl_joints_vel=None,
    ctrl_joints_tau=None,
    engine_params=None,
    compute_torque=False,
    set_gains=False,
    **kwds,
):
    import numpy as np
    import pybullet as p
    import pybullet_data
    from sims.utils import dict2list, list2slice, get_ctrl_inds
    from sims.utils import quat_rotate_inverse_np

    np_dtype = np.float32

    init_pos = [0, 0, 1]
    init_rot = [0, 0, 0, 1]
    init_lin_vel = [0, 0, 0]
    init_ang_vel = [0, 0, 0]
    if default_root_states is not None:
        init_pos = default_root_states[:3]
        init_rot = default_root_states[3:7]
        init_lin_vel = default_root_states[7:10]
        init_ang_vel = default_root_states[10:13]

    render = render and not headless
    mode = p.GUI if render else p.DIRECT
    p.connect(mode)
    if render:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        # p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
    data_path = pybullet_data.getDataPath()
    p.setAdditionalSearchPath(data_path)
    plane_params = dict(
        lateralFriction=lateral_friction,
        spinningFriction=spinning_friction,
        rollingFriction=rolling_friction,
    )
    plane_params = {k: v for k, v in plane_params.items() if v is not None}
    if plane_path is not None:
        plane = p.loadURDF(plane_path)
        if any(plane_params.values()):
            p.changeDynamics(
                plane,
                -1,
                **plane_params,
            )
    dt = 1 / freq if dt is None else dt
    freq = 1 / dt if freq is None else freq
    # fixed_timestep = 1 / 240
    decimation = int(dt // fixed_timestep) if decimation is None else decimation
    fixed_timestep = 1 / (freq * decimation) if fixed_timestep is None else fixed_timestep
    p.setPhysicsEngineParameter(fixedTimeStep=fixed_timestep)
    print('decimation', decimation)
    print('fixed_timestep', fixed_timestep)
    # p.setPhysicsEngineParameter(enableConeFriction=0)
    if engine_params is not None:
        p.setPhysicsEngineParameter(**engine_params)

    p.setGravity(0, 0, -9.81)
    urdfFlags = 0
    if urdf_flags is not None:
        for f in urdf_flags:
            urdfFlags |= getattr(p, f)
    if asset_root is not None:
        p.setAdditionalSearchPath(asset_root)
    robot = p.loadURDF(urdf_path, init_pos, init_rot, flags=urdfFlags, useFixedBase=fix_base_link)

    jointIds = []
    dof_names = []

    num_joints = p.getNumJoints(robot)
    for j in range(num_joints):
        p.changeDynamics(robot, j, linearDamping=0, angularDamping=0)
        info = p.getJointInfo(robot, j)
        jointName = info[1]
        jointType = info[2]
        if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
            jointName = jointName.decode() if isinstance(jointName, bytes) else jointName
            dof_names.append(jointName)
            jointIds.append(j)
            p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, targetPosition=0, force=0)
    num_dofs = len(dof_names)
    print('num_joints', num_joints)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    _ctrl_inds_pos, _ctrl_inds_vel, _ctrl_inds_tau = ctrl_inds_all
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    print('jointIds', jointIds)

    p.setRealTimeSimulation(0)

    pd_kwds = {}
    kps = np.zeros(num_dofs, dtype=np_dtype)
    kds = np.zeros(num_dofs, dtype=np_dtype)
    kp_inds = None
    if isinstance(Kp, dict):
        kp_inds, Kp = dict2list(Kp, dof_names, Kp.pop('_default', None))
    kd_inds = None
    if isinstance(Kd, dict):
        kd_inds, Kd = dict2list(Kd, dof_names, Kd.pop('_default', None))
    if Kp is not None:
        kps[kp_inds] = Kp
    if Kd is not None:
        kds[kd_inds] = Kd
    if not compute_torque and set_gains:
        pd_kwds['positionGains'] = kps
        pd_kwds['velocityGains'] = kds
    def_dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    ddp_inds = None
    if isinstance(default_dof_pos, dict):
        ddp_inds, default_dof_pos = dict2list(default_dof_pos, dof_names, default_dof_pos.pop('_default', None))
    if default_dof_pos is not None:
        def_dof_pos[ddp_inds] = default_dof_pos
    default_dof_pos = def_dof_pos
    if isinstance(torque_limits, dict):
        tl_inds, torque_limits = dict2list(torque_limits, dof_names, torque_limits.pop('_default', None))
        torque_limits = np.array(torque_limits)
        if tl_inds is not None:
            limits = np.zeros(num_dofs)
            limits[tl_inds] = torque_limits
            torque_limits = limits.copy()
    print('torque_limits', torque_limits)
    action_pos = np.zeros(num_dofs, dtype=np_dtype)
    action_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_tau = np.zeros(num_dofs, dtype=np_dtype)
    dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    dof_vel = np.zeros(num_dofs, dtype=np_dtype)
    dof_tau = np.zeros(num_dofs, dtype=np.float32)
    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    def reset_fn():
        p.resetBasePositionAndOrientation(robot, init_pos, init_rot)
        p.resetBaseVelocity(robot, init_lin_vel, init_ang_vel)
        for i, j in enumerate(jointIds):
            p.resetJointState(robot, j, default_dof_pos[i], targetVelocity=0)
            p.setJointMotorControl2(bodyIndex=robot,
                                    jointIndex=j,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=0,
                                    force=0)
        joints = p.getJointStates(robot, jointIds)
        dof_states = np.array([j[:2] for j in joints])
        dof_pos[:] = dof_states[:, 0]
        dof_vel[:] = dof_states[:, 1]

    def step_fn():
        for i in range(decimation):
            # p.stepSimulation()
            # continue
            tau = None
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                tau = np.zeros(num_dofs, dtype=np_dtype) if not ctrl_inds_tau else action_tau
                joints = p.getJointStates(robot, jointIds)
                dof_states = np.array([j[:2] for j in joints])
                q = dof_states[:, 0]
                qd = dof_states[:, 1]
                if ctrl_inds_pos:
                    tau_pos = Kp * (action_pos - q) + Kd * (-qd)
                    tau[ctrl_inds_pos] = tau_pos[ctrl_inds_pos]
                if ctrl_inds_vel:
                    tau_vel = Kd * (action_vel - qd)
                    tau[ctrl_inds_vel] = tau_vel[ctrl_inds_vel]
            if not compute_torque and ctrl_inds_pos:
                p.setJointMotorControlArray(
                    robot,
                    _ctrl_inds_pos,
                    p.POSITION_CONTROL,
                    targetPositions=action_pos[ctrl_inds_pos],
                    targetVelocities=action_vel[ctrl_inds_pos],
                    **pd_kwds,
                )
            if not compute_torque and ctrl_inds_vel:
                p.setJointMotorControlArray(
                    robot,
                    _ctrl_inds_vel,
                    p.VELOCITY_CONTROL,
                    targetVelocities=action_vel[ctrl_inds_vel],
                    **pd_kwds,
                )
            if ctrl_inds_tau:
                tau = action_tau if tau is None else tau
            if tau is not None:
                if torque_limits is not None:
                    tau = np.clip(tau, -torque_limits, torque_limits)
                inds = _ctrl_inds_pos + _ctrl_inds_vel + _ctrl_inds_tau
                p.setJointMotorControlArray(
                    robot,
                    inds,
                    p.TORQUE_CONTROL,
                    forces=tau,
                )
            p.stepSimulation()
        joints = p.getJointStates(robot, jointIds)
        dof_states = np.array([j[:2] for j in joints])
        dof_pos[:] = dof_states[:, 0]
        dof_vel[:] = dof_states[:, 1]
        pos, rot = p.getBasePositionAndOrientation(robot)
        _rpy = p.getEulerFromQuaternion(rot)
        rot = np.array(rot)
        imu[0:3] = _rpy
        # print('pos', pos)
        # print('rot', rot)
        # print('rpy', rpy)
        # joints = p.getJointStates(robot, jointIds)
        # joints = np.array([j[:2] for j in joints])
        # print('joints', joints, joints.shape)
        lin_vel, ang_vel = p.getBaseVelocity(robot)
        # print(rot, lin_vel, ang_vel)
        base_lin_vel = quat_rotate_inverse_np(rot, np.array(lin_vel))
        base_ang_vel = quat_rotate_inverse_np(rot, np.array(ang_vel))
        imu[6:9] = base_ang_vel
        imu[9:12] = base_lin_vel
        root_states[0:3] = pos
        root_states[3:7] = rot
        root_states[7:10] = lin_vel
        root_states[10:13] = ang_vel

    def close_fn():
        p.resetSimulation()
        p.disconnect()

    return {
        'robot': robot,
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
