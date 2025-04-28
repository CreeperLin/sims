_default_world_str = '''#VRML_SIM R2023b utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2023b/projects/objects/floors/protos/RectangleArena.proto"

WorldInfo {
<WorldInfoStr>
CFM 1e-07
ERP 0.8
}
Viewpoint {
position -5 0 1
}
TexturedBackground {
}
TexturedBackgroundLight {
}
RectangleArena {
  translation 0.0 0.0 0
  floorSize 100 100
}
'''


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
    proto_path=None,
    world_path=None,
    clip_q_ctrl=True,
    run_webots=True,
    world_info_params=None,
    headless=False,
    robot_name='URDFRobot',
):
    import os
    import numpy as np
    from sims.utils import get_ctrl_inds, dict2list, list2slice

    np_dtype = np.float32
    sim_dt = dt / decimation
    sim_dt_ms = int(1000 * sim_dt)

    init_pos = [0, 0, 1]
    init_rot = [0, 0, 0, 1]
    init_lin_vel = [0, 0, 0]
    init_ang_vel = [0, 0, 0]
    if default_root_states is not None:
        init_pos = default_root_states[:3]
        init_rot = default_root_states[3:7]
        init_lin_vel = default_root_states[7:10]
        init_ang_vel = default_root_states[10:13]

    def mat2rpy(m):
        pi = np.pi
        m00 = m[..., 0, 0]
        m22 = m[..., 2, 2]
        m21 = m[..., 2, 1]
        m02 = m[..., 0, 2]
        m20 = m[..., 2, 0]
        m01 = m[..., 0, 1]
        m10 = m[..., 1, 0]
        if abs(m20 - 1.0) < 1.0e-15:
            a = 0.0
            b = -pi / 2.0
            c = np.arctan2(-m01, -m02)
        elif abs(m20 + 1.0) < 1.0e-15:
            a = 0.0
            b = pi / 2.0
            c = -np.arctan2(m01, m02)
        else:
            a = np.arctan2(m21, m22)
            c = np.arctan2(m10, m00)
            cosC = np.cos(c)
            sinC = np.sin(c)
        if abs(cosC) > abs(sinC):
            b = np.arctan2(-m20, m00 / cosC)
        else:
            b = np.arctan2(-m20, m10 / sinC)
        rpy = np.stack([a, b, c], axis=-1)
        return rpy

    import tempfile
    # world_file = tempfile.NamedTemporaryFile(mode='w', delete=True, dir='.', suffix='.wbt')
    world_file = tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.wbt')
    new_world_path = world_file.name
    new_world_dir = os.path.dirname(new_world_path)

    node_str = ''
    if proto_path is None and urdf_path is not None:
        print('urdf_path', urdf_path)
        from urdf2webots.importer import convertUrdfContent
        urdf_dir = os.path.dirname(urdf_path)
        with open(urdf_path, 'r') as f:
            urdf_str = f.read()
        node_str = convertUrdfContent(
            input=urdf_str,
            robotName=robot_name,
            initTranslation=' '.join(map(str, init_pos)),
            # initRotation=' '.join(map(str, init_rot)),
            relativePathPrefix=urdf_dir,
            outputDirectory=new_world_dir,
        )
        node_str = node_str[:-2] + '  supervisor TRUE\n}\n'
    elif proto_path is not None:
        with open(proto_path, 'r') as f:
            node_str = f.read()

    if world_path is None:
        world_str = _default_world_str
    else:
        with open(world_path, 'r') as f:
            world_str = f.read()

    world_info_params = {} if world_info_params is None else world_info_params
    world_info_params['basicTimeStep'] = sim_dt_ms
    world_info_str = ''
    if world_path is None and world_info_params:
        import json
        world_info_str = json.dumps(world_info_params, separators=('\n', ' '))[1:-1]
        world_info_str = world_info_str.replace('"', '')
        print(world_info_str)
    world_str = world_str.replace('<WorldInfoStr>', world_info_str)
    if node_str:
        world_str += node_str
    if fix_base_link:
        idx = world_str.index('\n  physics Physics')
        idx2 = world_str[idx:].index('}')
        world_str = world_str[:idx] + world_str[idx + idx2 + 1:]
    with open(new_world_path, 'w') as f:
        f.write(world_str)
    with open('test.wbt', 'w') as f:
        f.write(world_str)

    proc = None
    WEBOTS_HOME = os.environ['WEBOTS_HOME']
    if run_webots:
        import time
        os.system('killall -9 webots')
        os.system('killall -9 webots-bin')
        time.sleep(1)
        webots_path = os.path.join(WEBOTS_HOME, 'webots')
        env = os.environ.copy()
        # env['WEBOTS_SAFE_MODE'] = 'true'
        import subprocess
        empty_world_path = os.path.join(WEBOTS_HOME, 'resources/projects/worlds/empty.wbt')
        proc = subprocess.Popen(args=[
            webots_path,
            '--no-rendering',
            '--minimize',
            empty_world_path,
        ])
        time.sleep(3)
        proc.terminate()
        os.system('killall -9 webots')
        os.system('killall -9 webots-bin')
        opts = [
            '--extern-urls',
        ]
        if headless:
            opts.append('--no-rendering')
        verbose = False
        if verbose:
            opts.extend([
                '--stdout',
                '--stderr',
            ])
        args = [
            webots_path,
            *opts,
            new_world_path,
        ]
        print(args)
        # env = None
        proc = subprocess.Popen(args, env=env)
        time.sleep(3)
        if proc.poll() is not None:
            raise RuntimeError('webots failed')

    # os.remove(new_world_path)

    import sys
    sys.path.append(os.path.join(WEBOTS_HOME, 'lib', 'controller', 'python'))
    from controller import Supervisor, Motor, PositionSensor
    robot = Supervisor()
    print('number_of_devices', robot.number_of_devices)
    # print('devices', robot.devices)
    devices = robot.devices
    motors = [v for k, v in devices.items() if isinstance(v, Motor)]
    pos_sensors = [v for k, v in devices.items() if isinstance(v, PositionSensor)]
    dof_names = [k for k, v in devices.items() if isinstance(v, Motor)]
    imus = []
    gyros = []
    accelerometers = []

    # for motor in motors:
    #     motor.enableForceFeedback(sim_dt_ms)
    #     motor.enableTorqueFeedback(sim_dt_ms)
    for sensor in pos_sensors:
        sensor.enable(sim_dt_ms)
    for imu in imus:
        imu.enable(sim_dt_ms)
    for gyro in gyros:
        gyro.enable(sim_dt_ms)
    for accelerater in accelerometers:
        accelerater.enable(sim_dt_ms)
    print('simulation_mode', robot.simulation_mode)
    # robot.simulation_mode = Supervisor.SIMULATION_MODE_PAUSE
    robot.simulation_mode = Supervisor.SIMULATION_MODE_FAST
    print('simulation_mode', robot.simulation_mode)
    print('mode', robot.mode)
    root_node = robot.getRoot()
    children_field = root_node.getField("children")
    n = children_field.getCount()
    print(f'This world contains {n} nodes:')
    # for i in range(n):
    # node = children_field.getMFNode(i)
    # print(f'-> {node.getTypeName()}')
    info_node = children_field.getMFNode(0)
    field = info_node.getField('gravity')
    gravity = field.getSFFloat()
    print(f'WorldInfo.gravity = {gravity}\n')
    timestep = int(robot.getBasicTimeStep())
    assert timestep == sim_dt_ms
    node = children_field.getMFNode(-1)
    body = node
    dof_pos_max = np.array([m.getMaxPosition() for m in motors])
    dof_pos_min = np.array([m.getMinPosition() for m in motors])
    dof_vel_max = np.array([m.getMaxVelocity() for m in motors])
    dof_torque_max = np.array([m.getMaxTorque() for m in motors])
    dof_force_max = np.array([m.getMaxForce() for m in motors])
    print('dof_pos_max', dof_pos_max)
    print('dof_pos_min', dof_pos_min)
    print('dof_vel_max', dof_vel_max)
    print('dof_torque_max', dof_torque_max)
    print('dof_force_max', dof_force_max)

    num_dofs = len(dof_names)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    _ctrl_inds_pos, _ctrl_inds_vel, _ctrl_inds_tau = ctrl_inds_all
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

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
    dof_tau = np.zeros(num_dofs, dtype=np_dtype)
    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    n_window = 2
    wnd_q = np.zeros((n_window, num_dofs), dtype=np_dtype)
    q = wnd_q[0]
    qd = None
    if compute_torque:
        for i, m in enumerate(motors):
            m.setControlPID(kps[i], 0, kds[i])

    def update_q_qd():
        nonlocal qd
        wnd_q[1:] = wnd_q[:-1]
        q_cur = wnd_q[0]
        for i, s in enumerate(pos_sensors):
            v = s.getValue()
            q_cur[i] = 0 if np.isnan(v) else v
        dq = wnd_q[:-1] - wnd_q[1:]
        qd = np.mean(dq / sim_dt, axis=0)
        # print('q_cur', q_cur)
        # print('q', q)
        # print('qd', qd)

    def reset_fn():
        wnd_q[:] = 0
        robot.simulationReset()
        robot.simulationResetPhysics()
        for i in range(num_dofs):
            # body.setJointPosition(default_dof_pos[i])
            motors[i].setPosition(default_dof_pos[i])
        robot.step()
        update_q_qd()
        qd[:] = 0
        print('reset')

    def step_fn():
        for k in range(decimation):
            tau = None
            if ctrl_inds_pos:
                ctrl = action_pos
                if clip_q_ctrl:
                    ctrl = np.clip(ctrl, dof_pos_min, dof_pos_max)
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                tau = np.zeros(num_dofs, dtype=np_dtype) if not ctrl_inds_tau else action_tau
                if ctrl_inds_pos:
                    tau_pos = kps * (ctrl - q) + kds * (-qd)
                    tau[ctrl_inds_pos] = tau_pos[ctrl_inds_pos]
                if ctrl_inds_vel:
                    tau_vel = kds * (action_vel - qd)
                    tau[ctrl_inds_vel] = tau_vel[ctrl_inds_vel]
            if not compute_torque and ctrl_inds_pos:
                for i in _ctrl_inds_pos:
                    motors[i].setPosition(ctrl[i])
            if not compute_torque and ctrl_inds_vel:
                for i in _ctrl_inds_vel:
                    motors[i].setVelocity(action_vel[i])
            if ctrl_inds_tau:
                tau = action_tau if tau is None else tau
            if tau is not None:
                if torque_limits is not None:
                    tau = np.clip(tau, -torque_limits, torque_limits)
                inds = _ctrl_inds_pos + _ctrl_inds_vel + _ctrl_inds_tau
                for i in inds:
                    motors[i].setTorque(tau[i])
            # t0 = time.perf_counter()
            robot.step()
            # print(i, time.perf_counter() - t0)
            update_q_qd()
        pos = np.array(body.getPosition())
        base_vel = np.array(body.getVelocity())
        lin_vel = base_vel[0:3]
        body_rot_w = body.getOrientation()
        body_rot_mat = np.array([body_rot_w[0:3], body_rot_w[3:6], body_rot_w[6:9]])
        _rpy = mat2rpy(body_rot_mat)
        ang_vel = base_vel[3:6]
        # print('_rpy', _rpy)
        # print('q', q)
        # print('qd', qd)
        dof_pos[:] = q
        dof_vel[:] = qd
        # rot = np.array(rot)
        imu[0:3] = _rpy
        # lin_vel, ang_vel
        root_states[0:3] = pos
        # root_states[3:7] = rot
        root_states[7:10] = lin_vel
        root_states[10:13] = ang_vel

    def close_fn():
        robot.simulationQuit(0)
        if proc is not None:
            proc.kill()
            proc.terminate()

    return {
        'dt': dt,
        # 'dt': False,
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
