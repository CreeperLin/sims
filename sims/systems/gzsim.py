# $ gz topic -l
# /clock
# /gazebo/resource_paths
# /stats
# /world/default/clock
# /world/default/dynamic_pose/info
# /world/default/pose/info
# /world/default/scene/deletion
# /world/default/scene/info
# /world/default/state
# /world/default/stats
# /world/default/light_config
# /world/default/material_color

# <physics name="1ms" type="ode">
#   <max_step_size>{sim_dt}</max_step_size>
#   <real_time_factor>1.0</real_time_factor>
# </physics>

# <gui fullscreen='0'>
# <camera name='user_camera'>
# <pose frame=''>15.4065 -7.32713 9.3415 -0 0.531643 2.6842</pose>
# <view_controller>orbit</view_controller>
# <projection_type>perspective</projection_type>
# </camera>
# </gui>

_default_world_str = '''
<?xml version='1.0'?>
<sdf version="1.6">
  <world name="default">
    <gravity>0 0 -9.81</gravity>
    <physics name="default_physics" default="0" type="ode">
      <ode>
        <solver>
          <iters>200</iters>
          <sor>1.3</sor>
          <thread_position_correction>1</thread_position_correction>
        </solver>
      </ode>
      <max_step_size>{sim_dt}</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <plugin
      filename="gz-sim-physics-system"
      name="gz::sim::systems::Physics">
    </plugin>
    <plugin
      filename="gz-sim-user-commands-system"
      name="gz::sim::systems::UserCommands">
    </plugin>
    <plugin
      filename="gz-sim-scene-broadcaster-system"
      name="gz::sim::systems::SceneBroadcaster">
    </plugin>
    <plugin
      filename="gz-sim-contact-system"
      name="gz::sim::systems::Contact">
    </plugin>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    {ground_model_str}
  </world>
</sdf>
'''

_default_ground_model_str = '''
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>
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
    clip_q_ctrl=True,
    headless=True,
    world=None,
):
    import os
    import tempfile
    import numpy as np
    # from gz.sim8 import TestFixture, Joint, Model, World, world_entity
    # from gz.common5 import set_verbosity
    # import sdformat14
    os.system("pkill -f -9 'gz sim'")
    from gz.sim9 import TestFixture, Joint, Model, World, world_entity, Link
    from gz.common6 import set_verbosity
    from gz.math8 import Pose3d, Vector3d
    import sdformat15 as sdformat
    from sims.utils import get_ctrl_inds, dict2list, list2slice

    np_dtype = np.float32
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
    _init_pose = Pose3d(*init_pos, 0, 0, 0)

    world_sdf = sdformat.Root()
    if world is None:
        world_str = _default_world_str
        ground_model_str = '' if fix_base_link else _default_ground_model_str
        world_str = world_str.format(
            ground_model_str=ground_model_str,
            sim_dt=sim_dt,
        )
        world_sdf.load_sdf_string(world_str)
    elif isinstance(world, str):
        if os.path.exists(world):
            world_sdf.load(world)
        else:
            world_sdf.load_sdf_string(world_str)
    else:
        world_sdf = world
    if urdf_path is not None:
        urdf = sdformat.Root()
        urdf.load(urdf_path)
        model = urdf.model()
        model_name = model.name()
        world_sdf.world_by_index(0).add_model(model)
        model = world_sdf.world_by_index(0).model_by_name(model_name)
    else:
        model = world_sdf.world_by_index(0).model_by_index(0)
    # print(world_sdf.to_string())
    assert model is not None
    model_name = model.name()

    print('model_name', model_name)
    print(model.joint_count())
    joints = [model.joint_by_index(i) for i in range(model.joint_count())]
    dof_names = [j.name() for j in joints]
    print('dof_names', dof_names)

    num_dofs = len(dof_names)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
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

    if fix_base_link:
        # world_link = sdformat.Link()
        # world_link.set_name('world')
        # model.add_link(world_link)
        j = sdformat.Joint()
        j.set_type(sdformat.JointType(sdformat.JointType.FIXED))
        j.set_name('world_fixed')
        j.set_parent_name('world')
        child_name = model.link_by_index(0).name()
        j.set_child_name(child_name)
        j.set_raw_pose(_init_pose)
        axis = sdformat.JointAxis()
        axis.set_xyz(Vector3d(0, 0, 1))
        j.set_axis(0, axis)
        model.add_joint(j)

    add_joint_position_controller = True
    # add_joint_position_controller = False
    if add_joint_position_controller:
        for n in dof_names:
            name = 'gz::sim::systems::JointPositionController'
            filename = 'gz-sim-joint-position-controller-system'
            content = '''
              <joint_name>{joint_name}</joint_name>
              <p_gain>200.</p_gain>
              <i_gain>0.</i_gain>
              <d_gain>20.</d_gain>
              <i_max>1</i_max>
              <i_min>-1</i_min>
              <cmd_max>1000</cmd_max>
              <cmd_min>-1000</cmd_min>
              <use_velocity_commands>true</use_velocity_commands>
            '''.format(joint_name=n)
            plugin = sdformat.Plugin(filename, name, content)
            model.add_plugin(plugin)

    add_joint_controller = False
    add_joint_controller = True
    if add_joint_controller:
        for n in dof_names:
            name = 'gz::sim::systems::JointController'
            filename = 'gz-sim-joint-controller-system'
            content = '''
              <joint_name>{joint_name}</joint_name>
              <p_gain>200.</p_gain>
              <i_gain>0.</i_gain>
              <d_gain>20.</d_gain>
              <use_force_commands>true</use_force_commands>
            '''.format(joint_name=n)
            plugin = sdformat.Plugin(filename, name, content)
            model.add_plugin(plugin)

    world_str = world_sdf.to_string()
    world_file = tempfile.NamedTemporaryFile(mode='w', delete=True)
    world_file.write(world_str)
    world_file.flush()

    with open('test.sdf', 'w') as f:
        f.write(world_str)

    world_path = world_file.name
    # world_file.write('<?xml version="1.0"?>'+world_sdf.to_string())
    set_verbosity(99)
    # file_path = os.path.dirname(os.path.realpath(__file__))
    # sim_dt = 0.001
    fixture = TestFixture(world_path)
    # num_iters = 1
    num_iters = int(dt // sim_dt)
    print('num_iters', num_iters)
    _world = None
    _model = None
    _joints = None
    _link = None

    _last_dof_pos = np.zeros(num_dofs, dtype=np_dtype)

    def on_step(_info, _ecm):
        nonlocal _world, _model, _joints, _link
        nonlocal reset
        nonlocal _last_dof_pos
        if _world is None:
            world_e = world_entity(_ecm)
            _world = World(world_e)
            _model = Model(_world.model_by_name(_ecm, model_name))
            print(_model.link_count(_ecm))
            print(_model.canonical_link(_ecm))
            _link = Link(_model.canonical_link(_ecm))
            print('_link', _link.name(_ecm))
            _joints = [Joint(_model.joint_by_name(_ecm, j)) for j in dof_names]
        if reset:
            _link.set_linear_velocity(_ecm, Vector3d(*init_lin_vel))
            _link.set_angular_velocity(_ecm, Vector3d(*init_ang_vel))
            _model.set_world_pose_cmd(_ecm, _init_pose)
            for i, j in enumerate(_joints):
                j.reset_position(_ecm, [default_dof_pos[i]])
                j.reset_velocity(_ecm, [0.])
                j.set_force(_ecm, [0.])
                assert j.valid(_ecm)
                q = j.position(_ecm)
                qd = j.velocity(_ecm)
                print(j.name(_ecm), q, qd)
            _last_dof_pos[:] = default_dof_pos
            dof_pos[:] = default_dof_pos
            dof_vel[:] = 0.
            reset = False
            return
        # for j in _joints:
        # assert j.valid(_ecm)
        # dof_pos = j.position(_ecm)
        # dof_vel = j.velocity(_ecm)
        # j.set_force(_ecm, [0.])
        # if dof_pos is not None:
        # print(dof_pos, dof_vel)
        iters = _info.iterations
        if iters % num_iters == 0:
            # print('iters', iters)
            # print('dof_pos', dof_pos.tolist())
            # print('dof_vel', dof_vel)
            # print(action_pos.shape, dof_pos.shape, dof_vel.shape)
            for i, j in enumerate(_joints):
                assert j.valid(_ecm)
                q = j.position(_ecm)
                qd = j.velocity(_ecm)
                dof_pos[i] = q[0]
                if qd is not None:
                    dof_vel[i] = qd[0]
            if qd is None:
                dof_vel[:] = (dof_pos - _last_dof_pos) / dt
                _last_dof_pos[:] = dof_pos
            if compute_torque:
                tau = kps * (action_pos - dof_pos) + kds * (-dof_vel)
                # print(tau.tolist())
                for i, j in enumerate(_joints):
                    assert j.valid(_ecm)
                    j.set_force(_ecm, [tau[i]])
            else:
                for i, j in enumerate(_joints):
                    assert j.valid(_ecm)
                    j.reset_position(_ecm, [action_pos[i]])
            pose = _link.world_pose(_ecm)
            # print('pose', pose)
            root_states[0:3] = [pose.x(), pose.y(), pose.z()]
            ang_vel = _link.world_angular_velocity(_ecm)
            if ang_vel:
                ang_vel = [ang_vel.x(), ang_vel.y().ang_vel.z()]
                root_states[10:13] = ang_vel
            lin_vel = _link.world_linear_velocity(_ecm)
            if lin_vel:
                lin_vel = [lin_vel.x(), lin_vel.y().lin_vel.z()]
                root_states[7:10] = lin_vel
            imu[0] = pose.roll()
            imu[1] = pose.pitch()
            imu[2] = pose.yaw()

    def on_post_update_cb(_info, _ecm):
        pass
        # print('on_post_update_cb')
        # print(_info.sim_time, _info.dt, _info.paused, _info.iterations)
        on_step(_info, _ecm)

    def on_pre_update_cb(_info, _ecm):
        # print('on_pre_update_cb')
        # print(_info.sim_time, _info.paused, _info.iterations)
        pass

    def on_update_cb(_info, _ecm):
        # print('on_update_cb')
        pass
        # print(_info.sim_time, _info.dt, _info.paused, _info.iterations)
        # on_step(_info, _ecm)

    fixture.on_post_update(on_post_update_cb)
    fixture.on_update(on_update_cb)
    fixture.on_pre_update(on_pre_update_cb)
    fixture.finalize()
    server = fixture.server()
    # server.run(True, 2, False)
    # server.run(False, 100000, False)
    reset = False

    def step_fn():
        # blocking=True; iterations=1; paused=False
        # blocking=True; iterations=1; paused=True
        blocking = True
        iterations = num_iters
        paused = False
        # blocking=True; iterations=num_iters; paused=True
        # blocking=False; iterations=1; paused=False
        # blocking=True; iterations=2; paused=True
        server.run(blocking, iterations, paused)
        # print(server.is_running())

    def close_fn():
        print('close')
        world_file.close()
        os.system("pkill -f -9 'gz sim'")
        # server.stop()
        # os.remove(world_file)
        pass

    def reset_fn():
        nonlocal reset
        print('reset')
        reset = True
        blocking = True
        iterations = 1
        paused = False
        server.run(blocking, iterations, paused)

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
