def init_fn(
    urdf_path=None,
    xml_path=None,
    dt=0.02,
    nstep=10,
    actuation_spec=None,
    Kp=None,
    Kd=None,
    torque_limits=None,
    ctrl_joints_pos='*',
    ctrl_joints_vel=None,
    ctrl_joints_tau=None,
    collision_groups=None,
    default_dof_pos=None,
    default_root_states=None,
    decimation=1,
    compute_torque=False,
    fix_base_link=False,
    headless=False,
    use_sensor=False,
    use_mjx=False,
    model_opt=None,
    model_params=None,
    verbose=False,
    device=None,
    sensor_quat='orientation',
    sensor_ang_vel='angular-velocity',
    sensor_lin_vel='velocimeter',
    sensor_lin_acc='accelerometer',
    save_xml=False,
    no_collision_mesh=False,
):

    init_pos = [0, 0, 1]
    init_rot = [0, 0, 0, 1]
    init_lin_vel = [0, 0, 0]
    init_ang_vel = [0, 0, 0]
    if default_root_states is not None:
        init_pos = default_root_states[:3]
        init_rot = default_root_states[3:7]
        init_lin_vel = default_root_states[7:10]
        init_ang_vel = default_root_states[10:13]

    if use_mjx:
        import jax
        import jax.numpy as jp
        from mujoco import mjx
        if device is not None:
            jax.config.update("jax_default_device", device)
    import mujoco
    import mujoco.viewer
    import numpy as np
    from sims.utils import update_array, dict2list, update_obj, quat2rpy_np3, get_ctrl_inds, list2slice

    np_dtype = np.float32

    cylinder2box = True if use_mjx else False
    if xml_path is None and urdf_path is not None:
        import tempfile
        try:
            from urdf2mjcf.convert import convert_urdf_to_mjcf
            xml_file = tempfile.NamedTemporaryFile(mode='w', delete=True)
            xml_path = xml_file.name
            actuator_type = 'motor' if compute_torque else 'position'
            convert_urdf_to_mjcf(
                urdf_path=urdf_path,
                mjcf_path=xml_path,
                no_collision_mesh=no_collision_mesh,
                # copy_meshes=args.copy_meshes,
                # camera_distance=args.camera_distance,
                # camera_height_offset=args.camera_height_offset,
                copy_meshes=True,
                no_frc_limit=True,
                # default_position=default_position,
                fix_base_link=fix_base_link,
                cylinder2box=cylinder2box,
                use_sensor=use_sensor,
                actuator_type=actuator_type,
            )
            print('convert_urdf_to_mjcf', xml_path, fix_base_link)
            if save_xml:
                save_path = 'test.xml' if save_xml is True else save_xml
                import os
                os.system(f'cp {xml_path} {save_path}')
        except:
            print('convert_urdf_to_mjcf failed')
            import traceback
            traceback.print_exc()
            import xml.etree.ElementTree as ET
            with open(urdf_path, 'r') as f:
                urdf_string = f.read()
            root = ET.fromstring(urdf_string)
            xml_file = tempfile.NamedTemporaryFile(mode='w', delete=True)
            xml_file.write(ET.tostring(root, encoding='unicode'))
            xml_file.flush()
            xml_path = xml_file.name
            xml_path = urdf_path
    model = mujoco.MjModel.from_xml_path(xml_path)
    sim_dt = dt / nstep / decimation
    print('sim_dt', sim_dt)
    model.opt.timestep = sim_dt
    if model_opt is not None:
        update_obj(model.opt, model_opt)
    if model_params is not None:
        update_obj(model, model_params)
    print('model.opt', model.opt)
    print('model.qpos0', len(model.qpos0), model.qpos0)
    num_dofs = len(model.name_actuatoradr)
    print('model.geom_contype', len(model.geom_contype), model.geom_contype)
    print('num_dofs', num_dofs)
    if actuation_spec is None:
        dof_names = [model.actuator(i).name for i in range(0, num_dofs)]
    else:
        dof_names = []
        for name in actuation_spec:
            # print(model.actuator(name))
            dof_names.append(name)
        print(model.actuator(name))
    print('dof_names', dof_names)
    num_dofs = len(dof_names)
    print('num_dofs', num_dofs)

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    print('model.actuator_gainprm', model.actuator_gainprm[:, :3].tolist())
    print('model.actuator_biasprm', model.actuator_biasprm[:, :3].tolist())
    print('model.actuator_forcerange', model.actuator_forcerange.tolist())
    kps = np.zeros(num_dofs)
    update_array(kps, Kp, dof_names)
    print('kps', kps)
    kds = np.zeros(num_dofs)
    update_array(kds, Kd, dof_names)
    print('kds', kds)
    tau_limits = np.zeros(num_dofs)
    tau_limits[:] = model.actuator_forcerange[:, 1]
    update_array(tau_limits, torque_limits, dof_names)
    torque_limits = tau_limits
    # torque_limits = None
    # print('torque_limits', torque_limits)
    if torque_limits is not None and np.max(torque_limits) < 1e-3:
        torque_limits = None
    if torque_limits is not None:
        model.actuator_forcerange[:, 0] = -torque_limits
        model.actuator_forcerange[:, 1] = torque_limits
    if not compute_torque:
        model.actuator_gainprm[:, 0] = kps
        model.actuator_biasprm[:, 1] = -kps
        model.actuator_biasprm[:, 2] = -kds
    else:
        model.actuator_gainprm[:] = 0.
        model.actuator_biasprm[:] = 0.
        model.actuator_gainprm[:, 0] = 1.
    print('model.actuator_gainprm', model.actuator_gainprm[:, :3].tolist())
    print('model.actuator_biasprm', model.actuator_biasprm[:, :3].tolist())
    print('model.actuator_forcerange', model.actuator_forcerange.tolist())
    print('torque_limits', torque_limits)

    if collision_groups is not None:
        for name, geom_names in collision_groups:
            col_group = list()
            for geom_name in geom_names:
                mj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                assert mj_id != -1, f"geom \"{geom_name}\" not found! Can't be used for collision-checking."
                col_group.append(mj_id)
            collision_groups[name] = set(col_group)

    ddp_inds = None
    def_dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    if isinstance(default_dof_pos, dict):
        ddp_inds, default_dof_pos = dict2list(default_dof_pos, dof_names, default_dof_pos.pop('_default', None))
    if default_dof_pos is not None:
        def_dof_pos[ddp_inds] = default_dof_pos
    default_dof_pos = def_dof_pos
    action_pos = np.zeros(num_dofs, dtype=np_dtype)
    action_vel = np.zeros(num_dofs, dtype=np_dtype)
    action_tau = np.zeros(num_dofs, dtype=np_dtype)
    dof_pos = np.zeros(num_dofs, dtype=np_dtype)
    dof_vel = np.zeros(num_dofs, dtype=np_dtype)
    dof_tau = np.zeros(num_dofs, dtype=np_dtype)

    step_kwds = {}
    if nstep is not None:
        step_kwds['nstep'] = nstep
    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    if fix_base_link:
        init_q = np.zeros(num_dofs)
        init_qd = np.zeros(num_dofs)
    else:
        init_q = np.zeros(7 + num_dofs)
        init_qd = np.zeros(6 + num_dofs)
        init_q[:3] = init_pos
        init_q[3] = init_rot[3]
        init_q[4:7] = init_rot[:3]
        init_qd[0:3] = init_lin_vel
        init_qd[3:6] = init_ang_vel
    init_q[-num_dofs:] = default_dof_pos
    print('init_q', init_q.tolist())

    qpos = None
    qvel = None
    data = mujoco.MjData(model)
    if use_mjx:
        print('use_mjx', use_mjx)
        model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mjx_model = mjx.put_model(model)
        mjx_data = mjx.make_data(mjx_model)
    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.sync()

    def export_fn():
        if use_sensor:
            rot = data.sensor(sensor_quat).data
        else:
            rot = qpos[3:7]
        _rpy = quat2rpy_np3(rot, w_first=True)
        if use_sensor:
            ang_vel = data.sensor(sensor_ang_vel).data
        else:
            ang_vel = qvel[3:6]
        if use_sensor:
            lin_vel = data.sensor(sensor_lin_vel).data
        else:
            lin_vel = qvel[0:3]
        root_states[0:3] = qpos[0:3]
        root_states[3:6] = rot[1:4]
        root_states[6] = rot[0]
        root_states[10:13] = ang_vel
        imu[0:3] = _rpy
        if use_sensor:
            lin_acc = data.sensor(sensor_lin_acc).data
            imu[3:6] = lin_acc
            # print(lin_acc)
        imu[6:9] = ang_vel
        imu[9:12] = lin_vel
        dof_pos[:] = qpos[-num_dofs:]
        dof_vel[:] = qvel[-num_dofs:]

    if use_mjx:

        def _mjx_step(mjx_data, ctrl):
            ctrl = jp.array(ctrl)
            # ctrl = mjx_data.ctrl.set(ctrl)
            mjx_data = mjx_data.replace(ctrl=ctrl)
            mjx_data = mjx.step(mjx_model, mjx_data)
            return mjx_data

        _mjx_step = jax.jit(_mjx_step)
        _mjx_step(mjx_data, jp.zeros(num_dofs))

        def mjx_step(ctrl):
            nonlocal mjx_data, qpos, qvel
            mjx_data = _mjx_step(mjx_data, ctrl)
            # mjx.get_data_into(data, model, mjx_data)
            attrs = [
                'qpos',
                'qvel',
                'geom_xpos',
                'geom_xmat',
            ]
            for attr in attrs:
                v = getattr(data, attr)
                v[:] = getattr(mjx_data, attr).reshape(*v.shape)
            qpos = data.qpos
            qvel = data.qvel

        init_q = jp.array(init_q)
        init_qd = jp.array(init_qd)

        def reset_fn():
            nonlocal mjx_data, qpos, qvel
            mjx_data = mjx_data.replace(qpos=init_q, qvel=init_qd)
            qpos = np.asarray(mjx_data.qpos)
            qvel = np.asarray(mjx_data.qvel)
            export_fn()

        sim_step_fn = mjx_step
    else:

        def mj_step(ctrl):
            nonlocal qpos, qvel
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data, **step_kwds)
            qpos = data.qpos
            qvel = data.qvel

        def reset_fn():
            nonlocal qpos, qvel
            mujoco.mj_resetData(model, data)
            data.qpos[:] = init_q
            data.qvel[:] = init_qd
            qpos = data.qpos
            qvel = data.qvel
            export_fn()

        sim_step_fn = mj_step

    def step_fn():
        for i in range(decimation):
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                ctrl = np.zeros(num_dofs) if not ctrl_inds_tau else action_tau.copy()
                q = qpos[-num_dofs:]
                qd = qvel[-num_dofs:]
                if ctrl_inds_pos:
                    tau_pos = kps * (action_pos - q) + kds * (-qd)
                    ctrl[ctrl_inds_pos] = tau_pos[ctrl_inds_pos]
                if ctrl_inds_vel:
                    tau_vel = kds * (action_vel - qd)
                    ctrl[ctrl_inds_vel] = tau_vel[ctrl_inds_vel]
                if torque_limits is not None:
                    ctrl = np.clip(ctrl, -torque_limits, torque_limits)
                if verbose:
                    print('q_ctrl', action_pos.tolist())
                    print('q', q.tolist())
                    print('qd', qd.tolist())
                    print('ctrl', ctrl.tolist())
                dof_tau[:] = ctrl
            else:
                if ctrl_inds_pos:
                    ctrl = action_pos
                if ctrl_inds_vel:
                    ctrl = action_vel
            sim_step_fn(ctrl)
        # if not fix_base_link:
        export_fn()
        if viewer is not None:
            viewer.sync()

    def close_fn():
        pass

    return {
        'reset_fn': reset_fn,
        'step_fn': step_fn,
        'close_fn': close_fn,
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
