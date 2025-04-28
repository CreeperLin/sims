def init_fn(
    urdf_path=None,
    xml_path=None,
    dt=None,
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
    headless=True,
    backend='generalized',
    # backend='spring',
    # backend='positional',
    n_frames=1,
    html_path='viz.html',
    # renderer=None,
    renderer='mujoco',
    device='cpu',
    model_opt=None,
    model_params=None,
):
    import numpy as np
    from sims.utils import get_ctrl_inds, dict2list, list2slice, update_obj
    from sims.utils import quat2rpy_np3, quat2mat_np, quat_mul_np

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

    sim_dt = dt / n_frames / decimation
    print('sim_dt', sim_dt)

    if xml_path is None and urdf_path is not None:
        import tempfile
        from urdf2mjcf.convert import convert_urdf_to_mjcf
        xml_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        xml_path = xml_file.name
        convert_urdf_to_mjcf(
            urdf_path=urdf_path,
            mjcf_path=xml_path,
            # no_collision_mesh=args.no_collision_mesh,
            # copy_meshes=args.copy_meshes,
            # camera_distance=args.camera_distance,
            # camera_height_offset=args.camera_height_offset,
            no_frc_limit=True,
            # default_position=default_position,
            fix_base_link=fix_base_link,
            cylinder2box=True,
            # use_sensor=use_sensor,
        )
        print('convert_urdf_to_mjcf', xml_path, fix_base_link)
        import os
        os.system(f'cp {xml_path} test.xml')

    import brax
    from brax.io import mjcf
    from brax import base
    # from brax import actuator
    import jax
    import jax.numpy as jp

    import json
    from brax.io import html
    from brax.io import json as brax_json
    from brax.base import System, State
    from typing import List, Optional, Union

    def fmt_obj(obj, decimals=3):
        if isinstance(obj, dict):
            return {k: fmt_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [fmt_obj(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, decimals)
        return obj

    def lite_render(
        sys: System,
        states: List[State],
        height: Union[int, str] = 480,
        colab: bool = True,
        base_url: Optional[str] = None,
    ):
        j = brax_json.dumps(sys, states)
        d = json.loads(j)
        d = fmt_obj(d)
        j = json.dumps(d, separators=(',', ':'))
        return html.render_from_json(j, height, colab, base_url)

    rollout = []

    model = mjcf.load_mjmodel(xml_path)
    if model_opt is not None:
        update_obj(model.opt, model_opt)
    if model_params is not None:
        update_obj(model, model_params)
    model.opt.timestep = sim_dt
    import mujoco
    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    print('opt', model.opt)
    model.opt.cone = 0
    print('name_numericadr', model.name_numericadr)
    print('name_tupleadr', model.name_tupleadr)
    print('geom_priority', model.geom_priority)
    print('geom_type', model.geom_type)
    print('geom_bodyid', model.geom_bodyid)
    print('name_tupleadr', model.name_tupleadr)
    print('geom_contype', model.geom_contype)
    print('geom_conaffinity', model.geom_conaffinity)
    print('jnt_type', model.jnt_type)
    print('jnt_bodyid', model.jnt_bodyid)
    print('jnt_pos', model.jnt_pos)
    print('jnt_axis', model.jnt_axis)
    print('jnt_range', model.jnt_range)
    print('jnt_stiffness', model.jnt_stiffness)
    print('body_pos', model.body_pos)
    print('gravity', model.opt.gravity)
    print('qpos0', model.qpos0)
    print('actuator_gear', model.actuator_gear[:, 0])
    print('actuator_gainprm', model.actuator_gainprm[:, 0])
    print('actuator_bias_q', model.actuator_biasprm[:, 1])
    print('actuator_bias_qd', model.actuator_biasprm[:, 2])
    print('actuator_ctrlrange', model.actuator_ctrlrange)
    print('actuator_ctrllimited', model.actuator_ctrllimited)
    print('actuator_forcerange', model.actuator_forcerange)
    print('actuator_forcelimited', model.actuator_forcelimited)

    num_dofs = len(model.actuator_length0)
    print('num_dofs', num_dofs)
    dof_names = [model.actuator(i).name for i in range(0, num_dofs)]

    ctrl_inds_all = get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names=dof_names)
    _ctrl_inds_pos, _ctrl_inds_vel, _ctrl_inds_tau = ctrl_inds_all
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)

    _debug = False
    state = None
    _n_frames = n_frames

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
    root_states = np.zeros(13, dtype=np_dtype)
    imu = np.zeros(12, dtype=np_dtype)

    if not compute_torque:
        if Kp is not None:
            model.actuator_gainprm[:, 0] = kps
            model.actuator_biasprm[:, 1] = -kps
        if Kd is not None:
            model.actuator_biasprm[:, 2] = -kds
    else:
        model.actuator_gainprm[:] = 0.
        model.actuator_biasprm[:] = 0.
        model.actuator_gainprm[:, 0] = 1.

    device = jax.devices(device)[0]
    print('device', device)
    with jax.default_device(device):
        sys = mjcf.load_model(model)
        link_names = sys.link_names
        print('link_names', sys.link_names)
        num_dofs = sys.act_size()
        pipeline = getattr(__import__(f'brax.{backend}.pipeline'), backend).pipeline
        init_q = np.zeros(sys.q_size())
        init_q[:3] = init_pos
        init_q[3] = init_rot[3]
        init_q[4:7] = init_rot[:3]
        init_qd = np.zeros(sys.qd_size())
        init_q[-num_dofs:] = default_dof_pos
        init_qd[0:3] = init_lin_vel
        init_qd[3:6] = init_ang_vel
        print('init_q', init_q)
        print('init_qd', init_qd)
        init_q = jp.array(init_q)
        init_qd = jp.array(init_qd)

    renderer = 'mujoco' if renderer is None else renderer
    if renderer == 'mujoco':
        import mujoco
        import mujoco.viewer
        # from mujoco import mjx
        geom_bodyid = model.geom_bodyid
        num_bodies = len(model.body_mass)
        body_names = [model.body(i).name for i in range(num_bodies)]
        link_inds = [(link_names.index(b) if b in link_names else None) for b in body_names]
        geom_inds = [link_inds[i] for i in geom_bodyid]
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.sync()
        # xpos_init = data.geom_xpos.copy()
        xpos_init = model.geom_pos.copy()
        xmat_init = data.geom_xmat.copy().reshape(-1, 3, 3)
        # geom_quat = model.geom_quat
        # print(xmat_init)
        # mat_flip = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
        # rot_flip = np.array([[0.7,0.,0.7,0.]])

    def pipeline_step(pipeline_state, action: jax.Array) -> base.State:
        """Takes a physics step using the physics pipeline."""
        if _n_frames < 2:
            return pipeline.step(sys, state, action, _debug)

        def f(state, _):
            return (
                pipeline.step(sys, state, action, _debug),
                None,
            )

        return jax.lax.scan(f, pipeline_state, (), _n_frames)[0]

    jit_pipeline_step = jax.jit(pipeline_step, device=device)

    def reset_fn():
        nonlocal state
        with jax.default_device(device):
            state = pipeline.init(sys, init_q, init_qd, _debug)
        rollout.clear()

    with jax.default_device(device):
        reset_fn()
        jit_pipeline_step(state, jp.zeros(num_dofs))

    def step_fn():
        nonlocal state
        for i in range(decimation):
            q, qd = state.q, state.qd
            tau = None
            if compute_torque and (ctrl_inds_pos or ctrl_inds_vel):
                tau = np.zeros(num_dofs, dtype=np_dtype) if not ctrl_inds_tau else action_tau
                if ctrl_inds_pos:
                    tau_pos = Kp * (action_pos - q) + Kd * (-qd)
                    tau[ctrl_inds_pos] = tau_pos[ctrl_inds_pos]
                if ctrl_inds_vel:
                    tau_vel = Kd * (action_vel - qd)
                    tau[ctrl_inds_vel] = tau_vel[ctrl_inds_vel]
            if not compute_torque and ctrl_inds_pos:
                actions = action_pos
                actions = actions + default_dof_pos
            if not compute_torque and ctrl_inds_vel:
                actions = action_vel
            if ctrl_inds_tau:
                tau = action_tau if tau is None else tau
            if tau is not None:
                if torque_limits is not None:
                    tau = np.clip(tau, -torque_limits, torque_limits)
                inds = _ctrl_inds_pos + _ctrl_inds_vel + _ctrl_inds_tau
                actions = tau
            # torque = actuator.to_tau(sys, motor_targets, q, qd)
            with jax.default_device(device):
                actions = jp.array(actions)
                # state = pipeline_step(state, actions)
                state = jit_pipeline_step(state, actions)

        q, qd = np.asarray(state.q), np.asarray(state.qd)
        _dof_pos = q[-num_dofs:]
        pos = q[0:3]
        rot = q[3:7]
        _dof_vel = qd[-num_dofs:]
        ang_vel = qd[3:6]
        _rpy = quat2rpy_np3(rot)
        dof_pos[:] = _dof_pos
        dof_vel[:] = _dof_vel
        # rot = np.array(rot)
        # x, xd = state.x, state.xd
        # ang_vel = brax.math.rotate(xd.ang[0], brax.math.quat_inv(x.rot[0]))
        imu[0:3] = _rpy
        # print(pos)
        # print(rot)
        # print(dof_pos)
        # print(qd)
        # lin_vel, ang_vel
        root_states[0:3] = pos
        # root_states[3:7] = rot
        # root_states[7:10] = lin_vel
        root_states[10:13] = ang_vel
        if not headless:
            if renderer == 'mujoco':
                # print(state.x.pos.shape)
                # print(state.x.rot.shape)
                pos = np.asarray(state.x.pos)
                rot = np.asarray(state.x.rot)
                # print('render')
                # print(data.geom_xmat[0])
                # print(data.geom_xmat[1])
                # print(data.geom_xmat[2])
                # print(rot[0])
                # rot = np.stack([-rot[:, 0], rot[:, 1], rot[:, 3], rot[:, 2]], axis=-1)
                # rot = np.stack([rot[:, 1], rot[:, 2], rot[:, 3], rot[:, 0]], axis=-1)
                # print(rot[0])
                # rot = quat_mul_np(rot, rot_flip)
                mat = quat2mat_np(rot, w_first=True)
                # mat = mat @ xmat_init
                mat = mat[geom_inds[1:]] @ xmat_init[1:]
                mat = mat.reshape(-1, 9)
                # print(mat[0])
                # print(data.geom_xpos.shape)
                # print(data.geom_xmat.shape)
                pos = pos[geom_inds[1:]]
                pos = pos + xpos_init[1:]
                data.geom_xpos[1:] = pos
                data.geom_xmat[1:] = mat

                viewer.sync()
            else:
                rollout.append(state)

    def close_fn():
        print('closing')
        if len(rollout):
            # s_rollout = []
            # for state in rollout:
            # ns = jax.tree_map(lambda x: x[0], state)
            # s_rollout.append(ns)
            # h = lite_render(sys.replace(dt=dt), s_rollout)
            h = lite_render(sys.replace(dt=dt), rollout)
            with open(html_path, 'w') as f:
                f.write(h)
            print('saved', len(rollout), html_path)

    return {
        'dt': dt,
        'step_fn': step_fn,
        'close_fn': close_fn,
        'reset_fn': reset_fn,
        'dof_names': dof_names,
        'dof_pos': dof_pos,
        'dof_vel': dof_vel,
        'action_pos': action_pos,
        'action_vel': action_vel,
        'action_tau': action_tau,
        'root_states': root_states,
        'imu': imu,
    }
