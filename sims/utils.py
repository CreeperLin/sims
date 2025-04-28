import re
import numpy as np


def quat_rotate_np(q, v, w_first=False):
    q_w, q_vec = (q[..., 0], q[..., 1:]) if w_first else (q[..., -1], q[..., :3])
    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v) * q_w[..., np.newaxis] * 2.0
    c = q_vec * np.sum(q_vec[..., np.newaxis, :] * v[..., np.newaxis, :], axis=-1) * 2.0
    return a + b + c


def quat_rotate_inverse_np(q, v, w_first=False):
    q_w, q_vec = (q[..., 0], q[..., 1:]) if w_first else (q[..., -1], q[..., :3])
    a = v * (2.0 * q_w**2 - 1.0)[..., np.newaxis]
    b = np.cross(q_vec, v) * q_w[..., np.newaxis] * 2.0
    c = q_vec * np.sum(q_vec[..., np.newaxis, :] * v[..., np.newaxis, :], axis=-1) * 2.0
    return a - b + c


def quat_mul_np(a, b, w_first=False):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    i, j, k, w = ([1, 2, 3, 0] if w_first else [0, 1, 2, 3])
    x1, y1, z1, w1 = a[:, i], a[:, j], a[:, k], a[:, w]
    x2, y2, z2, w2 = b[:, i], b[:, j], b[:, k], b[:, w]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    els = [w, x, y, z] if w_first else [x, y, z, w]
    quat = np.stack(els, axis=-1).reshape(shape)
    return quat


def quat2mat_np(quat, w_first=False):
    shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)
    if w_first:
        w, x, y, z = quat.T
    else:
        x, y, z, w = quat.T
    norm = np.linalg.norm(quat, axis=-1)
    s = 2.0 / norm
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    rot_matrix = np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY], [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                           [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])
    # return rot_matrix
    return np.transpose(rot_matrix, (2, 0, 1)).reshape(*shape, 3, 3)


def mat2quat_np(m, w_first=False):
    m00 = m[..., 0, 0]
    m11 = m[..., 1, 1]
    m22 = m[..., 2, 2]
    m21 = m[..., 2, 1]
    m12 = m[..., 1, 2]
    m02 = m[..., 0, 2]
    m20 = m[..., 2, 0]
    m01 = m[..., 0, 1]
    m10 = m[..., 1, 0]
    w = np.sqrt(np.maximum(1 + m00 + m11 + m22, 0))
    x = np.sqrt(np.maximum(1 + m00 - m11 - m22, 0))
    y = np.sqrt(np.maximum(1 - m00 + m11 - m22, 0))
    z = np.sqrt(np.maximum(1 - m00 - m11 + m22, 0))
    x = np.copysign(x, m21 - m12)
    y = np.copysign(y, m02 - m20)
    z = np.copysign(z, m10 - m01)
    els = [w, x, y, z] if w_first else [x, y, z, w]
    return 0.5 * np.stack(els, axis=-1)


def quat2rpy_np3(q, w_first=False):
    if w_first:
        inds = [1, 2, 3, 0]
    else:
        inds = [0, 1, 2, 3]
    x, y, z, w = [q[..., i] for i in inds]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = np.atan2(t0, t1)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    # pitch_y = np.asin(t2)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    # yaw_z = np.atan2(t3, t4)
    yaw_z = np.arctan2(t3, t4)
    return np.stack([roll_x, pitch_y, yaw_z], axis=-1)


def set_seed(seed):
    import os
    import random
    import numpy as np
    import torch
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def match_joint_specs(name, specs):
    if specs is None:
        return False
    if isinstance(specs, str) and specs == '*':
        return True
    if not isinstance(specs, (tuple, list)):
        specs = [specs]
    for spec in specs:
        assert isinstance(spec, str)
        if spec in name:
            return True
        match = re.match(spec, name)
        if match is not None:
            return True
    return False


def dict2list(dct, keys, default=None, match_fn=match_joint_specs):
    inds = []
    vals = []
    for i, key in enumerate(keys):
        matched = False
        val = None
        for k, v in dct.items():
            matched = key == k
            if matched:
                val = v
                break
        if not matched:
            for k, v in dct.items():
                matched = match_fn(key, k)
                # print(key, k, matched)
                if matched:
                    val = v
                    break
        # print(matched, key, k)
        if not matched and default is not None:
            val = default
            matched = True
        if matched:
            inds.append(i)
            vals.append(val)
    return inds, vals


def update_array(x, obj, keys):
    import torch
    if obj is None:
        return
    inds = slice(None)
    n = len(x)
    if isinstance(obj, (float, int)):
        obj = [obj] * n
    if isinstance(obj, dict):
        inds, obj = dict2list(obj, keys, obj.pop('_default', None))
        # print('update_array', inds, obj)
        n = len(inds)
    if isinstance(x, torch.Tensor):
        obj = torch.from_numpy(obj) if isinstance(obj, np.ndarray) else torch.tensor(obj)
        obj = obj.view(-1).to(dtype=x.dtype, device=x.device)
    x[inds] = obj[:n]


def update_obj(obj, dct):
    if dct is None:
        return
    for key, val in dct.items():
        attr = getattr(obj, key, None)
        # if isinstance(val, dict) and isinstance(attr, object) and attr is not None:
        if isinstance(val, dict) and type(attr).__module__ != 'builtins' and attr is not None:
            update_obj(attr, val)
        else:
            eq = attr != val
            if isinstance(eq, bool):
                if eq:
                    setattr(obj, key, val)
                    print(type(obj).__name__, key, attr, val)
            else:
                attr[:] = val


def obj2dict(obj, memo=None):
    if type(obj).__module__ in ['builtins', 'numpy']:
        return obj
    dct = {}
    memo = set() if memo is None else memo
    for key in dir(obj):
        # if key.startswith('_'):
        #     continue
        if key.startswith('__'):
            continue
        attr = getattr(obj, key)
        if callable(attr):
            continue
        if attr is obj:
            continue
        # print(obj, key)
        if key in memo:
            dct[key] = attr
            continue
        memo.add(key)
        dct[key] = obj2dict(attr, memo)
    return dct


def list2slice(lst):
    if len(lst) < 2:
        return lst
    st, ed = lst[0], lst[-1]
    if len(lst) == (ed - st + 1) and tuple(sorted(lst)) == tuple(lst):
        lst = slice(st, ed + 1)
    return lst


def get_ctrl_inds(ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau, dof_names):
    num_dofs = len(dof_names)
    joints_all = [ctrl_joints_pos, ctrl_joints_vel, ctrl_joints_tau]
    ctrl_inds_all = [[], [], []]
    for i, name in enumerate(dof_names):
        for ctrl_inds, joints in zip(ctrl_inds_all, joints_all):
            matched = match_joint_specs(name, joints)
            if matched:
                ctrl_inds.append(i)
    ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = ctrl_inds_all
    rest_inds = list(set(range(num_dofs)) - set(ctrl_inds_pos + ctrl_inds_vel))
    print('ctrl_inds_pos', ctrl_inds_pos, [dof_names[i] for i in ctrl_inds_pos])
    print('ctrl_inds_vel', ctrl_inds_vel, [dof_names[i] for i in ctrl_inds_vel])
    print('ctrl_inds_tau', ctrl_inds_tau, [dof_names[i] for i in ctrl_inds_tau])
    print('rest_inds', rest_inds, [dof_names[i] for i in rest_inds])
    print('ctrl_inds_all', ctrl_inds_all)
    return ctrl_inds_all
    # _ctrl_inds_pos, _ctrl_inds_vel, _ctrl_inds_tau = ctrl_inds_all
    # ctrl_inds_pos, ctrl_inds_vel, ctrl_inds_tau = map(list2slice, ctrl_inds_all)
