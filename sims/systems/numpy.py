import time
import threading

# ctx
# dof_names: List[str]
# step_fn: Callable
# close_fn: Callable
# reset_fn: Callable
# action_pos: numpy.ndarray
# action_vel: numpy.ndarray
# action_tau: numpy.ndarray
# dof_pos: numpy.ndarray
# dof_vel: numpy.ndarray
# dt: float


class NumpyContextSystem():

    _init_fn = None

    def __init__(self, sim_ctx=None, init_fn=None, fallback=False, strict=False, realtime=True, **kwargs) -> None:
        br_ctx = {}
        self.br_ctx = br_ctx
        init_fn = self._init_fn if init_fn is None else init_fn
        sim_ctx = init_fn(**kwargs) if sim_ctx is None else sim_ctx
        self.sim_ctx = sim_ctx
        self.stop = False
        dof_names = sim_ctx['dof_names']
        print('dof_names', dof_names)
        self.dof_names = tuple(dof_names)
        dof_map = {k: i for i, k in enumerate(dof_names)}
        self.dof_map = dof_map
        br_ctx['name'] = dof_names
        self.recv_name = None
        self.recv_position = None
        self.recv_velocity = None
        self.recv_effort = None
        self.fallback = fallback
        self.strict = strict
        self.realtime = realtime
        self.dof_inds_cache = {}
        dt = sim_ctx.get('dt', 0.02)

        if not realtime:
            self.th_sim = None
            return

        strict_dt = False
        strict_dt = True

        def th_sim_ctx():
            step_fn = sim_ctx['step_fn']
            close_fn = sim_ctx['close_fn']
            reset_fn = sim_ctx['reset_fn']
            reset_fn()
            ts = time.perf_counter()
            while True:
                t0 = time.perf_counter()
                # print(t0)
                if self.stop:
                    break
                step_fn()
                ts += dt
                if dt is False:
                    continue
                if strict_dt:
                    t_s = ts - time.perf_counter()
                else:
                    t_s = dt - (time.perf_counter() - t0)
                if t_s > 0:
                    time.sleep(t_s)
                else:
                    print('sim timeout', t_s)
            close_fn()

        # import multiprocessing
        # self.th_sim = multiprocessing.Process(target=th_sim_ctx)
        self.th_sim = threading.Thread(target=th_sim_ctx)

    # cb_init = None
    def cb_init(self):
        if self.th_sim is not None:
            self.th_sim.start()
        else:
            self.sim_ctx['reset_fn']()
        return self.br_ctx

    def cb_recv(self, msg):
        if msg is True:
            if self.stop:
                return
            self.stop = True
            if self.th_sim is not None and self.th_sim.is_alive():
                self.th_sim.join()
            else:
                self.sim_ctx['close_fn']()
            return
        if msg is False:
            self.sim_ctx['reset_fn']()
            return
        name = msg['name']
        self.recv_name = name
        position = msg.get('position', [])
        velocity = msg.get('velocity', [])
        effort = msg.get('effort', [])
        if not len(position) == len(velocity) == len(effort) == len(name):
            if self.strict:
                raise ValueError(f'invalid cmd msg: {msg}')
            print(len(position), len(name))
            num_joints = len(name)
            position.extend([0. for _ in range(num_joints - len(position))])
            velocity.extend([0. for _ in range(num_joints - len(velocity))])
            effort.extend([0. for _ in range(num_joints - len(effort))])
        sim_ctx = self.sim_ctx
        action_pos = sim_ctx['action_pos']
        action_vel = sim_ctx['action_vel']
        action_tau = sim_ctx['action_tau']
        key = tuple(name)
        if key == self.dof_names:
            action_pos[:] = position
            action_vel[:] = velocity
            action_tau[:] = effort
        else:
            dof_map = self.dof_map
            dof_inds = self.dof_inds_cache.get(key)
            if dof_inds is None:
                # print(name)
                src_inds = []
                dest_inds = []
                for i, k in enumerate(name):
                    idx = dof_map.get(k)
                    if idx is None:
                        continue
                    dest_inds.append(idx)
                    src_inds.append(i)
                if not len(src_inds):
                    print('no dof set', dof_map, name)
                else:
                    dof_inds = []
                    for inds in [src_inds, dest_inds]:
                        st = inds[0]
                        ed = inds[-1]
                        if len(inds) == (ed - st + 1) and tuple(sorted(inds)) == tuple(inds):
                            inds = slice(st, ed + 1)
                        dof_inds.append(inds)
                    print('dof_inds', len(key), dof_inds)
                    self.dof_inds_cache[key] = dof_inds
            src_inds, dest_inds = dof_inds
            action_pos[dest_inds] = position[src_inds]
            action_vel[dest_inds] = velocity[src_inds]
            action_tau[dest_inds] = effort[src_inds]
        if self.fallback:
            if self.recv_position is None:
                pass
            self.recv_position = position
            self.recv_velocity = velocity
            self.recv_effort = effort
        if not self.realtime:
            self.sim_ctx['step_fn']()

    # cb_send = None
    def cb_send(self):
        if self.stop:
            return None
        br_ctx = self.br_ctx
        dof_pos = self.sim_ctx['dof_pos']
        dof_vel = self.sim_ctx['dof_vel']
        dof_tau = self.sim_ctx['dof_tau']
        if self.fallback:
            if self.recv_name is None:
                return None
            dof_map = self.dof_map
            if self.recv_velocity is None or self.recv_position is None:
                return None
            recv_position = self.recv_position
            recv_velocity = self.recv_velocity
            recv_name = self.recv_name
            effort = self.recv_effort
            position = [(dof_pos[dof_map[k]] if k in dof_map else recv_position[i]) for i, k in enumerate(recv_name)]
            velocity = [(dof_vel[dof_map[k]] if k in dof_map else recv_velocity[i]) for i, k in enumerate(recv_name)]
            name = recv_name
        else:
            position = dof_pos
            velocity = dof_vel
            effort = dof_tau
            name = br_ctx['name']
        br_ctx.update({
            'name': name,
            'position': position,
            'velocity': velocity,
            'effort': effort,
            'root_states': self.sim_ctx.get('root_states'),
            # rpy, acc, ang_vel, lin_vel
            'imu': self.sim_ctx.get('imu'),
        })
        return br_ctx
