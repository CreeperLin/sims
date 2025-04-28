import numpy as np

_default_recv_noise = {
    # 'position': [0, 0.02],
    'position': None,
    'velocity': None,
    'effort': None,
}

_default_send_noise = {
    # 'imu': None,
    'imu': ['normal', 0, 0.1],
    'root_states': None,
    'position': ['normal', 0, 0.02],
    'velocity': ['normal', 0, 1.0],
    'effort': None,
}


class NoisyWrapper:

    def __init__(
        self,
        system,
        default_mean=0.,
        default_std=0.05,
        recv_noise=None,
        send_noise=None,
        seed=11235,
    ) -> None:
        self.system = system
        self.default_mean = default_mean
        self.default_std = default_std
        self.recv_noise = _default_recv_noise if recv_noise is None else recv_noise
        self.send_noise = _default_send_noise if send_noise is None else send_noise
        default_noise = [default_mean, default_std]
        default_noise = None if any([x is None for x in default_noise]) else default_noise
        self.default_noise = default_noise
        self.rng = np.random.RandomState(seed)

    def cb_init(self):
        return self.system.cb_init()

    def cb_recv(self, msg):
        recv_noise = self.recv_noise
        default_noise = self.default_noise
        if recv_noise is not False and (recv_noise or default_noise) and isinstance(msg, dict):
            for k, v in msg.items():
                if not isinstance(v, np.ndarray):
                    continue
                params = recv_noise.get(k, default_noise)
                if params is None:
                    continue
                fn, args = params[0], params[1:]
                eps = getattr(self.rng, fn)(*args, v.shape)
                msg[k] = v + eps
        ret = self.system.cb_recv(msg)
        return ret

    def cb_send(self):
        ctx = self.system.cb_send()
        send_noise = self.send_noise
        default_noise = self.default_noise
        if send_noise is not False and (send_noise or default_noise):
            for k, v in ctx.items():
                if not isinstance(v, np.ndarray):
                    continue
                params = send_noise.get(k, default_noise)
                if params is None:
                    continue
                fn, args = params[0], params[1:]
                eps = getattr(self.rng, fn)(*args, v.shape)
                v[:] = v + eps
        return ctx
