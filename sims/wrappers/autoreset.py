import numpy as np


class AutoResetWrapper:

    def __init__(
        self,
        system,
        reset_rp=0.8,
        # reset_z=0.4,
        reset_z=0.,
        # reset_max_steps=0,
        reset_max_steps=1024,
    ) -> None:
        self.system = system
        self.reset_z = reset_z
        self.reset_rp = reset_rp
        self.reset_max_steps = reset_max_steps
        self.step = 0
        self.last_reset = 0

    def cb_init(self):
        return self.system.cb_init()

    def cb_recv(self, msg):
        return self.system.cb_recv(msg)

    def cb_send(self):
        self.step += 1
        ctx = self.system.cb_send()
        reset = False
        if self.reset_rp > 0:
            rp = ctx['imu'][:2]
            if np.max(np.abs(rp)) > self.reset_rp:
                print('auto reset rp', rp)
                reset = True
        if self.reset_z > 0:
            z = ctx['root_states'][2]
            if z > 0 and z < self.reset_z:
                print('auto reset z', z)
                reset = True
        if self.reset_max_steps > 0 and self.step - self.last_reset > self.reset_max_steps:
            print('auto reset max steps', self.reset_max_steps)
            reset = True
        if reset:
            print('sims reset', self.step, self.step - self.last_reset, ctx['root_states'][:7])
            self.system.cb_recv(False)
            self.last_reset = self.step
        return ctx
