import numpy as np
from collections import deque


def copy_dct(dct):
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in dct.items()}


class FakeLatencyWrapper:

    def __init__(
        self,
        system,
        send_lat=0,
        recv_lat=1,
    ) -> None:
        self.system = system
        # obs latency
        self.send_lat = send_lat
        # action latency
        self.recv_lat = recv_lat
        self.last_msg = None
        self.last_ctx = None
        if send_lat > 1:
            self.q_ctx = deque(maxlen=send_lat)
        if recv_lat > 1:
            self.q_msg = deque(maxlen=recv_lat)

    def cb_init(self):
        return self.system.cb_init()

    def cb_recv(self, msg):
        recv_lat = self.recv_lat
        if recv_lat and isinstance(msg, dict):
            orig_msg = copy_dct(msg)
            if recv_lat == 1:
                last_msg = self.last_msg
                msg = msg if last_msg is None else last_msg
                self.last_msg = orig_msg
            else:
                msg = self.q_msg[0] if len(self.q_msg) else msg
                self.q_msg.append(orig_msg)
        ret = self.system.cb_recv(msg)
        return ret

    def cb_send(self):
        ctx = self.system.cb_send()
        send_lat = self.send_lat
        if send_lat:
            orig_ctx = copy_dct(ctx)
            if send_lat == 1:
                last_ctx = self.last_ctx
                ctx = ctx if last_ctx is None else last_ctx
                self.last_ctx = orig_ctx
            else:
                ctx = self.q_ctx[0] if len(self.q_ctx) else ctx
                self.q_ctx.append(orig_ctx)
        return ctx
