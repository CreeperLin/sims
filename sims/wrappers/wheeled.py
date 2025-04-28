def run_fn_roslibpy(
    cb_recv,
    sub_topic='/cmd_vel',
    verbose=False,
    host='127.0.0.1',
    port=9090,
):
    import threading
    from roslibpy import Ros, Topic

    ros = Ros(host, port)
    sub = Topic(ros, sub_topic, 'geometry_msgs/Twist')

    def receive_message(msg):
        if verbose:
            print('recv', msg['header']['stamp'])
        cb_recv(msg)

    def start_receiving():
        sub.subscribe(receive_message)

    t1 = threading.Thread(target=start_receiving)
    t1.start()

    def close_fn():
        sub.unsubscribe()
        ros.close()
        t1.join()

    return close_fn


class WheeledControllerWrapper:

    def __init__(
        self,
        system,
        wheel_joint_names=['left_wheel_joint', 'right_wheel_joint'],
        wheel_radius=0.1,
        wheel_base=0.5,
    ) -> None:
        self.wheel_joint_names = wheel_joint_names
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.system = system
        self.last_cmd_msg = None
        self.close_fn = run_fn_roslibpy(self.on_cmd_msg)

    def on_cmd_msg(self, cmd_msg):
        self.last_cmd_msg = cmd_msg
        lin_vel = cmd_msg['linear']['x']
        ang_vel = cmd_msg['angular']['z']
        joint_vels = self.get_joint_vel(lin_vel, ang_vel)
        names = self.wheel_joint_names
        msg = {
            'name': names,
            'velocity': joint_vels,
            'position': [0.] * len(names),
            'effort': [0.] * len(names),
        }
        return self.system.cb_recv(msg)

    def cb_init(self):
        return self.system.cb_init()

    def get_joint_vel(self, lin_vel, ang_vel):
        wheel_base = self.wheel_base
        wheel_radius = self.wheel_radius
        vel_left = (2 * lin_vel - ang_vel * wheel_base) / (2 * wheel_radius)
        vel_right = (2 * lin_vel + ang_vel * wheel_base) / (2 * wheel_radius)
        return vel_left, vel_right

    def cb_recv(self, msg):
        if msg is True:
            self.close_fn()
        else:
            cmd_msg = self.last_cmd_msg
            if cmd_msg is not None:
                lin_vel = cmd_msg['linear']['x']
                ang_vel = cmd_msg['angular']['z']
                print(lin_vel, ang_vel)
                joint_vels = self.get_joint_vel(lin_vel, ang_vel)
                names = self.wheel_joint_names
                inds = [msg['name'].index(n) for n in names]
                for i, v in zip(inds, joint_vels):
                    msg['velocity'][i] = v
        return self.system.cb_recv(msg)

    def cb_send(self):
        return self.system.cb_send()
