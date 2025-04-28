import threading
import time

from roslibpy import Message, Ros, Time, Topic
from roslibpy.ros2 import Header
from roslibpy.core import RosTimeoutError

# apt install ros-humble-rosbridge-suite
# start the rosbridge server first:
# ros2 launch rosbridge_server rosbridge_websocket_launch.xml


def run_rosbridge_server(host, port):
    import time
    import subprocess

    def try_conn():
        import socket
        s = socket.socket()
        fail = False
        try:
            s.connect((host, port))
        except Exception:
            fail = True
        finally:
            s.close()
        return fail

    if not try_conn():
        print('rosbridge server already running')
        return None
    args = ['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml', f'address:={host}', f'port:={port}']
    proc = subprocess.Popen(args)
    time.sleep(2)
    ret = proc.poll()
    if ret is not None:
        proc.kill()
        raise RuntimeError('rosbridge server failed to start')
    time.sleep(2)
    if try_conn():
        proc.kill()
        raise RuntimeError('failed connecting to rosbridge server')
    return proc


def run_fn(
    cb_init=None,
    cb_recv=None,
    cb_send=None,
    pub_dt=0.1,
    sub_topic='/isaac_joint_commands',
    pub_topic='/isaac_joint_states',
    verbose=False,
    run_server=True,
    host='127.0.0.1',
    port=9090,
):
    if run_server:
        proc = run_rosbridge_server(host, port)

    context = {}
    context['stop'] = False

    ros = Ros(host, port)

    try:
        ros.run()
    except RosTimeoutError as e:
        import traceback
        traceback.print_exc()
        cb_recv(True)
        ros.close()
        raise e

    sub = Topic(ros, sub_topic, 'sensor_msgs/JointState')
    pub = Topic(ros, pub_topic, 'sensor_msgs/JointState')

    _ctx = None
    if cb_init is not None:
        _ctx = cb_init()

    def receive_message(msg):
        if verbose:
            print('recv', msg['header']['stamp'])
        if cb_recv is not None:
            cb_recv(msg)

    def start_sending():
        t0 = time.perf_counter()
        while True:
            if context['stop']:
                break
            if cb_send is not None:
                ctx = cb_send()
            else:
                ctx = _ctx
            if ctx is None:
                # print('empty send msg')
                time.sleep(1)
                continue
            send_name = ctx.get('name')
            send_position = ctx.get('position')
            send_velocity = ctx.get('velocity')
            send_effort = ctx.get('effort')
            if send_name is None or send_position is None or send_velocity is None or send_effort is None:
                print('invalid send msg')
                time.sleep(1)
                continue
            msg = dict(header=Header(stamp=Time.now(), frame_id=''))
            # print('send_position', send_position)
            msg['name'] = send_name
            msg['position'] = send_position
            msg['velocity'] = send_velocity
            msg['effort'] = send_effort
            msg = Message(msg)
            pub.publish(msg)
            t0 += pub_dt
            t_s = t0 - time.perf_counter()
            if t_s > 0:
                time.sleep(t_s)
            else:
                print('pub timeout', t_s)
            if verbose:
                print('send', msg['header']['stamp'], t_s)
        pub.unadvertise()

    def start_receiving():
        sub.subscribe(receive_message)

    t1 = threading.Thread(target=start_receiving)
    t2 = threading.Thread(target=start_sending)

    t1.start()
    t2.start()

    def close_fn():
        cb_recv(True)
        sub.unsubscribe()
        context['stop'] = True
        ros.close()
        t1.join()
        t2.join()
        if run_server and proc is not None:
            proc.terminate()
            time.sleep(1)
            proc.kill()
            proc.wait()

    return close_fn
