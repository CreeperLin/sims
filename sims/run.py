import os
import yaml
import argparse


def import_obj(name):
    spec = name.split(':')
    mod_path = spec[0]
    mod = __import__(mod_path, fromlist=[None])
    attr = spec[1] if len(spec) > 1 else getattr(mod, '__default__')
    return getattr(mod, attr)


def run(
    runner='roslibpy',
    system='sims.systems.pyb',
    config=None,
    system_config=None,
    runner_config=None,
    wrapper_config=None,
    preproc_config=None,
    use_numpy_sys=True,
    run_time=0,
):
    config, system_config, runner_config = [{} if c is None else c for c in [config, system_config, runner_config]]
    sys_kwds = config.get('system', {})
    sys_kwds.update(system_config)
    system = sys_kwds.pop('type', system)
    preproc_config = config.get('wrapper', preproc_config)
    preproc_config = [] if preproc_config is None or not len(preproc_config) else preproc_config
    preproc_config = preproc_config if isinstance(preproc_config, (tuple, list)) else [preproc_config]
    for conf in preproc_config:
        pp_name, pp_kwds = (conf, {}) if isinstance(conf, str) else (conf.pop('type'), conf)
        pp_cls = import_obj(pp_name)
        pp_cls(sys_kwds, **pp_kwds)
    if use_numpy_sys:
        from sims.systems.numpy import NumpyContextSystem
        if ':' not in system:
            system = system + ':init_fn'
        init_fn = import_obj(system)
        s = NumpyContextSystem(init_fn=init_fn, **sys_kwds)
    else:
        sys_cls = import_obj(system) if isinstance(system, str) else system
        s = sys_cls(**sys_kwds)
    if isinstance(runner, str) and '.' not in runner and ':' not in runner:
        runner = f'sims.runners.{runner}:run_fn'
    wrapper_config = config.get('wrapper', wrapper_config)
    wrapper_config = [] if wrapper_config is None or not len(wrapper_config) else wrapper_config
    wrapper_config = wrapper_config if isinstance(wrapper_config, (tuple, list)) else [wrapper_config]
    for conf in wrapper_config:
        w_name, kwds = (conf, {}) if isinstance(conf, str) else (conf.pop('type'), conf)
        w_cls = import_obj(w_name)
        s = w_cls(s, **kwds)
    if run_time is False:
        return s
    runner_kwds = config.get('runner', {})
    runner_kwds.update(runner_config)
    runner = runner_kwds.pop('type', runner)
    run_fn = import_obj(runner) if isinstance(runner, str) else runner
    close_fn = run_fn(
        cb_init=s.cb_init,
        cb_recv=s.cb_recv,
        cb_send=s.cb_send,
        **runner_kwds,
    )
    import time
    try:
        time.sleep(run_time or 2**20)
    finally:
        close_fn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--runner', default='roslibpy')
    parser.add_argument('-s', '--system', default='sims.systems.pyb')
    parser.add_argument('-c', '--config', default='')
    parser.add_argument('-sc', '--system_config', default='')
    parser.add_argument('-rc', '--runner_config', default='')
    parser.add_argument('-wc', '--wrapper_config', default='')
    parser.add_argument('-pc', '--preproc_config', default='')
    parser.add_argument('-np', '--use_numpy_sys', action='store_true')
    parser.add_argument('-t', '--run_time', type=int, default=0)
    args = parser.parse_args()
    configs = ['config', 'system_config', 'runner_config', 'wrapper_config']
    cfgs = [getattr(args, c) for c in configs]
    cfgs = [(yaml.safe_load(open(c, 'r') if os.path.exists(c) else c) or {}) for c in cfgs]
    kwds = vars(args)
    kwds.update({k: v for k, v in zip(configs, cfgs)})
    run(**kwds)
