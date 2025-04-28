def run_fn(
    cb_init=None,
    cb_recv=None,
    cb_send=None,
):
    if cb_init is not None:
        cb_init()

    running = True

    def close_fn():
        nonlocal running
        running = False
        cb_recv(True)

    return close_fn
