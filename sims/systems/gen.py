class GenericSystem():

    def cb_init(self):
        ctx = {}
        self.ctx = ctx
        return ctx

    def cb_recv(self, msg):
        if msg is True:
            return
        self.ctx['name'] = msg['name']
        self.ctx['position'] = msg['position']
        self.ctx['velocity'] = msg['velocity']
        self.ctx['effort'] = msg['effort']

    cb_send = None
