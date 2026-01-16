''' metrics.py
A simple metric collector and reporter
'''

import time
from contextlib import contextmanager

EPOCH_FIELDS = (
    "running_time",
    "training_time",
    "testing_time",
    "sent_to_split",
    "recv_from_split",
    "server_work_time",
)

ROUND_EXTRA_FIELDS = (
    "sent_to_fed",
    "recv_from_fed",
    "fed_wait_time",
    "init_time",
    "send_weights_time",
)

OVERALL_EXTRA_FIELDS = (
    "initial_loading_time",
    "total_round_init_time",
    "total_training_time",
    "total_running_time",
)

ROUND_FIELDS = EPOCH_FIELDS + ROUND_EXTRA_FIELDS
ALL_FIELDS = ROUND_FIELDS + OVERALL_EXTRA_FIELDS


class MetricsBlock(fields):
    __slots__ = fields

    def __init__(self):
        self.reset()

    def reset(self):
        for name in self.__slots__:
            setattr(self, name, 0)

    def add_from(self, other, fields):
        for field in fields:
            setattr(self, field, get_attr(self, field) + getattr(other, field))

class Metrics:
    def __init__(self):
        self.overall = MetricsBlock()
        self.round = MetricsBlock()
        self.epoch = MetricsBlock()

        #Time-keepers
        self.last_server_work_time = 0
        self.last_step_time = 0

    def endEpoch(self):
        self.round.add_from(self.epoch, EPOCH_FIELDS)
        self.epoch.reset()

    def endRound(self):tra
        self.overall.add_from(self.round, ROUND_FIELDS)
        self.round.reset()

   

    ###########################################################################
    # timer contexts
    ###########################################################################
    @contextmanager
    def server_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.last_server_work_time = elapsed
            self.epoch.server_work_time += elapsed

    @contextmanager
    def step_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.last_step_time = elapsed

    @contextmanager
    def epoch_running_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.epoch.running_time = elapsed
            self.endEpoch()

    def epoch_training_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.epoch.training_time = elapsed

    @contextmanager
    def round_running_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.round.running_time = elapsed

    def overall_running_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.overall.running_time = elapsed

    ################## setup timers ####################
    @contextmanager
    def round_init_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.round.init_time = elapsed
            self.overall.total_round_init_time += elapsed

    @contextmanager
    def initial_loading_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.overall.initial_loading_time = elapsed

    @contextmanager
    def weights_receiving_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.round.fed_wait_time = elapsed

    @contextmanager
    def weights_sending_timer(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.round.send_weights_time = elapsed

    ###########################################################################
    # Reporting
    ###########################################################################
    
    def reportEpoch(self, round_ epoch, log=None):
        metric = self.epoch
        out = (
            f"========= Client R{round_} E{epoch} Statistics ==========\n"
            f"Time statistics:\n"
            f"  Running time  : {m.running_time:.4f}\n"
            f"  Training time : {m.training_time:.4f}\n"
            f"Training networking statistics:\n"
            f"  Sent to split : {m.sent_to_split}\n"
            f"  Recv from split: {m.recv_from_split}\n"
            f"=================================="
        )
        print(out)
        if log:
            log.info(out)


    def reportRound(self, round_, log=None):
        metric = self.round_
        total_sent = metric.sent_to_split + metric.sent_to_fed
        total_recv = metric.recv_from_split + recv_from_fed
        total_transmitted = total_sent + total_recv

        out = (
            f"======== Round {r} Summary ========\n"
            f"Time statistics:\n"
            f"Running time: {metric.running_time:.4f}\n"
            f"Training time: {metric.training_time:.4f}\n"
            f"Server work time: {metric.server_work_time:.4f}\n"
            f"Fed wait time: {metric.fed_wait_time:.4f}\n"
            f"Round initialization time: {metric.init_time}\n"
            f"Networking Statistics:\n"
            f"Data sent to Split Server: {metric.sent_to_split}\n"
            f"Data sent to Fed Server: {metric.sent_to_fed}\n"
            f"Total sent {total_sent} bytes\n"
            f"Data received from Split Server: {metric.recv_from_split} bytes\n"
            f"Data received from Fed Server: {metric.recv_from_fed} bytes\n"
            f"Total received: {total_recv} bytes\n"
            f"===== Total Transmitted this Round: {total_transmitted} bytes ====="
        )
        if log:
            log.info(out)
    
    def reportOverall(self, log=None):
        metric = self.metric
        total_sent = metric.sent_to_split + metric.sent_to_fed
        total_recv = metric.recv_from_split + metric_recv_from_fed
        total_transmitted = total_sent + total_recv


        out = (
            f"======== Global Summary ========\n"
            f"Time statistics:\n"
            f"Running time: {metric.running_time:.4f}\n"
            f"Initial loading time: {getattr(metric, 'initial_loading_time', 0):.4f}\n"
            f"Round init time: {getattr(metric, 'round_init_time', 0):.4f}\n"
            f"Total training time: {getattr(metric, 'total_training_time', 0):.4f}\n"
            f"Server work time: {metric.server_work_time:.4f}\n"
            f"Fed wait time: {metric.fed_wait_time:.4f}\n"
            f"Networking statistics:\n"
            f"Data sent to Split Server: {metric.sent_to_split} bytes\n"
            f"Data sent to Fed Server: {metric.sent_to_fed} bytes\n"
            f"Total sent: {total_sent} bytes\n"
            f"Data received from Split Server: {metric.recv_from_split} bytes\n"
            f"Data received from Fed Server: {metric.recv_from_fed} bytes\n"
            f"Total received: {total_recv} bytes\n"
            f"Total transmitted: {total_transmitted} bytes\n"
            f"Model statistics:\n"
        )

    print(out)
    if log:
        log.info(out)
