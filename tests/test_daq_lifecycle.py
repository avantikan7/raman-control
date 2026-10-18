import numpy as np
from nidaqmx.errors import DaqError

from raman_control import daq


class FakeChannels:
    def __init__(self):
        self.added = []

    def add_ao_voltage_chan(self, *args, **kwargs):
        self.added.append((args, kwargs))

    def add_do_chan(self, *args, **kwargs):
        self.added.append((args, kwargs))


class FakeStartTrigger:
    def __init__(self):
        self.source = None

    def cfg_dig_edge_start_trig(self, source):
        self.source = source

    def disable_start_trig(self):
        self.source = None


class FakeTiming:
    def __init__(self):
        self.config = None

    def cfg_samp_clk_timing(self, *args, **kwargs):
        self.config = (args, kwargs)


class FakeTask:
    def __init__(self, name):
        self.name = name
        self._handle = object()
        self.ao_channels = FakeChannels()
        self.do_channels = FakeChannels()
        self.timing = FakeTiming()
        self.triggers = type(
            "Triggers", (), {"start_trigger": FakeStartTrigger()}
        )()
        self.stop_error = None
        self.closed = False
        self.written = None

    def stop(self):
        if self.stop_error is not None:
            raise self.stop_error

    def close(self):
        self.closed = True
        self._handle = None

    def write(self, points, auto_start=False):
        self.written = (points, auto_start)

    def start(self):
        pass


def _fake_tasks(monkeypatch):
    tasks = []

    def create_task(name):
        task = FakeTask(name)
        tasks.append(task)
        return task

    monkeypatch.setattr(daq.nidaqmx, "Task", create_task)
    daq.DaqController._instance = None
    return tasks


def test_closed_singleton_is_replaced_on_reconnect(monkeypatch):
    tasks = _fake_tasks(monkeypatch)
    first = daq.DaqController.instance()
    first.close()

    second = daq.DaqController.instance()

    assert second is not first
    assert second.galvo is tasks[2]
    assert tasks[0].closed
    assert tasks[1].closed


def test_prepare_recreates_task_after_invalid_task_error(monkeypatch):
    tasks = _fake_tasks(monkeypatch)
    controller = daq.DaqController.instance()
    tasks[0].stop_error = DaqError("invalid task", -200088)

    controller.prepare_for_collection(np.zeros((2, 2)))

    assert controller.galvo is tasks[2]
    assert tasks[0].closed
    assert tasks[2].written[1] is True
    assert tasks[2].timing.config[1]["source"] == "/Dev1/PFI0"


def test_set_galvo_position_replaces_timed_task_with_on_demand_task(
    monkeypatch,
):
    tasks = _fake_tasks(monkeypatch)
    controller = daq.DaqController.instance()
    original = controller.galvo
    controller.prepare_for_collection(np.zeros((2, 2)))

    controller.set_galvo_position(np.array([0.25, -0.5]))

    assert original.closed
    assert controller.galvo is tasks[2]
    assert tasks[2].timing.config is None
    np.testing.assert_array_equal(
        tasks[2].written[0], np.array([0.25, -0.5])
    )
    assert tasks[2].written[1] is True


def test_single_collection_point_uses_voltage_hold_without_padding(monkeypatch):
    tasks = _fake_tasks(monkeypatch)
    controller = daq.DaqController.instance()
    controller.prepare_for_collection(np.zeros((2, 2)))
    timed = controller.galvo
    for batch in (False, True):
        controller.prepare_for_collection(np.array([[0.25], [-0.5]]), batch=batch, exposure=1)
        assert controller.galvo.timing.config is None
        assert controller.galvo.triggers.start_trigger.source is None
        np.testing.assert_array_equal(controller.galvo.written[0], [0.25, -0.5])
        assert controller.galvo.written[1] is True
    assert timed.closed
    controller.prepare_for_collection(np.zeros((2, 3)))
    assert controller.galvo.timing.config[1]["samps_per_chan"] == 3


def test_switch_between_batch_and_camera_clock(monkeypatch):
    _fake_tasks(monkeypatch)
    controller = daq.DaqController.instance()
    for _ in range(2):
        controller.prepare_for_collection(np.zeros((2, 25)), batch=True, exposure=.1)
        args, kwargs = controller.galvo.timing.config
        assert args == (250.0,)
        assert kwargs["source"] == ""
        assert kwargs["samps_per_chan"] == 25
        assert controller.galvo.triggers.start_trigger.source == "/Dev1/PFI0"
        controller.prepare_for_collection(np.zeros((2, 3)))
        assert controller.galvo.timing.config[1]["source"] == "/Dev1/PFI0"
        assert controller.galvo.triggers.start_trigger.source is None
