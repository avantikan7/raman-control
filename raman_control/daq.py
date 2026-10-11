from __future__ import annotations

import nidaqmx
import numpy as np

SAMPLERATE = 100000

__all__ = [
    "DigitalStateContextManager",
    "DaqController",
]


class DigitalStateContextManager:
    def __init__(self, shutter, open_):
        self.shutter = shutter
        self.open_ = open_

    def __enter__(self):
        self.shutter.write(self.open_)

    def __exit__(self, *exc):
        self.shutter.write(not self.open_)

    def __call__(self):
        self.shutter.write(self.open_)


class DaqController:
    """
    Interface for everything we contorl through the daq board.
    - laser shutter
    - laser galvo mirrors
    - focus filter actuator
    """

    _instance = None

    @classmethod
    def instance(
        cls,
        sampleClockSource: str = "/Dev1/PFI0",
        devName: str = "Dev1",
        channels: list[str] = ["Dev1/ao0", "Dev1/ao1"],
    ) -> DaqController:
        if (
            cls._instance is None
            or cls._instance._task_is_closed(cls._instance._galvo)
        ):
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = cls(sampleClockSource, devName, channels)
        return cls._instance

    def __init__(
        self,
        sampleClockSource="/Dev1/PFI0",
        devName="Dev1",
        channels=["Dev1/ao0", "Dev1/ao1"],
    ) -> None:
        self._sample_clock_source = sampleClockSource
        self._dev_name = devName
        self._channels = tuple(channels)

        # galvo mirror
        self._galvo = self._create_galvo_task()

        # laser shutter
        # self._shutter = nidaqmx.Task("shutterDO")
        # self._shutter.do_channels.add_do_chan("Dev1/port0/line0")
        # self._open_shutter = DigitalStateContextManager(self._shutter, True)
        # self._close_shutter = DigitalStateContextManager(self._shutter, False)

        # # focus filter actuator
        self._filter = nidaqmx.Task("filterDO")
        self._filter.do_channels.add_do_chan("Dev1/port0/line1")
        self._remove_filter = DigitalStateContextManager(self._filter, False)
        self._insert_filter = DigitalStateContextManager(self._filter, True)

    @staticmethod
    def _task_is_closed(task) -> bool:
        return task is None or getattr(task, "_handle", None) is None

    def _create_galvo_task(self):
        galvo = nidaqmx.Task("galvoAO")
        galvo.ao_channels.add_ao_voltage_chan(
            self._channels[0], "x", min_val=-10, max_val=10
        )
        galvo.ao_channels.add_ao_voltage_chan(
            self._channels[1], "y", min_val=-10, max_val=10
        )
        return galvo

    @staticmethod
    def _close_task(task):
        if DaqController._task_is_closed(task):
            return
        try:
            task.stop()
        except nidaqmx.errors.DaqError:
            pass
        try:
            task.close()
        except nidaqmx.errors.DaqError:
            pass

    def _replace_galvo_task(self):
        self._close_task(self._galvo)
        self._galvo = self._create_galvo_task()

    def _stop_galvo_for_reconfiguration(self):
        if self._task_is_closed(self._galvo):
            self._replace_galvo_task()
            return
        try:
            self._galvo.stop()
        except nidaqmx.errors.DaqError as error:
            if error.error_code != -200088:
                raise
            self._replace_galvo_task()

    def set_galvo_position(self, volts: np.ndarray) -> None:
        """Hold the galvos at one X/Y voltage using on-demand output.

        Collection configures the task for a finite external sample clock.
        Recreate it here so a manual point command is emitted immediately
        instead of waiting for another camera trigger.
        """
        volts = np.asarray(volts, dtype=float)
        if volts.shape != (2,):
            raise ValueError(
                f"galvo position must have shape (2,), got {volts.shape}"
            )
        if not np.all(np.isfinite(volts)):
            raise ValueError("galvo position must contain finite voltages")

        self._replace_galvo_task()
        self._galvo.write(
            np.ascontiguousarray(volts),
            auto_start=True,
        )

    @property
    def remove_filter(self) -> DigitalStateContextManager:
        return self._remove_filter

    @property
    def insert_filter(self) -> DigitalStateContextManager:
        return self._insert_filter

    @property
    def open_shutter(self) -> DigitalStateContextManager:
        return self._open_shutter

    @property
    def close_shutter(self) -> DigitalStateContextManager:
        return self._close_shutter

    @property
    def galvo(self) -> nidaqmx.Task:
        return self._galvo

    def close(self):
        """
        stop then close the galvo and shutter daq connections
        """
        # self._shutter.stop()
        # self._shutter.close()
        self._close_task(self._galvo)
        self._close_task(self._filter)
        self._galvo = None
        self._filter = None
        if type(self)._instance is self:
            type(self)._instance = None

    def prepare_for_collection(self, points: np.ndarray, batch=False, exposure=None):
        """
        Set up the galvo to aim at positions on camera frames.

        Parameters
        ----------
        points : 2xN array
            In volts.
        """
        points = np.ascontiguousarray(points)
        if points.ndim != 2 or points.shape[0] != 2 or points.shape[1] < 1:
            raise ValueError("Collection points must have shape (2, N), N >= 1")
        if points.shape[1] == 1:
            # A finite clocked task requires multiple samples. Hold this one
            # voltage pair on demand instead; the detector still captures
            # exactly the requested number of frames, without padding spectra.
            self.set_galvo_position(points[:, 0])
            return
        self._stop_galvo_for_reconfiguration()
        # xy_grid, volts = make_grid(N)
        if not batch:
            SAMPLERATE = 100000
            self._galvo.timing.cfg_samp_clk_timing(
                SAMPLERATE,
                source=self._sample_clock_source,
                active_edge=nidaqmx.constants.Edge.RISING,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=points.shape[1],
            )
            self._galvo.write(points, auto_start=True)
        else:
            sample_rate = points.shape[1] / exposure
            self._galvo.timing.cfg_samp_clk_timing(
                sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=points.shape[1],
            )
            self._galvo.triggers.start_trigger.cfg_dig_edge_start_trig(
                self._sample_clock_source
            )
            self._galvo.write(points)
            self._galvo.start()



# from __future__ import annotations

# import nidaqmx
# import numpy as np
# import time

# SAMPLERATE = 100000

# __all__ = [
#     "DigitalStateContextManager",
#     "DaqController",
# ]


# class DigitalStateContextManager:
#     def __init__(self, shutter, open_):
#         self.shutter = shutter
#         self.open_ = open_

#     def __enter__(self):
#         self.shutter.write(self.open_)

#     def __exit__(self, *exc):
#         self.shutter.write(not self.open_)

#     def __call__(self):
#         self.shutter.write(self.open_)


# class DaqController:
#     """
#     Interface for everything we contorl through the daq board.
#     - laser shutter
#     - laser galvo mirrors
#     - focus filter actuator
#     """

#     _instance = None

#     @classmethod
#     def instance(
#         cls,
#         sampleClockSource: str = "PFI0",
#         devName: str = "Dev1",
#         channels: list[str] = ["Dev1/ao0", "Dev1/ao1"],
#     ) -> DaqController:
#         if cls._instance is None:
#             cls._instance = cls(sampleClockSource, devName, channels)
#         return cls._instance

#     def __init__(
#         self,
#         sampleClockSource="PFI0",
#         devName="Dev1",
#         channels=["Dev1/ao0", "Dev1/ao1"],
#     ) -> None:
#         # galvo mirror
#         self._galvo = nidaqmx.Task("galvoAO")
#         self._galvo.ao_channels.add_ao_voltage_chan(
#             channels[0], "x", min_val=-10, max_val=10
#         )
#         self._galvo.ao_channels.add_ao_voltage_chan(
#             channels[1], "y", min_val=-10, max_val=10
#         )

#         # laser shutter
#         self._shutter = nidaqmx.Task("shutterDO")
#         self._shutter.do_channels.add_do_chan("Dev1/port0/line0")
#         self._open_shutter = DigitalStateContextManager(self._shutter, True)
#         self._close_shutter = DigitalStateContextManager(self._shutter, False)

#         # focus filter actuator
#         self._filter = nidaqmx.Task("filterDO")
#         self._filter.do_channels.add_do_chan("Dev1/port2/line4")
#         self._remove_filter = DigitalStateContextManager(self._filter, False)
#         self._insert_filter = DigitalStateContextManager(self._filter, True)

#     @property
#     def remove_filter(self) -> DigitalStateContextManager:
#         return self._remove_filter

#     @property
#     def insert_filter(self) -> DigitalStateContextManager:
#         return self._insert_filter

#     @property
#     def open_shutter(self) -> DigitalStateContextManager:
#         return self._open_shutter

#     @property
#     def close_shutter(self) -> DigitalStateContextManager:
#         return self._close_shutter

#     @property
#     def galvo(self) -> nidaqmx.Task:
#         return self._galvo

#     def close(self):
#         """
#         stop then close the galvo and shutter daq connections
#         """
#         self._shutter.stop()
#         self._shutter.close()
#         self._galvo.stop()
#         self._galvo.close()
#         self._filter.stop()
#         self._filter.close()

#     def prepare_for_collection(self, points: np.ndarray):
#         """
#         Set up the galvo to aim at positions on camera frames.

#         Parameters
#         ----------
#         points : 2xN array
#             In volts.
#         """
#         SAMPLERATE = 100000
#         SAMPLECLOCKSOURCE = "PFI0"

#         # first move to the first point so that later on we can use 
#         # falling edge without having an off by one error.
#         self._galvo.stop()
#         self._galvo.timing.cfg_samp_clk_timing(
#             SAMPLERATE,
#             source="",
#             active_edge=nidaqmx.constants.Edge.RISING,
#             sample_mode=nidaqmx.constants.AcquisitionType.HW_TIMED_SINGLE_POINT,
#             samps_per_chan=1,
#         )
#         self._galvo.write(np.ascontiguousarray(points[:, 0]), auto_start=True)

#         # wait at least two cycles of the onboard sample clock
#         # in order to ensure that we've moved
#         # this should be extremely fast (like 1e-5 seconds by default)
#         time.sleep(2 * (1 / self._galvo.timing.samp_clk_rate)) 

#         self._galvo.stop()

#         self._galvo.timing.cfg_samp_clk_timing(
#             SAMPLERATE,
#             source=SAMPLECLOCKSOURCE,
#             active_edge=nidaqmx.constants.Edge.FALLING,
#             sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
#             samps_per_chan=points.shape[1] - 1,
#         )
#         self._galvo.write(np.ascontiguousarray(points[:, 1:]), auto_start=True)
