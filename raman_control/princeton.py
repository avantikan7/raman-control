
from __future__ import annotations

import ctypes
import os

# Import python sys module
import sys
import time

# Import the .NET class library
import clr

# numpy impor︃t
import numpy as np

# fmt: off
# Import c compatible List and String
from System import *  # noqa
from System.Collections.Generic import List  # noqa
from System.Runtime.InteropServices import GCHandle, GCHandleType  # noqa


from .calibration import CoordTransformer
from .daq import DaqController
from .spectral_axis import wavelengths_to_raman_shift
from .utils import make_grid
# Add needed dll references
sys.path.append(os.environ["LIGHTFIELD_ROOT"])
sys.path.append(os.environ["LIGHTFIELD_ROOT"] + "\\AddInViews")
clr.AddReference("PrincetonInstruments.LightFieldViewV5")
clr.AddReference("PrincetonInstruments.LightField.AutomationV5")
clr.AddReference("PrincetonInstruments.LightFieldAddInSupportServices")


from PrincetonInstruments.LightField.AddIns import *  # noqa
from PrincetonInstruments.LightField.Automation import *  # noqa

# fmt: on


class SpectraCollector:
    MAX_VOLTS: float = 10
    _instance = None

    @classmethod
    def instance(
        cls,
        lightFieldConfig: str = "Pixis_2Mhz",
        laser_controller: DaqController = None,
        coord_transformer: CoordTransformer = None,
    ) -> SpectraCollector:
        if cls._instance is None:
            cls._instance = cls(lightFieldConfig, laser_controller, coord_transformer)
        return cls._instance

    def __init__(
        self,
        lightFieldConfig: str = "Pixis_2MHz",
        laser_controller: DaqController = None,
        coord_transformer: CoordTransformer = None,
    ) -> None:
        self._setup_lightfield(lightFieldConfig)
        self._daq_controller = laser_controller or DaqController.instance()
        if coord_transformer is None:
            coord_transformer = CoordTransformer.from_json()  # load the default model
        self._coord_transformer = coord_transformer

    @property
    def coord_transformer(self) -> CoordTransformer:
        return self._coord_transformer

    @coord_transformer.setter
    def coord_transformer(self, val: CoordTransformer):
        if not isinstance(val, CoordTransformer):
            raise TypeError("Must be a CoordTransformer.")
        self._coord_transformer = val

    @property
    def daq(self) -> DaqController:
        return self._daq_controller

    def remove_filter(self, wait: int = 4000):
        """
        Remove the Focus filter to allow raman

        Parameters
        ----------
        wait : int
            How long to wait for the acututor to finish moving in ms.
        """
        self._daq_controller.remove_filter()
        time.sleep(wait)

    def insert_filter(self, wait: int = 4000):
        """
        Insert the Focus filter to allow raman autofocus

        Parameters
        ----------
        wait : int
            How long to wait for the acututor to finish moving in ms.
        """
        self._daq_controller.insert_filter()
        time.sleep(wait)

    @staticmethod
    def _convert_buffer(net_array, image_format):
        src_hndl = GCHandle.Alloc(net_array, GCHandleType.Pinned)
        try:
            src_ptr = src_hndl.AddrOfPinnedObject().ToInt64()

            # Possible data types returned from acquisition
            if image_format == ImageDataFormat.MonochromeUnsigned16:  # noqa: F405
                buf_type = ctypes.c_ushort * len(net_array)
            elif image_format == ImageDataFormat.MonochromeUnsigned32:  # noqa: F405
                buf_type = ctypes.c_uint * len(net_array)
            elif image_format == ImageDataFormat.MonochromeFloating32:  # noqa: F405
                buf_type = ctypes.c_float * len(net_array)

            cbuf = buf_type.from_address(src_ptr)
            resultArray = np.frombuffer(cbuf, dtype=cbuf._type_)

        # Free the handle
        finally:
            if src_hndl.IsAllocated:
                src_hndl.Free()

        # Make a copy of the buffer
        return np.copy(resultArray)

    def _convert_capture(self, cap, spectrum_length=1340):
        """
        parameters
        ----------
        cap : experiment.Capture
        spectrum_length : int
            1340 for current setup

        returns
        -------
        numpy array
            rows are samples
        """
        arr = np.zeros([cap.Frames, spectrum_length], dtype=np.uint16)
        for i in range(cap.Frames):
            frame = cap.GetFrame(0, i)
            arr[i] = self._convert_buffer(frame.GetData(), frame.Format)
        return arr

    def _setup_lightfield(self, config: str):
        """
        Start and set up lightfield.
        usage
        -----
        auto, experiment, set_value, set_rm_exposure = setup_lightfield()
        """
        # Create the LightField Application (true for visible)
        # The 2nd parameter forces LF to load with no experiment
        auto = Automation(True, List[String]())  # noqa: F405
        experiment = auto.LightFieldApplication.Experiment
        experiment.Load(config)

        self._auto = auto
        self._experiment = experiment
        self._set_full_vertical_binning()

    def _set_full_vertical_binning(self):
        """Combine every sensor row into one spectrum at runtime.

        ``SetLineSensorRegion`` changes only the currently loaded experiment.  The
        LightField experiment file is not modified unless it is explicitly saved.
        """
        sensor_height = int(self._experiment.FullSensorRegion.Height)
        if sensor_height < 1:
            raise RuntimeError("LightField reported an invalid sensor height")
        self._experiment.SetLineSensorRegion(sensor_height)

    def _set_value(self, setting, value):
        # Check for existence before setting
        # gain, adc rate, or adc quality
        if self._experiment.Exists(setting):
            self._experiment.SetValue(setting, value)

    def set_rm_exposure(self, exposure: float):
        """
        Sets the exposure time for raman

        Parameters
        ----------
        exposure : float
            camera exposure in milliseconds
        """
        self._set_value(
            CameraSettings.ShutterTimingExposureTime, float(exposure)  # noqa: F405
        )

    def get_wavelengths(self) -> np.ndarray:
        """Return LightField's calibrated wavelength for every detector column."""
        wavelengths = np.asarray(
            self._experiment.SystemColumnCalibration, dtype=float
        )
        if wavelengths.ndim != 1 or wavelengths.size == 0:
            raise RuntimeError(
                "LightField did not provide a wavelength calibration"
            )
        return wavelengths

    def get_laser_wavelength(self) -> float:
        """Return LightField's laser line used for relative wavenumbers."""
        setting = (
            ExperimentSettings  # noqa: F405
            .OnlineExportCsvFormatOptionsUnitsWavelengthLaserLine
        )
        if not self._experiment.Exists(setting):
            raise RuntimeError(
                "Configure a Relative Wavenumbers laser line in LightField"
            )
        wavelength = float(self._experiment.GetValue(setting))
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise RuntimeError(
                "LightField's Relative Wavenumbers laser line is invalid"
            )
        return wavelength

    def get_wavenumbers(self, expected_length: int | None = None) -> np.ndarray:
        """Return the LightField-calibrated Raman shift axis in cm^-1."""
        wavenumbers = wavelengths_to_raman_shift(
            self.get_wavelengths(), self.get_laser_wavelength()
        )
        if expected_length is not None and len(wavenumbers) != expected_length:
            raise RuntimeError(
                "LightField calibration length "
                f"({len(wavenumbers)}) does not match spectrum length "
                f"({expected_length})"
            )
        return wavenumbers

    def collect_spectra_relative(self, points, exposure=20):
        """
        Parameters
        ----------
        points : (N, 2) arraylike
        exposure : float, default 20


        Returns
        -------
        spectra : array
            with shape (N, 1340)
        """
        points = np.asarray(points)
        if points.min() < 0 or points.max() > 1:
            raise ValueError("Points must be in [0, 1]")
        if points.shape[1] != 2 or points.ndim != 2:
            raise ValueError(
                f"volts must have shape (N, 2) but has shape {points.shape}"
            )
        volts = self._coord_transformer.BF_to_volts(points)
        return self.collect_spectra_volts(volts, exposure)

    def collect_spectra_volts(self, volts, exposure=20):
        """
        Parameters
        ----------
        volts : arraylike
            shape (N, 2)
        exposure : float (Default: 20)
            exposure time in ms

        Returns
        -------
        spectra : array
            with shape (N, 1340)
        """
        points = np.asarray(volts)
        if points.shape[1] != 2 or points.ndim != 2:
            raise ValueError(
                f"volts must have shape (N, 2) but has shape {points.shape}"
            )
        if points.max() > self.MAX_VOLTS or points.min() < -self.MAX_VOLTS:
            raise ValueError(
                "Voltages out of "
                f"the safe range: [{-self.MAX_VOLTS:0.2f}, {self.MAX_VOLTS:0.2f}]"
            )
        # st = time.perf_counter()
        self.set_rm_exposure(exposure)
        # print(f"setting exposure time in LF took {time.perf_counter() - st} s")
        # st = time.perf_counter()
        # transpose to put into shape (2, N)
        # and make contiguous after bc transpose would undo that
        points = np.ascontiguousarray(points.T)
        self._daq_controller.prepare_for_collection(points)
        self._experiment.Stop()
        # print(f"preparing for collection took {time.perf_counter() - st}s")
        # st = time.perf_counter()
        # Reapply the runtime-only ROI in case it was changed in the LightField UI.
        self._set_full_vertical_binning()
        # print(f"setting the vertical binning took {time.perf_counter() - st} s")
        # st = time.perf_counter()
        # with self._daq_controller.open_shutter:
        dataset = self._experiment.Capture(points.shape[1])
        # print(f"Capturing took {time.perf_counter() - st} s")
        # st = time.perf_counter()
        self._daq_controller.galvo.stop()
        # print(f"Stopping the galvo took {time.perf_counter() - st} s")
        return self._convert_capture(dataset)

    def collect_spectra_pts(self, volts, exposure=20):
        """Collect one LightField frame per requested galvo point."""
        return self.collect_spectra_volts(volts, exposure)

    def collect_spectra_pts_batch(self, volts, exposure=20):
        """Integrate all galvo points into one frame over exposure milliseconds."""
        points = np.asarray(volts, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            raise ValueError("Batch voltages must have shape (N, 2), N >= 1")
        if not np.isfinite(points).all() or np.any(np.abs(points) > self.MAX_VOLTS):
            raise ValueError(f"Batch voltages must be finite and within +/-{self.MAX_VOLTS} V")
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("Batch exposure must be positive and finite")
        self._experiment.Stop()
        self.set_rm_exposure(exposure)
        self._set_full_vertical_binning()
        try:
            self._daq_controller.prepare_for_collection(
                np.ascontiguousarray(points.T), batch=True, exposure=exposure / 1000)
            dataset = self._experiment.Capture(1)
        finally:
            self._daq_controller.galvo.stop()
        spectra = self._convert_capture(dataset)
        if len(spectra) != 1:
            raise RuntimeError("Integrated batch capture must return exactly one spectrum")
        return spectra

    def capture_rm_grid(
        self,
        N: int = 75,
        exposure: float = 20,
        min_volts: float = -0.6,
        max_volts: float = 0.6,
    ):
        """
        Collect a uniform grid of raman spectra.

        Parameters
        ----------
        N : int
            Length of the grid sides
        exposure : float
            exposure time in ms
        min_volts, max_volts :
            The max voltage to apply to the galvos

        Returns
        -------
        spectra : array
            shape (N, N, 1340)
        volts : 1D array
            volts for passing to laser-pointer
        """
        xy_grid, volts = make_grid(N, min_volts, max_volts, stacked=True)
        return (
            self.collect_spectra(xy_grid, exposure).squeeze().reshape(N, N, 1340),
            volts,
        )

    def close(self):
        """Properly close open resources"""
        self._daq_controller.close()
        self._experiment.Stop()
        self._auto.Dispose()
