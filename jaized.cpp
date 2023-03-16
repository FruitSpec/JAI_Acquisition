#include "jaized.hpp"


namespace py = pybind11;

bool JaiZed::connect_cameras_wrapper(int fps, int exposure_rgb, int exposure_800, int exposure_975,
                                     string output_dir, bool output_fsi, bool output_rgb, bool output_800,
                                     bool output_975, bool output_svo, bool view, bool use_clahe_stretch,
                                     bool debug_mode) {
    acq_.video_conf = parse_args(fps, exposure_rgb, exposure_800, exposure_975, output_dir, output_fsi, output_rgb,
                    output_800, output_975, output_svo, view, use_clahe_stretch, debug_mode);
    connected = connect_cameras(acq_);
    return connected;
}

bool JaiZed::start_acquisition_wrapper() {
    if (connected) {
        started = start_acquisition(acq_);
        return started;
    }
    return false;
}

bool JaiZed::stop_acquisition_wrapper() {
    if (connected) {
        stop_acquisition(acq_);
        started = false;
        return true;
    }
    return false;
}

bool JaiZed::disconnect_cameras_wrapper() {
    if (connected and not started) {
        disconnect_cameras(acq_);
        return true;
    }
    return false;
}

AcquisitionParameters JaiZed::acq_;
bool JaiZed::connected = false;
bool JaiZed::started = false;

PYBIND11_MODULE(jaized, m) {
    py::class_<JaiZed>(m, "JaiZed")
    .def(py::init<>())
    .def("connect_cameras", &JaiZed::connect_cameras_wrapper)
    .def("start_acquisition", &JaiZed::start_acquisition_wrapper)
    .def("stop_acquisition", &JaiZed::stop_acquisition_wrapper)
    .def("disconnect_cameras", &JaiZed::disconnect_cameras_wrapper);
}