#include "jaized.hpp"


namespace py = pybind11;

py::tuple JaiZed::connect_cameras_wrapper(int fps, bool debug_mode) {
    acq_.debug = debug_mode;
    JaiZedStatus jzs = connect_cameras(acq_, fps);
    py::tuple tpl = py::make_tuple(jzs.jai_connected, jzs.zed_connected);
    acq_.is_connected = jzs.jai_connected and jzs.zed_connected;
    return tpl;
}

void JaiZed::start_acquisition_wrapper(int fps, int exposure_rgb, int exposure_800, int exposure_975,
                                     const string& output_dir, bool output_fsi, bool output_rgb, bool output_800,
                                     bool output_975, bool output_svo, bool view, bool use_clahe_stretch,
                                     bool debug_mode) {
    acq_.video_conf = parse_args(fps, exposure_rgb, exposure_800, exposure_975, output_dir, output_fsi, output_rgb,
                    output_800, output_975, output_svo, view, use_clahe_stretch, debug_mode);
    start_acquisition(acq_);
}

void JaiZed::stop_acquisition_wrapper() {
    if (acq_.is_running) {
        stop_acquisition(acq_);
    }
}

void JaiZed::disconnect_cameras_wrapper() {
    if (acq_.is_connected and not acq_.is_running)
        disconnect_cameras(acq_);
}

AcquisitionParameters JaiZed::acq_;

PYBIND11_MODULE(jaized, m) {
    py::class_<JaiZed>(m, "JaiZed")
    .def(py::init<>())
    .def("connect_cameras", &JaiZed::connect_cameras_wrapper)
    .def("start_acquisition", &JaiZed::start_acquisition_wrapper)
    .def("stop_acquisition", &JaiZed::stop_acquisition_wrapper)
    .def("disconnect_cameras", &JaiZed::disconnect_cameras_wrapper);
}