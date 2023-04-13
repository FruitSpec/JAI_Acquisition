#include "jaized.hpp"

#include <utility>

// Type wrappers

EnumeratedJAIFrameWrapper::EnumeratedJAIFrameWrapper(EnumeratedJAIFrame e_frame){
    this->e_frame = std::move(e_frame);
}

int EnumeratedJAIFrameWrapper::get_frame_number() const {
    return this->e_frame.BlockID;
}

py::array_t<uint8_t> EnumeratedJAIFrameWrapper::get_np_frame() {
    if (not this->np_frame) {
        int rows = this->e_frame.frame.rows;
        int cols = this->e_frame.frame.cols;
        int channels = this->e_frame.frame.channels();
        py::capsule free_when_done(this->e_frame.frame.data, [](void *f) { delete static_cast<cv::Mat *>(f); });
        this->np_frame = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.frame.data, free_when_done);
    }
    return this->np_frame;
};

// Function wrappers

py::tuple JaiZed::connect_cameras_wrapper(int fps, bool debug_mode) {
    acq_.debug = debug_mode;
    acq_.jz_streamer = JaiZedStream();
    JaiZedStatus jzs = connect_cameras(acq_, fps);
    py::tuple tpl = py::make_tuple(jzs.jai_connected, jzs.zed_connected);
    acq_.is_connected = jzs.jai_connected and jzs.zed_connected;
    return tpl;
}

void JaiZed::start_acquisition_wrapper(int fps, int exposure_rgb, int exposure_800, int exposure_975,
                                     const string& output_dir, bool output_fsi, bool output_rgb, bool output_800,
                                     bool output_975, bool output_svo, bool view, bool use_clahe_stretch,
                                     bool debug_mode) {
    cout << "starting" << endl;
    acq_.video_conf = parse_args(fps, exposure_rgb, exposure_800, exposure_975, output_dir, output_fsi, output_rgb,
                    output_800, output_975, output_svo, view, use_clahe_stretch, debug_mode);
    start_acquisition(acq_);
}

EnumeratedJAIFrameWrapper JaiZed::pop_wrapper(){
    cout << "popping" << endl;
    EnumeratedJAIFrame jai_frame = acq_.jz_streamer.pop_jai();
    EnumeratedJAIFrameWrapper e_frame(jai_frame);
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
    py::class_<EnumeratedJAIFrameWrapper>(m, "JaiFrame")
        .def(py::init<EnumeratedJAIFrameWrapper>())
        .def_property_readonly("frame", &EnumeratedJAIFrameWrapper::get_np_frame)
        .def_property_readonly("frame_number", &EnumeratedJAIFrameWrapper::get_frame_number);

    py::class_<JaiZed>(m, "JaiZed")
        .def(py::init<>())
        .def("connect_cameras", &JaiZed::connect_cameras_wrapper)
        .def("start_acquisition", &JaiZed::start_acquisition_wrapper)
        .def("pop", &JaiZed::pop_wrapper)
        .def("stop_acquisition", &JaiZed::stop_acquisition_wrapper)
        .def("disconnect_cameras", &JaiZed::disconnect_cameras_wrapper);
}