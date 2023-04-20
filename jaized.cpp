#include "jaized.hpp"

#include <utility>

// Type wrappers

EnumeratedJAIFrameWrapper::EnumeratedJAIFrameWrapper(EnumeratedJAIFrame e_frame){
    this->e_frame = std::move(e_frame);
    this->retrieved_frame = false;
}

int EnumeratedJAIFrameWrapper::get_frame_number() const {
    return this->e_frame.BlockID;
}

py::array_t<uint8_t> EnumeratedJAIFrameWrapper::get_np_frame() {
    if (not this->retrieved_frame) {
        cout << "JAI 1" << endl;
        this->retrieved_frame = true;
        int rows = this->e_frame.frame.rows;
        int cols = this->e_frame.frame.cols;
        int channels = this->e_frame.frame.channels();
        cout << "POP JAI PARAMS: " << rows << ", " << cols << ", " << channels << endl;
        this->np_frame = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.frame.data);
    }
    return this->np_frame;
}

EnumeratedZEDFrameWrapper::EnumeratedZEDFrameWrapper(EnumeratedZEDFrame& e_frame){
    this->e_frame = std::move(e_frame);
}

int EnumeratedZEDFrameWrapper::get_frame_number() const {
    return this->e_frame.BlockID;
}

py::array_t<uint8_t> EnumeratedZEDFrameWrapper::get_np_frame(){
    if (not this->retrieved_frame) {
        this->retrieved_frame = true;
        int rows = this->e_frame.frame.getHeight();
        int cols = this->e_frame.frame.getWidth();
        int channels = this->e_frame.frame.getChannels();
        cout << "POP ZED PARAMS: " << rows << ", " << cols << ", " << channels << endl;
        this->np_frame = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.frame.getPtr<uint8_t>());
    }
    return this->np_frame;
}

IMUData EnumeratedZEDFrameWrapper::get_imu_data() {
    return this->e_frame.imu;
}


// Function wrappers

py::tuple JaiZed::connect_cameras_wrapper(short fps, bool debug_mode) {
    acq_.debug = debug_mode;
    acq_.jz_streamer = JaiZedStream();
    JaiZedStatus jzs = connect_cameras(acq_, fps);
    py::tuple tpl = py::make_tuple(jzs.jai_connected, jzs.zed_connected);
    acq_.is_connected = jzs.jai_connected and jzs.zed_connected;
    return tpl;
}

void JaiZed::start_acquisition_wrapper(short fps, short exposure_rgb, short exposure_800, short exposure_975,
                                     const string& output_dir, bool output_fsi, bool output_rgb, bool output_800,
                                     bool output_975, bool output_svo, bool view, bool use_clahe_stretch,
                                     bool debug_mode) {
    cout << "starting" << endl;
    acq_.video_conf = parse_args(fps, exposure_rgb, exposure_800, exposure_975, output_dir, output_fsi, output_rgb,
                    output_800, output_975, output_svo, view, use_clahe_stretch, debug_mode);
    start_acquisition(acq_);
}

EnumeratedJAIFrameWrapper JaiZed::pop_jai_wrapper(){
    EnumeratedJAIFrame jai_frame = acq_.jz_streamer.pop_jai();
    EnumeratedJAIFrameWrapper e_frame(jai_frame);
    return e_frame;
}

EnumeratedZEDFrameWrapper JaiZed::pop_zed_wrapper(){
    EnumeratedZEDFrame zed_frame = acq_.jz_streamer.pop_zed();
    EnumeratedZEDFrameWrapper e_frame(zed_frame);
    return e_frame;
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

    py::class_<EnumeratedZEDFrameWrapper>(m, "ZedFrame")
        .def(py::init<EnumeratedZEDFrameWrapper>())
        .def_property_readonly("frame", &EnumeratedZEDFrameWrapper::get_np_frame)
        .def_property_readonly("imu", &EnumeratedZEDFrameWrapper::get_imu_data)
        .def_property_readonly("frame_number", &EnumeratedZEDFrameWrapper::get_frame_number);

    py::class_<IMUData>(m, "IMUData")
        .def(py::init<IMUData>())
        .def_readonly("angular_velocity", &IMUData::angular_velocity)
        .def_readonly("linear_acceleration", &IMUData::linear_acceleration);

    py::class_<JaiZed>(m, "JaiZed")
        .def(py::init<>())
        .def("connect_cameras", &JaiZed::connect_cameras_wrapper)
        .def("start_acquisition", &JaiZed::start_acquisition_wrapper)
        .def("pop_jai", &JaiZed::pop_jai_wrapper)
        .def("pop_zed", &JaiZed::pop_zed_wrapper)
        .def("stop_acquisition", &JaiZed::stop_acquisition_wrapper)
        .def("disconnect_cameras", &JaiZed::disconnect_cameras_wrapper);
}