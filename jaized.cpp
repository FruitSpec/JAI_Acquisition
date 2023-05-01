#include "jaized.hpp"

#include <utility>

// Type wrappers

EnumeratedJAIFrameWrapper::EnumeratedJAIFrameWrapper(EnumeratedJAIFrame e_frame){
    this->e_frame = std::move(e_frame);
}

int EnumeratedJAIFrameWrapper::get_frame_number() const {
    return this->e_frame.BlockID;
}

py::str EnumeratedJAIFrameWrapper::get_timestamp() const{
    return this->e_frame.timestamp;
}


py::array_t<uint8_t> EnumeratedJAIFrameWrapper::get_fsi_np_frame() {
    if (not this->retrieved_fsi) {
        this->retrieved_fsi = true;
        int rows = this->e_frame.fsi_frame.rows;
        int cols = this->e_frame.fsi_frame.cols;
        int channels = this->e_frame.fsi_frame.channels();
        this->np_fsi_frame = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.fsi_frame.data);
    }
    return this->np_fsi_frame;
}

py::array_t<uint8_t> EnumeratedJAIFrameWrapper::get_rgb_np_frame() {
    if (not this->retrieved_rgb) {
        this->retrieved_rgb = true;
        int rows = this->e_frame.rgb_frame.rows;
        int cols = this->e_frame.rgb_frame.cols;
        int channels = this->e_frame.rgb_frame.channels();
        this->np_rgb_frame = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.rgb_frame.data);
    }
    return this->np_rgb_frame;
}

EnumeratedZEDFrameWrapper::EnumeratedZEDFrameWrapper(EnumeratedZEDFrame& e_frame){
    this->e_frame = std::move(e_frame);
}

int EnumeratedZEDFrameWrapper::get_frame_number() const {
    return this->e_frame.BlockID;
}

py::str EnumeratedZEDFrameWrapper::get_timestamp() const {
    return this->e_frame.timestamp;
}

py::array_t<uint8_t> EnumeratedZEDFrameWrapper::get_np_point_cloud(){
    if (not this->retrieved_point_cloud) {
        this->retrieved_point_cloud = true;
        int rows = this->e_frame.point_cloud.getHeight();
        int cols = this->e_frame.point_cloud.getWidth();
        int channels = this->e_frame.point_cloud.getChannels();
        cout << "POP ZED PARAMS: " << rows << ", " << cols << ", " << channels << endl;
        this->np_point_cloud = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.point_cloud.getPtr<uint8_t>());
    }
    return this->np_point_cloud;
}

py::array_t<uint8_t> EnumeratedZEDFrameWrapper::get_np_rgb(){
    if (not this->retrieved_rgb) {
        this->retrieved_rgb = true;
        int rows = this->e_frame.rgb.getHeight();
        int cols = this->e_frame.rgb.getWidth();
        int channels = this->e_frame.rgb.getChannels();
        cout << "POP ZED PARAMS: " << rows << ", " << cols << ", " << channels << endl;
        this->np_rgb = py::array_t<uint8_t>({rows, cols, channels}, this->e_frame.rgb.getPtr<uint8_t>());
    }
    return this->np_rgb;
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
                                       const string& output_dir, bool output_clahe_fsi, bool output_equalize_hist_fsi,
                                       bool output_rgb, bool output_800, bool output_975, bool output_svo, bool view,
                                       bool pass_clahe_stream, bool debug_mode) {
    cout << "starting" << endl;
    acq_.video_conf = parse_args(fps, exposure_rgb, exposure_800, exposure_975,
                                 output_dir, output_clahe_fsi, output_equalize_hist_fsi, output_rgb,
                                 output_800, output_975, output_svo, view, pass_clahe_stream, debug_mode);
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
            .def_property_readonly("timestamp", &EnumeratedJAIFrameWrapper::get_timestamp)
        .def_property_readonly("fsi", &EnumeratedJAIFrameWrapper::get_fsi_np_frame)
        .def_property_readonly("rgb", &EnumeratedJAIFrameWrapper::get_rgb_np_frame)
        .def_property_readonly("frame_number", &EnumeratedJAIFrameWrapper::get_frame_number);

    py::class_<EnumeratedZEDFrameWrapper>(m, "ZedFrame")
            .def(py::init<EnumeratedZEDFrameWrapper>())
            .def_property_readonly("timestamp", &EnumeratedZEDFrameWrapper::get_timestamp)
            .def_property_readonly("rgb", &EnumeratedZEDFrameWrapper::get_np_rgb)
            .def_property_readonly("point_cloud", &EnumeratedZEDFrameWrapper::get_np_point_cloud)
            .def_property_readonly("imu", &EnumeratedZEDFrameWrapper::get_imu_data)
            .def_property_readonly("frame_number", &EnumeratedZEDFrameWrapper::get_frame_number);

    py::class_<IMUData>(m, "IMUData")
        .def(py::init<IMUData>())
        .def_readonly("angular_velocity", &IMUData::angular_velocity)
        .def_readonly("linear_acceleration", &IMUData::linear_acceleration);

    py::class_<sl::float3>(m, "xyzVector")
            .def(py::init<sl::float3>())
            .def_readonly("x", &sl::float3::x)
            .def_readonly("y", &sl::float3::y)
            .def_readonly("z", &sl::float3::z);

    py::class_<JaiZed>(m, "JaiZed")
        .def(py::init<>())
        .def("connect_cameras", &JaiZed::connect_cameras_wrapper)
        .def("start_acquisition", &JaiZed::start_acquisition_wrapper)
        .def("pop_jai", &JaiZed::pop_jai_wrapper)
        .def("pop_zed", &JaiZed::pop_zed_wrapper)
        .def("stop_acquisition", &JaiZed::stop_acquisition_wrapper)
        .def("disconnect_cameras", &JaiZed::disconnect_cameras_wrapper);
}