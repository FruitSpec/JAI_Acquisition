#include "jaized.hpp"


namespace py = pybind11;

py::tuple JaiZed::connect_cameras_wrapper(int fps, bool debug_mode) {
    acq_.debug = debug_mode;
    acq_.batcher = BatchQueue();
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

py::tuple JaiZed::pop_wrapper(){
    cout << "popping" << endl;
    FramesBatch fb = acq_.batcher.pop();
    py::tuple npFrames(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        cv::Mat frame = fb.jai_frames[i];
        // Create a numpy array with a capsule to the cv::Mat object's data
        int rows = frame.rows;
        int cols = frame.cols;
        int channels = frame.channels();
        py::capsule free_when_done(frame.data, [](void *f) { delete static_cast<cv::Mat*>(f); });
        py::array_t<uint8_t> npFrame({ rows, cols, channels }, frame.data, free_when_done);

        // Set the correct numpy array flags
        npFrames[i] = npFrame;
    }
    cout << 99 << endl;
    return npFrames;
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
    .def("pop", &JaiZed::pop_wrapper)
    .def("stop_acquisition", &JaiZed::stop_acquisition_wrapper)
    .def("disconnect_cameras", &JaiZed::disconnect_cameras_wrapper);
}