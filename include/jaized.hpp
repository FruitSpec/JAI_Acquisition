#ifndef JAIZED_HPP
#define JAIZED_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include "acquisition.hpp"
#include <string>

namespace py = pybind11;
typedef sl::SensorsData::IMUData IMUData;

class EnumeratedJAIFrameWrapper {

public:
    EnumeratedJAIFrameWrapper(EnumeratedJAIFrame e_frame);

    int get_frame_number() const;

    py::array_t<uint8_t> get_np_frame();

private:
    EnumeratedJAIFrame e_frame;
    py::array_t<uint8_t> np_frame;
    bool retrieved_frame;
};

class EnumeratedZEDFrameWrapper {

public:
    EnumeratedZEDFrameWrapper(EnumeratedZEDFrame& e_frame);

    int get_frame_number() const;

    py::array_t<uint8_t> get_np_frame();

    SensorsData::IMUData get_imu_data();

private:
    EnumeratedZEDFrame e_frame;
    py::array_t<uint8_t> np_frame;
    bool retrieved_frame;
};

class JaiZed {
public:

    py::tuple connect_cameras_wrapper(short fps, bool debug_mode);

    void start_acquisition_wrapper(short fps, short exposure_rgb, short exposure_800, short exposure_975, const string& output_dir,
                                 bool output_fsi, bool output_rgb, bool output_800, bool output_975, bool output_svo,
                                 bool view, bool use_clahe_stretch, bool debug_mode);

    EnumeratedJAIFrameWrapper pop_jai_wrapper();

    EnumeratedZEDFrameWrapper pop_zed_wrapper();

    void stop_acquisition_wrapper();

    void disconnect_cameras_wrapper();


private:
    static AcquisitionParameters acq_;
    static std::queue<py::array_t<uint8_t>> jai_frames;
    static std::queue<py::array_t<uint8_t>> zed_frames;
};

#endif // JAIZED_HPP
