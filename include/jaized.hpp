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

class JaiZed {
public:

    py::tuple connect_cameras_wrapper(int fps, bool debug_mode);

    void start_acquisition_wrapper(int fps, int exposure_rgb, int exposure_800, int exposure_975, const string& output_dir,
                                 bool output_fsi, bool output_rgb, bool output_800, bool output_975, bool output_svo,
                                 bool view, bool use_clahe_stretch, bool debug_mode);

    py::tuple pop_wrapper();

    void stop_acquisition_wrapper();

    void disconnect_cameras_wrapper();


private:
    static AcquisitionParameters acq_;
};

#endif // JAIZED_HPP
