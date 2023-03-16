#ifndef JAIZED_HPP
#define JAIZED_HPP

#include <pybind11/pybind11.h>
#include "acquisition.hpp"
#include <string>

class JaiZed {
public:

    bool connect_cameras_wrapper(int fps, int exposure_rgb, int exposure_800, int exposure_975, string output_dir,
                                 bool output_fsi, bool output_rgb, bool output_800, bool output_975, bool output_svo,
                                 bool view, bool use_clahe_stretch, bool debug_mode);

    bool start_acquisition_wrapper();

    bool stop_acquisition_wrapper();

    bool disconnect_cameras_wrapper();

private:
    static AcquisitionParameters acq_;
    static bool connected, started;
};

#endif // JAIZED_HPP
