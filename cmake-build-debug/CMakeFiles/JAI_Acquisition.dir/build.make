# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mic-730ai/fruitspec/JAI_Acquisition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/JAI_Acquisition.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/JAI_Acquisition.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/JAI_Acquisition.dir/flags.make

CMakeFiles/JAI_Acquisition.dir/main.cpp.o: CMakeFiles/JAI_Acquisition.dir/flags.make
CMakeFiles/JAI_Acquisition.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/JAI_Acquisition.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/JAI_Acquisition.dir/main.cpp.o -c /home/mic-730ai/fruitspec/JAI_Acquisition/main.cpp

CMakeFiles/JAI_Acquisition.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/JAI_Acquisition.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mic-730ai/fruitspec/JAI_Acquisition/main.cpp > CMakeFiles/JAI_Acquisition.dir/main.cpp.i

CMakeFiles/JAI_Acquisition.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/JAI_Acquisition.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mic-730ai/fruitspec/JAI_Acquisition/main.cpp -o CMakeFiles/JAI_Acquisition.dir/main.cpp.s

CMakeFiles/JAI_Acquisition.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/JAI_Acquisition.dir/main.cpp.o.requires

CMakeFiles/JAI_Acquisition.dir/main.cpp.o.provides: CMakeFiles/JAI_Acquisition.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/JAI_Acquisition.dir/build.make CMakeFiles/JAI_Acquisition.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/JAI_Acquisition.dir/main.cpp.o.provides

CMakeFiles/JAI_Acquisition.dir/main.cpp.o.provides.build: CMakeFiles/JAI_Acquisition.dir/main.cpp.o


# Object files for target JAI_Acquisition
JAI_Acquisition_OBJECTS = \
"CMakeFiles/JAI_Acquisition.dir/main.cpp.o"

# External object files for target JAI_Acquisition
JAI_Acquisition_EXTERNAL_OBJECTS =

JAI_Acquisition: CMakeFiles/JAI_Acquisition.dir/main.cpp.o
JAI_Acquisition: CMakeFiles/JAI_Acquisition.dir/build.make
JAI_Acquisition: /usr/local/zed/lib/libsl_zed.so
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopenblas.so
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libusb-1.0.so
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libcuda.so
JAI_Acquisition: /usr/local/cuda-10.2/lib64/libcudart.so
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_gapi.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_barcode.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudabgsegm.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudafeatures2d.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudaobjdetect.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudastereo.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_sfm.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_xfeatures2d.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.6.0
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvAppUtils.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvBase.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvBuffer.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvCameraBridge.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvDevice.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvGUI.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvGenICam.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvPersistence.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvSerial.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvStream.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvSystem.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvTransmitter.so
JAI_Acquisition: /opt/jai/ebus_sdk/linux-aarch64-arm/lib/libPvVirtualDevice.so
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudacodec.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudaoptflow.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudalegacy.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudawarping.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudaimgproc.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudafilters.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudaarithm.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0
JAI_Acquisition: /usr/lib/aarch64-linux-gnu/libopencv_cudev.so.4.6.0
JAI_Acquisition: CMakeFiles/JAI_Acquisition.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable JAI_Acquisition"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/JAI_Acquisition.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/JAI_Acquisition.dir/build: JAI_Acquisition

.PHONY : CMakeFiles/JAI_Acquisition.dir/build

CMakeFiles/JAI_Acquisition.dir/requires: CMakeFiles/JAI_Acquisition.dir/main.cpp.o.requires

.PHONY : CMakeFiles/JAI_Acquisition.dir/requires

CMakeFiles/JAI_Acquisition.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/JAI_Acquisition.dir/cmake_clean.cmake
.PHONY : CMakeFiles/JAI_Acquisition.dir/clean

CMakeFiles/JAI_Acquisition.dir/depend:
	cd /home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mic-730ai/fruitspec/JAI_Acquisition /home/mic-730ai/fruitspec/JAI_Acquisition /home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug /home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug /home/mic-730ai/fruitspec/JAI_Acquisition/cmake-build-debug/CMakeFiles/JAI_Acquisition.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/JAI_Acquisition.dir/depend

