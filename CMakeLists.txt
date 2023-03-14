cmake_minimum_required(VERSION 3.0)
project (JAI_Acquisition)
set(CMAKE_CXX_STANDARD 17)

ADD_SUBDIRECTORY(pybind11)
pybind11_add_module(jaized jaized.cpp acquisition.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_definitions(-D_UNIX_=1)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# FindeBus.cmake - Find eBus sdk, version >= 4.
# quchao@seas.upenn.edu (Chao Qu)
# Modified from FindEigen.cmake by alexs.mac@gmail.com  (Alex Stewart)
#
# This module defines the following variables:
#
# eBus_FOUND:        TRUE if eBus is found.
# eBus_INCLUDE_DIRS: Include directories for eBus.
# eBus_LIBRARIES:    Libraries for all eBus component libraries
#                    and dependencies.
#
# eBus_VERSION: Extracted from lib/PvBase.so.x.y.z
# eBus_WORLD_VERSION: Equal to 4 if eBus_VERSION = 4.0.5
# eBus_MAJOR_VERSION: Equal to 0 if eBus_VERSION = 4.0.5
# eBus_MINOR_VERSION: Equal to 5 if eBus_VERSION = 4.0.5
#
# The following variables control the behaviour of this module:
#
# eBus_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for eBus includes, e.g: /foo/include.
# eBus_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for eBus libraries, e.g: /bar/lib.
#
# The following variables are also defined by this module, but in line with
# CMake recommended FindPackage() module style should NOT be referenced directly
# by callers (use the plural variables detailed above instead).  These variables
# do however affect the behaviour of the module via FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# eBus_INCLUDE_DIR: Include directory for eBus, not including the
#                    include directory of any dependencies.
# eBus_LIBRARY: eBus library, not including the libraries of any
#                dependencies.

# Called if we failed to find eBus or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(eBus_REPORT_NOT_FOUND REASON_MSG)
    unset(eBus_FOUND)
    unset(eBus_INCLUDE_DIRS)
    unset(eBus_LIBRARIES)
    unset(eBus_WORLD_VERSION)
    unset(eBus_MAJOR_VERSION)
    unset(eBus_MINOR_VERSION)
    # Make results of search visible in the CMake GUI if eBus has not
    # been found so that user does not have to toggle to advanced view.
    mark_as_advanced(CLEAR eBus_INCLUDE_DIR)
    # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
    # use the camelcase library name, not uppercase.
    if(eBus_FIND_QUIETLY)
        message(STATUS "Failed to find eBus - " ${REASON_MSG} ${ARGN})
    elseif(eBus_FIND_REQUIRED)
        message(FATAL_ERROR "Failed to find eBus - " ${REASON_MSG} ${ARGN})
    else()
        # Neither QUIETLY nor REQUIRED, use no priority which emits a message
        # but continues configuration and allows generation.
        message("-- Failed to find eBus - " ${REASON_MSG} ${ARGN})
    endif()
endmacro(eBus_REPORT_NOT_FOUND)

# Search user-installed locations first, so that we prefer user installs
# to system installs where both exist.
list(APPEND eBus_CHECK_INCLUDE_DIRS /opt/jai/ebus_sdk/linux-aarch64-arm/include)
list(APPEND eBus_CHECK_LIBRARY_DIRS /opt/jai/ebus_sdk/linux-aarch64-arm/lib)

# Search supplied hint directories first if supplied.
# Find include directory for eBus
find_path(eBus_INCLUDE_DIR
        NAMES PvBase.h
        PATHS ${eBus_INCLUDE_DIR_HINTS}
        ${eBus_CHECK_INCLUDE_DIRS}
        NO_DEFAULT_PATH)
if(NOT eBus_INCLUDE_DIR OR NOT EXISTS ${eBus_INCLUDE_DIR})
    eBus_REPORT_NOT_FOUND(
            "Could not find eBus include directory, set eBus_INCLUDE_DIR to "
            "path to eBus include directory,"
            "e.g. /opt/jai/ebus_sdk/linux-aarch64-arm/include.")
else()
    message(STATUS "eBus include dir found: " ${eBus_INCLUDE_DIR})
endif()

# Find library directory for eBus
find_library(eBus_LIBRARY
        NAMES PvBase
        PATHS ${eBus_LIBRARY_DIR_HINTS}
        ${eBus_CHECK_LIBRARY_DIRS}
        NO_DEFAULT_PATH)
if(NOT eBus_LIBRARY OR NOT EXISTS ${eBus_LIBRARY})
    eBus_REPORT_NOT_FOUND(
            "Could not find eBus library, set eBus_LIBRARY "
            "to full path to eBus library direcotory.")
else()
    string(REGEX MATCH ".*/" eBus_LIBRARY_DIR ${eBus_LIBRARY})
endif()

# Mark internally as found, then verify. eBus_REPORT_NOT_FOUND() unsets if
# called.
set(eBus_FOUND TRUE)

# Extract eBus version from ebus_sdk/linux-aarch64-arm/lib/libPvBase.so.x.y.z
if(eBus_LIBRARY_DIR)
    file(GLOB eBus_PVBASE
            RELATIVE ${eBus_LIBRARY_DIR}
            ${eBus_LIBRARY_DIR}/libPvBase.so.[0-9].[0-9].[0-9])
    # string(REGEX MATCH ""
    #       eBus_WORLD_VERSION ${eBus_PVBASE})
    # message(STATUS "eBus world version: " ${eBus_WORLD_VERSION})
endif()

# Catch case when caller has set eBus_INCLUDE_DIR in the cache / GUI and
# thus FIND_[PATH/LIBRARY] are not called, but specified locations are
# invalid, otherwise we would report the library as found.
if(eBus_INCLUDE_DIR AND NOT EXISTS ${eBus_INCLUDE_DIR}/PvBase.h)
    eBus_REPORT_NOT_FOUND("Caller defined eBus_INCLUDE_DIR: "
            ${eBus_INCLUDE_DIR}
            " does not contain PvBase.h header.")
endif()

# Set standard CMake FindPackage variables if found.
if(eBus_FOUND)
    set(eBus_INCLUDE_DIRS ${eBus_INCLUDE_DIR})
    file(GLOB eBus_LIBRARIES ${eBus_LIBRARY_DIR}libPv*.so)
endif()

# Handle REQUIRED / QUIET optional arguments.
include(FindPackageHandleStandardArgs)
if(eBus_FOUND)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(eBus DEFAULT_MSG
            eBus_INCLUDE_DIRS eBus_LIBRARIES)
endif()

# Only mark internal variables as advanced if we found eBus, otherwise
# leave it visible in the standard GUI for the user to set manually.
if(eBus_FOUND)
    mark_as_advanced(FORCE eBus_INCLUDE_DIR eBus_LIBRARY)
endif()

OPTION(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED 3 REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)

INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(/opt/jai/ebus_sdk/linux-aarch64-arm/include)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${ZED_INCLUDE_DIRS})

LINK_DIRECTORIES(${ZED_LIBRARY_DIR})
LINK_DIRECTORIES(${CUDA_LIBRARY_DIRS})

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

TARGET_LINK_LIBRARIES( jaized PRIVATE ${ZED_LIBS} ${OpenCV_LIBS} ${eBus_LIBRARIES} )