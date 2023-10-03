#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudacodec.hpp>
#include <sys/stat.h>
#include <PvSampleUtils.h>
#include <PvDevice.h>
#include <PvDeviceGEV.h>
#include <PvDeviceU3V.h>
#include <PvStream.h>
#include <PvStreamGEV.h>
#include <PvStreamU3V.h>
#include <PvBuffer.h>
#include <PvBufferWriter.h>
#include <PvDisplayWnd.h>
#include <list>
#include <map>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <thread>
#include <atomic>
#include <memory>
#include <sl/Camera.hpp>
#include <fstream>
#include <mntent.h>
#include <dirent.h>
#include "streamer.hpp"

#define JAI_FPS 10
#define ZED_FPS 15


using namespace cv;
using namespace cuda;
using namespace sl;
using namespace std;

typedef list<PvBuffer *> BufferList;

struct JaiZedStatus {
    bool jai_connected;
    bool zed_connected;
};

struct StreamInfo {
    PvStream *aStream{};
    int stream_index{};
    queue<SingleJAIChannel *> Frames;
};

struct VideoConfig {
    short exposure_rgb = 1000, exposure_800 = 2000, exposure_975 = 4000;
    int64_t width = 1536, height = 2048;
    std::vector<string> alc_true_areas, alc_false_areas;
    bool transfer_data;
};

struct RecordingConfig {
    short file_index = -1;
    bool output_clahe_fsi, output_equalize_hist_fsi, output_rgb, output_800, output_975, output_svo;
    bool output_zed_gray, output_zed_depth, output_zed_pc;
    bool pass_clahe_stream;
    string output_dir = string("/home/mic-730ai/Desktop/JAI_Results");
};

struct AcquisitionParameters {
    VideoConfig *video_conf;
    RecordingConfig *record_conf;

    StreamInfo *MyStreamInfos[3];
    PvDevice *lDevice = nullptr;
    Camera zed;

    BufferList lBufferLists[3];
    thread jai_t0, jai_t1, jai_t2, zed_t, merge_t;

    pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER;
    pthread_cond_t MergeFramesEvent[3];
    pthread_mutex_t grab_mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t acq_start_mtx = PTHREAD_MUTEX_INITIALIZER;

    VideoWriter mp4_clahe_FSI, mp4_equalize_hist_FSI, mp4_BGR, mp4_800, mp4_975;
    VideoWriter mp4_zed_rgb, mp4_zed_depth, mp4_zed_X, mp4_zed_Y, mp4_zed_Z;
    ofstream frame_drop_log_file, imu_log_file;
    ofstream jai_acquisition_log;

    bool jai_connected, zed_connected, debug;
    std::atomic<bool> is_running;
    std::atomic<bool> is_recording;

    JaiZedStream jz_streamer;
};

bool SelectDeviceLocally(PvString *aConnectionID);

PvDevice *ConnectToDevice(const PvString &aConnectionID, bool debug);

PvStream *OpenStream(const PvString &aConnectionID, bool debug);

void ConfigureStream(PvDevice *&aDevice, PvStream *aStream, int channel);

void CreateStreamBuffers(PvDevice *&aDevice, PvStream *aStream, BufferList *aBufferList);

void FreeStreamBuffers(BufferList *aBufferList);

VideoConfig * setup_acquisition(short exposure_rgb, short exposure_800, short exposure_975,
                                bool transfer_data, std::vector<string> alc_true_areas,
                                std::vector<string> alc_false_areas, bool debug_mode);

RecordingConfig * setup_recording(const string& output_dir, bool output_clahe_fsi, bool output_equalize_hist_fsi,
                                bool output_rgb, bool output_800, bool output_975, bool output_svo,
                                bool output_zed_gray, bool output_zed_depth, bool output_zed_pc,
                                bool pass_clahe_stream);

void set_parameters_per_source(PvGenParameterArray *&lDeviceParams, const PvString& source, int auto_exposure_max,
                               const PvString &pixel_format);

void set_acquisition_parameters(AcquisitionParameters &acq);

bool setup_JAI(AcquisitionParameters &acq);

void MP4CreateFirstTime(AcquisitionParameters &acq);

string gs_sink_builder(const string& output_type_name, const string& output_dir);

bool connect_ZED(AcquisitionParameters &acq);

void GrabThread(int stream_index, AcquisitionParameters &acq);

void ZedThread(AcquisitionParameters &acq);

void MergeThread(AcquisitionParameters &acq);

string get_current_time();

JaiZedStatus connect_cameras(AcquisitionParameters &acq);

bool start_acquisition(AcquisitionParameters &acq);

bool start_recording(AcquisitionParameters &acq);

void stop_recording(AcquisitionParameters &acq);

void stop_acquisition(AcquisitionParameters &acq);

void disconnect_jai(AcquisitionParameters &acq);

void disconnect_zed(AcquisitionParameters &acq);

void disconnect_cameras(AcquisitionParameters &acq);

#ifndef JAI_ACQUISITION_MAIN_H
#define JAI_ACQUISITION_MAIN_H

#endif //JAI_ACQUISITION_MAIN_H
