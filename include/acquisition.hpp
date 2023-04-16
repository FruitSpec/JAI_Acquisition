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
#include <list>
#include <PvDisplayWnd.h>
#include <map>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <thread>
#include <sl/Camera.hpp>
#include <fstream>
#include <mntent.h>
#include <dirent.h>
#include "streamer.hpp"


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
    queue<EnumeratedJAIFrame *> Frames;
};

struct VideoConfig {
    short bit_depth = 8;
    short FPS = 15, exposure_rgb = 1000, exposure_800 = 2000, exposure_975 = 4000;
    short file_index = -1;
    int64_t width = 1536, height = 2048;
    bool output_fsi = false, output_rgb = false, output_800 = false, output_975 = false, output_svo = false;
    bool view = false;
    bool use_clahe_stretch = false;
    string output_dir = string("/home/mic-730ai/Desktop/JAI_Results");
};

struct AcquisitionParameters {
    StreamInfo *MyStreamInfos[3];
    PvDevice *lDevice = nullptr;
    BufferList lBufferLists[3];
    thread jai_t0, jai_t1, jai_t2, zed_t, merge_t;
    Camera zed;
    VideoConfig *video_conf;
    pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER;
    pthread_cond_t MergeFramesEvent[3];
    pthread_mutex_t grab_mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t acq_start_mtx = PTHREAD_MUTEX_INITIALIZER;
    VideoWriter mp4_FSI, mp4_BGR, mp4_800, mp4_975;
    bool is_connected, is_running, debug;
    ofstream frame_drop_log_file;
    JaiZedStream jz_streamer;
};

bool SelectDeviceLocally(PvString *aConnectionID);

PvDevice *ConnectToDevice(const PvString &aConnectionID, bool debug);

PvStream *OpenStream(const PvString &aConnectionID, bool debug);

void ConfigureStream(PvDevice *&aDevice, PvStream *aStream, int channel);

void CreateStreamBuffers(PvDevice *&aDevice, PvStream *aStream, BufferList *aBufferList);

void FreeStreamBuffers(BufferList *aBufferList);

VideoConfig * parse_args(short bit_depth, short fps, short exposure_rgb, short exposure_800, short exposure_975, const string& output_dir,
                         bool output_fsi, bool output_rgb, bool output_800, bool output_975, bool output_svo, bool view,
                         bool use_clahe_stretch, bool debug_mode);

bool setup_JAI(AcquisitionParameters &acq);

void MP4CreateFirstTime(AcquisitionParameters &acq);

string gs_sink_builder(int file_index, const string& output_type_name, const string& output_dir);

bool exists(char path[100]);

bool connect_ZED(AcquisitionParameters &acq, int fps);

void GrabThread(int stream_index, AcquisitionParameters &acq);

void ZedThread(AcquisitionParameters &acq);

void MergeThread(AcquisitionParameters &acq);

JaiZedStatus connect_cameras(AcquisitionParameters &acq, int fps);

void start_acquisition(AcquisitionParameters &acq);

void stop_acquisition(AcquisitionParameters &acq);

void disconnect_cameras(AcquisitionParameters &acq);

#ifndef JAI_ACQUISITION_MAIN_H
#define JAI_ACQUISITION_MAIN_H

#endif //JAI_ACQUISITION_MAIN_H
