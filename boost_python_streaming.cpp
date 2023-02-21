#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
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
#include <stdio.h>
#include <thread>
#include <sl/Camera.hpp>
#include <fstream>
#include <mntent.h>
#include <dirent.h>
#include <boost/python.hpp>

using namespace cv;
using namespace cuda;
using namespace sl;
using namespace std;
using namespace boost::python;

// Start the acquisition process
void startAcquisition() {
    // Your code to start the acquisition process goes here
}

// Stop the acquisition process
void stopAcquisition() {
    // Your code to stop the acquisition process goes here
}

// Retrieve a stream of merged frames
object getMergedFrames() {
    // Your code to retrieve the merged frames goes here
    // Pack the frames into a Python list and return it
    list frames;
    return frames;
}

// Define the Python module and functions
BOOST_PYTHON_MODULE(my_module) {
    def("start_acquisition", startAcquisition);
    def("stop_acquisition", stopAcquisition);
    def("get_merged_frames", getMergedFrames);
}


#include <boost/python.hpp>
#include <thread>
#include <opencv2/opencv.hpp>

using namespace boost::python;
using namespace cv;

struct AcquisitionParameters {
    StreamInfo *MyStreamInfos[3];
    PvDevice *lDevice;
    BufferList *lBufferLists[3];
    thread *jai_t0, *jai_t1, *jai_t2, *zed_t, *merge_t;
    VideoConfig *video_conf;
    pthread_cond_t *GrabEvent;
    pthread_cond_t *MergeFramesEvent[3];
    bool is_running;
};

// Start the acquisition process
void startAcquisition(Acquisition& acq) {
    PvDevice *lDevice = NULL;
    PvStream *lStreams[3] = {NULL, NULL, NULL};
    BufferList lBufferLists[3];
    StreamInfo *MyStreamInfos[3];
    pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER, MergeFramesEvent[3];
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t tmp_mtx = PTHREAD_MUTEX_INITIALIZER;

    VideoConfig *video_conf;
    thread zed_t, jai_t0, jai_t1, jai_t2, merge_t;
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStart;
    Camera zed;
    VideoWriter mp4_FSI, mp4_BGR, mp4_800, mp4_975;
    char log_filename[100];
    int file_index;
    queue < EnumeratedFrame * > *Frames[3] = {&(MyStreamInfos[0]->Frames), &(MyStreamInfos[1]->Frames),
                                              &(MyStreamInfos[2]->Frames)};

    acq.MyStreamInfos = MyStreamInfos;
    acq.lDevice = &lDevice;
    acq.lBufferLists = lBufferLists;
    acq.jai_t0 = &jai_t0;
    acq.jai_t1 = &jai_t1;
    acq.jai_t2 = &jai_t2;
    acq.zed_t = &zed_t;
    acq.merge_t = &merge_t;
    acq.video_conf = &video_conf;
    acq.GrabEvent = &GrabEvent;
    acq.MergeFramesEvent = MergeFramesEvent;


    // check if this command is necessary
    PV_SAMPLE_INIT();

    pthread_cond_init(&GrabEvent, NULL)
    video_conf = parse_args(argc, argv);

    if (!setup_JAI(MyStreamInfos, lDevice, lBufferLists, video_conf, MergeFramesEvent))
        exit(-1);
    _abort = false;

    MP4CreateFirstTime(video_conf);

    sprintf(log_filename, (video_conf->output_dir + string("/frame_drop_%d.log")).c_str(), file_index);
    outfile.open(log_filename, std::ios_base::app); // append instead of overwrite

    // Enable streaming and send the AcquisitionStart command
    setup_ZED(zed, file_index);

    // Get device parameters need to control streaming - set acquisition start command
    lDeviceParams = lDevice->GetParameters();
    lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));

    zed_t = thread(ZedThread, zed, video_conf->output_svo);
    lStart->Execute();
    jai_t0 = thread(GrabThread, (void *) MyStreamInfos[0], &tmp_mtx);
    jai_t1 = thread(GrabThread, (void *) MyStreamInfos[1], &tmp_mtx);
    jai_t2 = thread(GrabThread, (void *) MyStreamInfos[2], &tmp_mtx);
    merge_t = thread(MergeThread, (void *) Frames, &mtx);

}

// Stop the acquisition process
void stopAcquisition(AcquisitionParameters &acq) {
    acq.is_running = false;
}

// Define the Python module and functions
BOOST_PYTHON_MODULE(my_module) {
    class_<Acquisition>("Acquisition")
        .def("start_acquisition", &startAcquisition)
        .def("stop_acquisition", &stopAcquisition);
}
