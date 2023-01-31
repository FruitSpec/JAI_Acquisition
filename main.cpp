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

using namespace cv;
using namespace cuda;
using namespace sl;
using namespace std;

typedef list<PvBuffer *> BufferList;

typedef struct {
    cv::Mat frame;
    int BlockID;
} EnumeratedFrame;

typedef struct {
    PvStream *aStream;
    int StreamIndex;
    queue<EnumeratedFrame *> Frames;
} StreamInfo;

typedef struct {
    int FPS = 15, exposure_rgb = 500, exposure_800 = 1000, exposure_975 = 3000;
    int width = 1536, height = 2048;
    int file_index = -1;
    bool output_fsi = false, output_rgb = false, output_800 = false, output_975 = false, output_svo = false;
    bool view = false;
    string output_dir = string("/home/mic-730ai/Desktop/JAI_Results");
} VideoConfig;

typedef struct {
    StreamInfo *MyStreamInfos[3];
    PvDevice *lDevice;
    BufferList *lBufferLists[3];
    thread *jai_t0, *jai_t1, *jai_t2, *zed_t, *merge_t;
    VideoConfig *video_conf;
    pthread_cond_t *GrabEvent;
    pthread_cond_t *MergeFramesEvent[3];
} StreamingParameters;

PV_INIT_SIGNAL_HANDLER();

bool SelectDeviceLocally(PvString *aConnectionID);

PvDevice *ConnectToDevice(const PvString &aConnectionID);

PvStream *OpenStream(const PvString &aConnectionID);

void ConfigureStream(PvDevice *aDevice, PvStream *aStream, int channel);

void CreateStreamBuffers(PvDevice *aDevice, PvStream *aStream, BufferList *aBufferList);

void FreeStreamBuffers(BufferList *aBufferList);

void parse_args(int argc, char* argv[]);

bool setup_JAI(StreamInfo *MyStreamInfos[3], PvDevice *&lDevice, BufferList lBufferLists[3], VideoConfig video_conf);

int MP4CreateFirstTime(VideoConfig *&video_conf, VideoWriter *&mp4_FSI, VideoWriter *&mp4_RGB, VideoWriter *&mp4_800,
                       VideoWriter *&mp4_975);

bool exists(char path[100]);

void setup_ZED(Camera *&zed, VideoConfig video_conf);

void GrabThread(void *StreamInfo);

void ZedThread(Camera zed);

void MergeThread(void *_Frames);

StreamingParameters start_acquisition();

void stop_acquisition(StreamingParameters st_params);
ofstream outfile;

bool _abort = false;

/// Code starts here

bool SelectDeviceLocally(PvString *aConnectionID)
{
    PvResult lResult;
    const PvDeviceInfo *lSelectedDI = NULL;
    PvSystem lSystem;

        lSystem.Find();

        // Detect, select device
        vector<const PvDeviceInfo *> lDIVector;
        for ( uint32_t i = 0; i < lSystem.GetInterfaceCount(); i++ ) {
            const PvInterface *lInterface = dynamic_cast<const PvInterface *>( lSystem.GetInterface( i ) );
            if ( lInterface != NULL ) {
                for ( uint32_t j = 0; j < lInterface->GetDeviceCount(); j++ ) {
                    const PvDeviceInfo *lDI = dynamic_cast<const PvDeviceInfo *>( lInterface->GetDeviceInfo( j ) );
                    if ( lDI != NULL ){
                        lDIVector.push_back( lDI );
                    }
                }
            }
        }

        // Read device selection, optional new IP address.
        uint32_t lIndex = 0;
        if ( lIndex == lDIVector.size() )
            // Abort the selection process
            return false;
        else if ( lIndex < lDIVector.size() )
            // The device is selected
            lSelectedDI = lDIVector[ lIndex ];

    *aConnectionID = lSelectedDI->GetConnectionID();
    return true;
}

PvDevice *ConnectToDevice(const PvString &aConnectionID) {
    PvDevice *lDevice;
    PvResult lResult;

    // Connect to the GigE Vision or USB3 Vision device
    cout << "Connecting to device." << endl;
    lDevice = PvDevice::CreateAndConnect(aConnectionID, &lResult);
    if (lDevice == NULL) {
        cout << "Unable to connect to device: " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

    return lDevice;
}

PvStream *OpenStream(const PvString &aConnectionID) {
    PvStream *lStream;
    PvResult lResult;

    // Open stream to the GigE Vision or USB3 Vision device
    cout << "Opening stream from device." << endl;
    lStream = PvStream::CreateAndOpen(aConnectionID, &lResult);
    if (lStream == NULL) {
        cout << "Unable to stream from device. " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

    return lStream;
}

void ConfigureStream(PvDevice *aDevice, PvStream *aStream, int channel) {
    // If this is a GigE Vision device, configure GigE Vision specific streaming parameters
    PvDeviceGEV *lDeviceGEV = dynamic_cast<PvDeviceGEV *>( aDevice );
    if (lDeviceGEV != NULL) {
        PvStreamGEV *lStreamGEV = static_cast<PvStreamGEV *>( aStream );

        // Negotiate packet size
        lDeviceGEV->NegotiatePacketSize(channel);

        // Configure device streaming destination
        lDeviceGEV->SetStreamDestination(lStreamGEV->GetLocalIPAddress(), lStreamGEV->GetLocalPort(), channel);
    }
}

void CreateStreamBuffers(PvDevice *aDevice, PvStream *aStream, BufferList *aBufferList) {
    // Reading payload size from device
    uint32_t lSize = aDevice->GetPayloadSize();

    // Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
    uint32_t lBufferCount = aStream->GetQueuedBufferMaximum();

    // Allocate buffers
    for (uint32_t i = 0; i < lBufferCount; i++) {
        // Create new buffer object
        PvBuffer *lBuffer = new PvBuffer;

        // Have the new buffer object allocate payload memory
        lBuffer->Alloc(static_cast<uint32_t>( lSize ));

        // Add to external list - used to eventually release the buffers
        aBufferList->push_back(lBuffer);
    }

    // Queue all buffers in the stream
    BufferList::iterator lIt = aBufferList->begin();
    while (lIt != aBufferList->end()) {
        aStream->QueueBuffer(*lIt);
        lIt++;
    }
}

void FreeStreamBuffers(BufferList *aBufferList) {
    // Go through the buffer list
    BufferList::iterator lIt = aBufferList->begin();
    while (lIt != aBufferList->end()) {
        delete *lIt;
        lIt++;
    }

    // Clear the buffer list
    aBufferList->clear();
}

VideoConfig parse_args(int argc, char *argv[]){
    VideoConfig video_conf = new VideoConfig;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--fps" or arg == "--FPS") video_conf.FPS = std::stoi(argv[++i]);
        if (arg == "--exposure-rgb") video_conf.exposure_rgb = std::stoi(argv[++i]);
        if (arg == "--exposure-800") video_conf.exposure_800 = std::stoi(argv[++i]);
        if (arg == "--exposure-975") video_conf.exposure_975 = std::stoi(argv[++i]);
        if (arg == "--output_dir") video_conf.output_dir = string(argv[++i]);
        else if (arg == "--output-fsi") video_conf.output_fsi = true;
        else if (arg == "--output-rgb") video_conf.output_rgb = true;
        else if (arg == "--output-800") video_conf.output_800 = true;
        else if (arg == "--output-975") video_conf.output_975 = true;
        else if (arg == "--output-svo") video_conf.output_svo = true;
        else if (arg == "--view") video_conf.view = true;
    }

    std::cout << "FPS: " << FPS << std::endl;
    if (view)
        std::cout << "view mode: on" << std::endl;
    else {
        std::cout << "view mode: off" << std::endl;
        std::cout << "output-fsi: " << std::boolalpha << output_fsi << std::endl;
        std::cout << "output-rgb: " << std::boolalpha << output_rgb << std::endl;
        std::cout << "output-800: " << std::boolalpha << output_800 << std::endl;
        std::cout << "output-975: " << std::boolalpha << output_975 << std::endl;
        std::cout << "output-svo: " << std::boolalpha << output_svo << std::endl;
    }

    return video_conf;
}

bool setup_JAI(StreamInfo *MyStreamInfos[3], PvDevice *&lDevice, BufferList lBufferLists[3], VideoConfig video_conf) {
    PvString lConnectionID;
    bool test_streaming = true;

    if (SelectDeviceLocally(&lConnectionID)) {
        lDevice = ConnectToDevice(lConnectionID);
        if (lDevice != NULL) {
            PvGenParameterArray *lDeviceParams = lDevice->GetParameters();
            PvStream *lStreams[3] = {NULL, NULL, NULL};

            lDevice->GetParameters()->SetEnumValue("SourceSelector", "Source0");
            lDevice->GetParameters()->SetFloatValue("ExposureAutoControlMax", video_conf.exposure_rgb);
            lDevice->GetParameters()->SetEnumValue("SourceSelector", "Source1");
            lDevice->GetParameters()->SetFloatValue("ExposureAutoControlMax", video_conf.exposure_800);
            lDevice->GetParameters()->SetEnumValue("SourceSelector", "Source2");
            lDevice->GetParameters()->SetFloatValue("ExposureAutoControlMax", video_conf.exposure_975);
            lDevice->GetParameters()->SetEnumValue("SourceSelector", "Source0");
            lDevice->GetParameters()->SetFloatValue("AcquisitionFrameRate", video_conf.FPS);
            lDevice->GetParameters()->GetIntegerValue("Width", video_conf.width);
            lDevice->GetParameters()->GetIntegerValue("Height", video_conf.height);

            for (int i = 0; i < 3; i++) {
                MyStreamInfos[i] = new StreamInfo;
                MergeFramesEvent[i] = PTHREAD_COND_INITIALIZER;
                pthread_cond_init(&MergeFramesEvent[i], NULL);

                lStreams[i] = OpenStream(lConnectionID);
                if (lStreams[i] != NULL) {
                    ConfigureStream(lDevice, lStreams[i], i);
                    CreateStreamBuffers(lDevice, lStreams[i], &lBufferLists[i]);
                    MyStreamInfos[i]->aStream = lStreams[i];
                    MyStreamInfos[i]->StreamIndex = i;
                } else
                    test_streaming = false;
            }

            lDevice->StreamEnable();
            return test_streaming;
        }
    }

    lDevice->StreamEnable();
    return false;
}

int MP4CreateFirstTime(VideoConfig *&video_conf, VideoWriter *&mp4_FSI, VideoWriter *&mp4_RGB, VideoWriter *&mp4_800,
                       VideoWriter *&mp4_975){
    bool is_exist = false;
    int i = 0;
    char stat_FSI[100], stat_RGB[100], stat_800[100], stat_975[100], stat_SVO[100];
    char f_fsi[100], f_rgb[100], f_800[100], f_975[100];
    string width_s, height_s, FPS_s;

    if (not(video_conf->output_fsi or video_conf->output_rgb or video_conf->output_800 or video_conf->output_975)) {
        video_conf - 1;
        return;
    }

    width_s = to_string(video_conf->width);
    height_s = to_string(video_conf->height);
    FPS_s = to_string(video_conf->FPS);

    do {
        sprintf(stat_FSI, (video_conf->output_dir + string("/Result_FSI_%d.mkv")).c_str(), ++i);
        sprintf(stat_RGB, (video_conf->output_dir + string("/Result_RGB_%d.mkv")).c_str(), i);
        sprintf(stat_800, (video_conf->output_dir + string("/Result_800_%d.mkv")).c_str(), i);
        sprintf(stat_975, (video_conf->output_dir + string("/Result_975_%d.mkv")).c_str(), i);
        sprintf(stat_SVO, (video_conf->output_dir + string("/ZED_%d.mkv")).c_str(), ++i);
        is_exist = exists(stat_FSI) || exists(stat_RGB) ||
                exists(stat_800) || exists(stat_975) || exists(stat_SVO);
    } while (is_exist);

    string gst_3c = string("appsrc ! video/x-raw, format=BGR, width=(int)") + width_s +
            string(", height=(int)") + height_s + string(", framerate=(fraction)") + FPS_s +
            string("/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv !") +
            string("nvv4l2h265enc bitrate=15000000 ! h265parse ! matroskamux ! filesink location=");

    string gst_1c = string("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=");
    string gs_fsi = gst_3c + f_fsi, gs_rgb = gst_3c + f_rgb;
    string gs_800 = gst_1c + f_800, gs_975 = gst_1c + f_975;

    if (output_fsi) {
        sprintf(f_fsi, (string("\"") + video_conf->output_dir + string("/Result_FSI_%d.mkv\"")).c_str(), i);
        mp4_FSI->open(gs_fsi, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height));
    }
    if (output_rgb) {
        sprintf(f_rgb, (string("\"") + video_conf->output_dir + string("/Result_RGB_%d.mkv\"")).c_str(), i);
        mp4_BGR->open(gs_rgb, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height));
    }
    if (output_800) {
        sprintf(f_800, (string("\"") + video_conf->output_dir + string("/Result_800_%d.mkv\"")).c_str(), i);
        mp4_800->open(gs_800, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height), false);
    }
    if (output_975) {
        sprintf(f_975, (string("\"") + video_conf->output_dir + string("/Result_975_%d.mkv\"")).c_str(), i);
        mp4_975->open(gs_975, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height), false);
    }

    video_conf->file_index = -1;
}

bool exists(char path[100]){
    struct stat buffer;
    return stat(path, &buffer) == 0;
}

void setup_ZED(Camera *&zed, VideoConfig *&video_conf) {
    char zed_filename[100];
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = video_conf->FPS; // Set fps

    ERROR_CODE err = zed->open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        exit(-1);
    }

    // Enable video recording
    sprintf(zed_filename, (video_conf->output_dir + string("/ZED_%d.svo")).c_str(), video_conf->file_index);

    RecordingParameters params(zed_filename, SVO_COMPRESSION_MODE::H265);
    if (output_svo) {
        err = zed->enableRecording(params);
        if (err != ERROR_CODE::SUCCESS) {
            std::cout << toString(err) << std::endl;
            exit(-1);
        }
    }
}

void ZedThread(Camera *&zed) {
    // Grab ZED data and write to SVO file

    for (int i = 1; !_abort; i++) {
        ERROR_CODE err = zed->grab();
//        if (err != ERROR_CODE::SUCCESS)
//            outfile << "ZED FRAME DROP - FRAME NO. " << i << endl;
    }

    if (output_svo)
        zed->disableRecording();
    zed->close();
}

void GrabThread(void *_StreamInfo, pthread_cond_t *&GrabEvent, pthread_cond_t *MergeFramesEvent[3],
                pthread_mutex_t &tmp_mtx) {
    uint64_t PrevBlockID = 0;
    StreamInfo *MyStreamInfo = (StreamInfo *) _StreamInfo;
    PvStream *lStream = (PvStream *) (MyStreamInfo->aStream);
    int StreamIndex = (MyStreamInfo->StreamIndex), height, width;
    int stream_fail_count = 0;
    PvBuffer *lBuffer = NULL;


    PvResult lResult, lOperationResult;
    uint64_t CurrentBlockID = 0;

    pthread_mutex_lock(&tmp_mtx);
    pthread_cond_wait(&GrabEvent, &tmp_mtx);
    pthread_mutex_unlock(&tmp_mtx);
    cout << "JAI STREAM " << StreamIndex << " STARTED" << endl;
    while (!_abort) {
        *lBuffer = NULL;
        PvResult lOperationResult;

        lResult = lStream->RetrieveBuffer(&lBuffer, &lOperationResult, 1000);

        if (lResult.IsOK()) {
            if (lOperationResult.IsOK()) {
                // We now have a valid buffer. This is where you would typically process the buffer.

                CurrentBlockID = lBuffer->GetBlockID();
                if (CurrentBlockID != PrevBlockID + 1 and PrevBlockID != 0) {
                    outfile << "JAI STREAM " << StreamIndex << " - FRAME DROP - FRAME No. " << PrevBlockID << endl;
                }
                PrevBlockID = CurrentBlockID;
                height = lBuffer->GetImage()->GetHeight(), width = lBuffer->GetImage()->GetWidth();
                cv::Mat frame(height, width, CV_8U, lBuffer->GetImage()->GetDataPointer());
                lStream->QueueBuffer(lBuffer);
                EnumeratedFrame *curr_frame = new EnumeratedFrame;
                curr_frame->frame = frame;
                curr_frame->BlockID = CurrentBlockID + stream_fail_count;
                pthread_mutex_lock(&mtx);
                MyStreamInfo->Frames.push(curr_frame);
                pthread_cond_signal(&MergeFramesEvent[StreamIndex]);
                pthread_mutex_unlock(&mtx);
            } else {
                stream_fail_count++;
                lStream->QueueBuffer(lBuffer);
                cout << StreamIndex << ": OPR - FAILURE" << endl;
            }
            // Re-queue the buffer in the stream object
        } else {
            stream_fail_count++;
            cout << StreamIndex << ": BAD RESULT!" << endl;
            // Retrieve buffer failure
            cout << lResult.GetCodeString().GetAscii() << "\n";
        }
//        pthread_cond_signal(&MergeFramesEvent[StreamIndex]);
        // SetEvent(MergeFramesEvent[StreamIndex]); // signals the MergeThread that the output from current stream is ready
    }
    cout << StreamIndex << ": Acquisition end with " << CurrentBlockID << endl;
}

void MergeThread(void *_Frames, pthread_mutex_t &mtx) {
    queue<EnumeratedFrame *> **FramesQueue = (queue<EnumeratedFrame *> **) _Frames;
    cv::Mat Frames[3], res;
    cuda::GpuMat cudaFSI;
    std::vector<cuda::GpuMat> cudaBGR(3), cudaFrames(3), cudaFrames_equalized(3), cudaFrames_normalized(3);
    std::vector<cv::Mat> images(3);
    uint64_t frame_no;
    double elapsed = 0;
    char filename[100];
    struct timespec max_wait = {0, 0};
    EnumeratedFrame *e_frames[3] = {new EnumeratedFrame, new EnumeratedFrame, new EnumeratedFrame};
    bool grabbed[3] = { false, false, false };

    cout << "MERGE THREAD START" << endl;
    sleep(1);
    pthread_cond_broadcast(&GrabEvent);
    for (frame_no = 1; !_abort; frame_no++) {
        for (int i = 0; i < 3; i++) {
            pthread_mutex_lock(&mtx);
            while ((*(FramesQueue[i])).empty() and !grabbed[i] and !_abort) {
                clock_gettime(CLOCK_REALTIME, &max_wait);
                max_wait.tv_sec += 1;
                const int timed_wait_rv = pthread_cond_timedwait(&MergeFramesEvent[i], &mtx, &max_wait);
            }
            if (_abort) {
                pthread_mutex_unlock(&mtx);
                break;
            }
            if (!grabbed[i]) {
                e_frames[i] = (*(FramesQueue[i])).front();
                Frames[i] = e_frames[i]->frame;
                (*(FramesQueue[i])).pop();
            }
            grabbed[i] = false;
            pthread_mutex_unlock(&mtx);
        }
        if (_abort)
            break;
        if (e_frames[0]->BlockID != e_frames[1]->BlockID or e_frames[0]->BlockID != e_frames[2]->BlockID) {
            int max_id = std::max({e_frames[0]->BlockID, e_frames[1]->BlockID, e_frames[2]->BlockID});
            // outfile << "MERGE DROP - AFTER FRAME NO. " << --frame_no << endl;
            for (int i = 0; i < 3; i++) {
                if (e_frames[i]->BlockID == max_id)
                    grabbed[i] = true;
            }
            continue;
        }
        cv::Mat res_fsi, res_bgr, res_800, res_975;
        // the actual bayer format we use is RGGB (or - BayerRG) but OpenCV reffers to it as BayerBG - https://github.com/opencv/opencv/issues/19629

        cudaFrames[0].upload(Frames[0]); // channel 0 = BayerBG8
        cudaFrames[2].upload(Frames[1]); // channel 1 = 800nm -> Red
        cudaFrames[1].upload(Frames[2]); // channel 2 = 975nm -> Green

        cv::cuda::demosaicing(cudaFrames[0], cudaFrames[0], cv::COLOR_BayerBG2BGR);
        cv::cuda::split(cudaFrames[0], cudaBGR);
        cudaFrames[0].download(res_bgr);
        cudaFrames[2].download(res_800);
        cudaFrames[1].download(res_975);

        cudaFrames[0] = cudaBGR[0];

        for (int i = 0; i < 3; i++) {
            cv::cuda::equalizeHist(cudaFrames[i], cudaFrames_equalized[i]);
            cv::cuda::normalize(cudaFrames_equalized[i], cudaFrames_normalized[i], 0, 255, cv::NORM_MINMAX, CV_8U);
        }
        cv::cuda::merge(cudaFrames_normalized, cudaFSI);
        cudaFSI.download(res_fsi);

        if (output_fsi)
            mp4_FSI.write(res_fsi);
        if (output_rgb)
            mp4_BGR.write(res_bgr);
        if (output_800)
            mp4_800.write(res_800);
        if (output_975)
            mp4_975.write(res_975);
    }
}

StreamingParameters start_acquisition() {
    PvDevice *lDevice = NULL;
    PvStream *lStreams[3] = {NULL, NULL, NULL};
    BufferList lBufferLists[3];
    StreamInfo *MyStreamInfos[3];
    pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER, MergeFramesEvent[3];
    pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t tmp_mtx = PTHREAD_MUTEX_INITIALIZER;

    VideoConfig video_conf;
    thread zed_t, jai_t0, jai_t1, jai_t2, merge_t;
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStart;
    Camera zed;
    VideoWriter mp4_FSI, mp4_BGR, mp4_800, mp4_975;
    char log_filename[100];
    int file_index;
    queue < EnumeratedFrame * > *Frames[3] = {&(MyStreamInfos[0]->Frames), &(MyStreamInfos[1]->Frames),
                                              &(MyStreamInfos[2]->Frames)};
    StreamingParameters st_params = new StreamingParameters;

    st_params.MyStreamInfos = MyStreamInfos;
    st_params.lDevice = &lDevice;
    st_params.lBufferLists = lBufferLists;
    st_params.jai_t0 = &jai_t0;
    st_params.jai_t1 = &jai_t1;
    st_params.jai_t2 = &jai_t2;
    st_params.zed_t = &zed_t;
    st_params.merge_t = &merge_t;
    st_params.video_conf = &video_conf;
    st_params.GrabEvent = &GrabEvent;
    st_params.MergeFramesEvent = MergeFramesEvent;


    // check if this command is necessary
    PV_SAMPLE_INIT();

    pthread_cond_init(&GrabEvent, NULL)
    video_conf = parse_args(argc, argv);

    if (!setup_JAI(MyStreamInfos, lDevice, lBufferLists, video_conf))
        exit(-1);
    _abort = false;

    MP4CreateFirstTime(video_conf);

    sprintf(log_filename, (video_conf.output_dir + string("/frame_drop_%d.log")).c_str(), file_index);
    outfile.open(log_filename, std::ios_base::app); // append instead of overwrite

    // Enable streaming and send the AcquisitionStart command
    setup_ZED(zed, file_index);

    // Get device parameters need to control streaming - set acquisition start command
    lDeviceParams = lDevice->GetParameters();
    lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));

    zed_t = thread(ZedThread, zed);
    lStart->Execute();
    jai_t0 = thread(GrabThread, (void *) MyStreamInfos[0], &tmp_mtx);
    jai_t1 = thread(GrabThread, (void *) MyStreamInfos[1], &tmp_mtx);
    jai_t2 = thread(GrabThread, (void *) MyStreamInfos[2], &tmp_mtx);
    merge_t = thread(MergeThread, (void *) Frames, &mtx);

    return st_params;
}

void stop_acquisition(StreamingParameters st_params) {
    cout << "stopping acquisition" << endl;

    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStop;

    StreamInfo *MyStreamInfos[3];
    PvDevice *lDevice = NULL;
    BufferList lBufferLists[3];
    thread *zed_t, *jai_t0, *jai_t1, *jai_t2, *merge_t;
    VideoConfig *video_conf;
    pthread_cond_t *GrabEvent = PTHREAD_COND_INITIALIZER, MergeFramesEvent[3];

    MyStreamInfos = st_params.MyStreamInfos;
    lDevice = st_params.lDevice;
    lBufferLists = st_params.lBufferLists;
    jai_t0 = st_params.jai_t0;
    jai_t1 = st_params.jai_t1;
    jai_t2 = st_params.jai_t2;
    zed_t = st_params.zed_t;
    merge_t = st_params.merge_t;
    video_conf = st_params.video_conf;
    GrabEvent = st_params.GrabEvent;
    st_params.MergeFramesEvent = MergeFramesEvent;


    // Get device parameters need to control streaming - set acquisition stop command
    lDeviceParams = st_params.lDevice->GetParameters();
    lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));

    _abort = true;

    jai_t0->join();
    jai_t1->join();
    jai_t2->join();
    zed_t->join();
    merge_t->join();

    if (video_conf->output_fsi)
        mp4_FSI.release();
    if (video_conf->output_rgb)
        mp4_BGR.release();
    if (video_conf->output_800)
        mp4_800.release();
    if (video_conf->output_975)
        mp4_975.release();


    // Tell the device to stop sending images + disable streaming
    lStop->Execute();
    lDevice->StreamDisable();

    // Abort all buffers from the streams and dequeue
    cout << "Aborting buffers still in streams" << endl << "closing streams" << endl;
    for (int i = 0; i < 3; i++) {
        MyStreamInfos[i]->aStream->AbortQueuedBuffers();

        // Empty the remaining frames that from the buffers
        while (MyStreamInfos[i]->aStream->GetQueuedBufferCount() > 0) {
            PvBuffer *lBuffer = NULL;
            PvResult lOperationResult;
            MyStreamInfos[i]->aStream->RetrieveBuffer(&lBuffer, &lOperationResult);
        }
        FreeStreamBuffers(&lBufferLists[i]);
        MyStreamInfos[i]->aStream->Close();
        PvStream::Free(MyStreamInfos[i]->aStream);
    }

    // Disconnect the device
    lDevice->Disconnect();

    PvDevice::Free(lDevice);
    pthread_cond_destroy(GrabEvent);
    for (int i = 0; i < 3; i++) pthread_cond_destroy(&MergeFramesEvent[i]);
    PV_SAMPLE_TERMINATE();
}

int main(int argc, char* argv[]) {
    cout << "TRYING TO START ACQUISITION" << endl;
    StreamingParameters st_params = start_acquisition();
    cout << "ACQUISITION STARTED" << endl;
    cin.get();
    cout << "TRYING TO STOP ACQUISITION" << endl;
    stop_acquisition(st_params);
    cout << "ACQUISITION STOPPED" << endl;
}