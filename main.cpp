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

typedef list<PvBuffer *> BufferList;

typedef struct {
    cv::Mat frame;
    int BlockID;
} EnumeratedFrame;

typedef struct {
    PvStream *aStream;
    int stream_index;
    queue<EnumeratedFrame *> Frames;
} StreamInfo;

typedef struct {
    int FPS = 15, exposure_rgb = 500, exposure_800 = 1000, exposure_975 = 3000;
    int64_t width = 1536, height = 2048;
    int file_index = -1;
    bool output_fsi = false, output_rgb = false, output_800 = false, output_975 = false, output_svo = false;
    bool view = false;
    string output_dir = string("/home/mic-730ai/Desktop/JAI_Results");
} VideoConfig;

typedef struct {
    StreamInfo *MyStreamInfos[3];
    PvDevice *lDevice = NULL;
    BufferList lBufferLists[3];
    thread jai_t0, jai_t1, jai_t2, zed_t, merge_t;
    Camera zed;
    VideoConfig *video_conf;
    pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER;
    pthread_cond_t MergeFramesEvent[3];
    pthread_mutex_t grab_mtx = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t acq_start_mtx = PTHREAD_MUTEX_INITIALIZER;
    VideoWriter mp4_FSI, mp4_BGR, mp4_800, mp4_975;
    bool is_connected, is_running;
} AcquisitionParameters;

PV_INIT_SIGNAL_HANDLER();

bool SelectDeviceLocally(PvString *aConnectionID);

PvDevice *ConnectToDevice(const PvString &aConnectionID);

PvStream *OpenStream(const PvString &aConnectionID);

void ConfigureStream(PvDevice *&aDevice, PvStream *aStream, int channel);

void CreateStreamBuffers(PvDevice *&aDevice, PvStream *aStream, BufferList *aBufferList);

void FreeStreamBuffers(BufferList *aBufferList);

VideoConfig * parse_args(int fps, int exposure_rgb, int exposure_800, int exposure_975, string output_dir,
                         bool output_fsi, bool output_rgb, bool output_800, bool output_975, bool output_svo,
                         bool view);

bool setup_JAI(AcquisitionParameters &acq);

void MP4CreateFirstTime(AcquisitionParameters &acq);

string gs_sink_builder(int file_index, string output_type_name, string output_dir);

bool exists(char path[100]);

bool connect_ZED(AcquisitionParameters &acq);

void GrabThread(int stream_index, AcquisitionParameters &acq);

void ZedThread(AcquisitionParameters &acq);

void MergeThread(AcquisitionParameters &acq);

bool connect_cameras(AcquisitionParameters &acq, int fps, int exposure_rgb, int exposure_800, int exposure_975,
                     string output_dir, bool output_fsi, bool output_rgb, bool output_800, bool output_975,
                     bool output_svo, bool view);

bool start_acquisition(AcquisitionParameters &acq);

void stop_acquisition(AcquisitionParameters &acq);

void disconnect_cameras(AcquisitionParameters &acq);

ofstream frame_drop_log_file;

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
    else
        cout << "STREAM CREATED" << endl;

    return lStream;
}

void ConfigureStream(PvDevice *&aDevice, PvStream *aStream, int channel) {
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

void CreateStreamBuffers(PvDevice *&aDevice, PvStream *aStream, BufferList *aBufferList) {
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

VideoConfig * parse_args(int fps, int exposure_rgb, int exposure_800, int exposure_975, string output_dir,
                         bool output_fsi, bool output_rgb, bool output_800, bool output_975, bool output_svo,
                         bool view){
    VideoConfig *video_conf = new VideoConfig;

    video_conf->FPS = fps;
    video_conf->exposure_rgb = exposure_rgb;
    video_conf->exposure_800 = exposure_800;
    video_conf->exposure_975 = exposure_975;
    video_conf->output_dir = output_dir;
    video_conf->output_fsi = output_fsi;
    video_conf->output_rgb = output_rgb;
    video_conf->output_800 = output_800;
    video_conf->output_975 = output_975;
    video_conf->output_svo = output_svo;
    video_conf->view = view;

    std::cout << "FPS: " << video_conf->FPS << std::endl;
    if (video_conf->view)
        std::cout << "view mode: on" << std::endl;
    else {
        std::cout << "view mode: off" << std::endl;
        std::cout << "output-fsi: " << std::boolalpha << video_conf->output_fsi << std::endl;
        std::cout << "output-rgb: " << std::boolalpha << video_conf->output_rgb << std::endl;
        std::cout << "output-800: " << std::boolalpha << video_conf->output_800 << std::endl;
        std::cout << "output-975: " << std::boolalpha << video_conf->output_975 << std::endl;
        std::cout << "output-svo: " << std::boolalpha << video_conf->output_svo << std::endl;
        std::cout << "output-dir: " << std::boolalpha << video_conf->output_dir << std::endl;
    }

    return video_conf;
}


bool setup_JAI(AcquisitionParameters &acq) {
    PvString lConnectionID;
    bool test_streaming = true;

    if (acq.video_conf->output_fsi or acq.video_conf->output_rgb or acq.video_conf->output_800 or acq.video_conf->output_975) {
        if (SelectDeviceLocally(&lConnectionID)) {
            acq.lDevice = ConnectToDevice(lConnectionID);
            if (acq.lDevice != NULL) {
                PvGenParameterArray *lDeviceParams = acq.lDevice->GetParameters();
                PvStream *lStreams[3] = {NULL, NULL, NULL};

                lDeviceParams->SetEnumValue("SourceSelector", "Source0");
                lDeviceParams->SetFloatValue("ExposureAutoControlMax", acq.video_conf->exposure_rgb);
                lDeviceParams->SetEnumValue("SourceSelector", "Source1");
                lDeviceParams->SetFloatValue("ExposureAutoControlMax", acq.video_conf->exposure_800);
                lDeviceParams->SetEnumValue("SourceSelector", "Source2");
                lDeviceParams->SetFloatValue("ExposureAutoControlMax", acq.video_conf->exposure_975);
                lDeviceParams->SetEnumValue("SourceSelector", "Source0");
                lDeviceParams->SetFloatValue("AcquisitionFrameRate", acq.video_conf->FPS);
                lDeviceParams->GetIntegerValue("Width", acq.video_conf->width);
                lDeviceParams->GetIntegerValue("Height", acq.video_conf->height);

                for (int i = 0; i < 3; i++) {
                    acq.MyStreamInfos[i] = new StreamInfo;
                    acq.MergeFramesEvent[i] = PTHREAD_COND_INITIALIZER;
                    pthread_cond_init(&acq.MergeFramesEvent[i], NULL);

                    lStreams[i] = OpenStream(lConnectionID);
                    if (lStreams[i] != NULL) {
                        ConfigureStream(acq.lDevice, lStreams[i], i);
                        CreateStreamBuffers(acq.lDevice, lStreams[i], &acq.lBufferLists[i]);
                        acq.MyStreamInfos[i]->aStream = lStreams[i];
                        acq.MyStreamInfos[i]->stream_index = i;
                    } else
                        test_streaming = false;
                }

                acq.lDevice->StreamEnable();
                return test_streaming;
            }
            else
                return false;
        }
        else
            return false;
    }
    else
        return true;
}

void MP4CreateFirstTime(AcquisitionParameters &acq){
    bool is_exist = false;
    int i = 0;
    char stat_FSI[100], stat_RGB[100], stat_800[100], stat_975[100], stat_SVO[100];
    string f_fsi, f_rgb, f_800, f_975;
    char zed_filename[100];
    string width_s, height_s, FPS_s;

    if (not(acq.video_conf->output_fsi or acq.video_conf->output_rgb or acq.video_conf->output_800 or acq.video_conf->output_975)) {
        acq.video_conf->file_index = - 1;
        return;
    }

    width_s = to_string(acq.video_conf->width);
    height_s = to_string(acq.video_conf->height);
    FPS_s = to_string(acq.video_conf->FPS);

    do {
        sprintf(stat_FSI, (acq.video_conf->output_dir + string("/Result_FSI_%d.mkv")).c_str(), ++i);
        sprintf(stat_RGB, (acq.video_conf->output_dir + string("/Result_RGB_%d.mkv")).c_str(), i);
        sprintf(stat_800, (acq.video_conf->output_dir + string("/Result_800_%d.mkv")).c_str(), i);
        sprintf(stat_975, (acq.video_conf->output_dir + string("/Result_975_%d.mkv")).c_str(), i);
        sprintf(stat_SVO, (acq.video_conf->output_dir + string("/ZED_%d.mkv")).c_str(), i);
        is_exist = exists(stat_FSI) || exists(stat_RGB) ||
                exists(stat_800) || exists(stat_975) || exists(stat_SVO);
    } while (is_exist);

    acq.video_conf->file_index = i;

    string gst_3c = string("appsrc ! video/x-raw, format=BGR, width=(int)") + width_s +
            string(", height=(int)") + height_s + string(", framerate=(fraction)") + FPS_s +
            string("/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv !") +
            string("nvv4l2h265enc bitrate=15000000 ! h265parse ! matroskamux ! filesink location=");

    string gst_1c = string("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=");

    cv::Size frame_size(acq.video_conf->width, acq.video_conf->height);
    int four_c = VideoWriter::fourcc('H', '2', '6', '5');
    if (acq.video_conf->output_fsi) {
        f_fsi = gs_sink_builder(acq.video_conf->file_index, "FSI", acq.video_conf->output_dir);
        string gs_fsi = gst_3c + f_fsi;
        acq.mp4_FSI.open(gs_fsi, four_c, acq.video_conf->FPS, frame_size);
    }
    if (acq.video_conf->output_rgb) {
        f_rgb = gs_sink_builder(acq.video_conf->file_index, "RGB", acq.video_conf->output_dir);
        string gs_rgb = gst_3c + f_rgb;
        acq.mp4_BGR.open(gs_rgb, four_c, acq.video_conf->FPS, frame_size);
    }
    if (acq.video_conf->output_800) {;
        f_800 = gs_sink_builder(acq.video_conf->file_index, "800", acq.video_conf->output_dir);
        string gs_800 = gst_1c + f_800;
        acq.mp4_800.open(gs_800, four_c, acq.video_conf->FPS, frame_size, false);
    }
    if (acq.video_conf->output_975) {
        f_975 = gs_sink_builder(acq.video_conf->file_index, "975", acq.video_conf->output_dir);
        string gs_975 = gst_1c + f_975;
        acq.mp4_975.open(gs_975, four_c, acq.video_conf->FPS, frame_size, false);
    }
    if (acq.video_conf->output_svo) {
        sprintf(zed_filename, (acq.video_conf->output_dir + string("/ZED_%d.svo")).c_str(), acq.video_conf->file_index);

        RecordingParameters params(zed_filename, SVO_COMPRESSION_MODE::H265);
        acq.zed.enableRecording(params);
    }
}

string gs_sink_builder(int file_index, string output_type_name, string output_dir){
    string gs_loc = string("\"") + output_dir;
    gs_loc += string("/Result_") + output_type_name + "_" + to_string(file_index) + string(".mkv\"");
    return gs_loc;
}


bool exists(char path[100]){
    struct stat buffer;
    return stat(path, &buffer) == 0;
}

bool connect_ZED(AcquisitionParameters &acq){
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = acq.video_conf->FPS; // Set fps

    if (acq.video_conf->output_svo) {
        ERROR_CODE err = acq.zed.open(init_params);
        return err == ERROR_CODE::SUCCESS;
    }
    else
        return true;
}


void ZedThread(AcquisitionParameters &acq) {
    // Grab ZED data and write to SVO file

    for (int i = 1; acq.is_running; i++) {
        ERROR_CODE err = acq.zed.grab();
        if (err != ERROR_CODE::SUCCESS)
            frame_drop_log_file << "ZED FRAME DROP - FRAME NO. " << i << endl;
    }

    if (acq.video_conf->output_svo)
        acq.zed.disableRecording();
    acq.zed.close();
}

void GrabThread(int stream_index, AcquisitionParameters &acq) {
    StreamInfo *MyStreamInfo = acq.MyStreamInfos[stream_index];
    PvStream *lStream = (PvStream *) (MyStreamInfo->aStream);
    PvResult lResult, lOperationResult;
    uint64_t CurrentBlockID = 0, PrevBlockID = 0;
    int height, width;

    pthread_mutex_lock(&acq.acq_start_mtx);
    pthread_cond_wait(&acq.GrabEvent, &acq.acq_start_mtx);
    pthread_mutex_unlock(&(acq.acq_start_mtx));
    cout << "JAI STREAM " << stream_index << " STARTED" << endl;
    while (acq.is_running) {
        PvBuffer *lBuffer = NULL;
        lResult = lStream->RetrieveBuffer(&lBuffer, &lOperationResult, 1000);

        if (lResult.IsOK()) {
            if (lOperationResult.IsOK()) {
                // We now have a valid buffer. This is where you would typically process the buffer.

                CurrentBlockID = lBuffer->GetBlockID();
                if (CurrentBlockID != PrevBlockID + 1 and PrevBlockID != 0) {
                    frame_drop_log_file << "JAI STREAM " << stream_index << " - FRAME DROP - FRAME No. " << PrevBlockID << endl;
                }
                PrevBlockID = CurrentBlockID;
                height = lBuffer->GetImage()->GetHeight(), width = lBuffer->GetImage()->GetWidth();
                cv::Mat frame(height, width, CV_8U, lBuffer->GetImage()->GetDataPointer());
                lStream->QueueBuffer(lBuffer);
                EnumeratedFrame *curr_frame = new EnumeratedFrame;
                curr_frame->frame = frame;
                curr_frame->BlockID = CurrentBlockID;
                pthread_mutex_lock(&acq.grab_mtx);
                MyStreamInfo->Frames.push(curr_frame);
                pthread_cond_signal(&acq.MergeFramesEvent[stream_index]);
                pthread_mutex_unlock(&acq.grab_mtx);
            } else {
                lStream->QueueBuffer(lBuffer);
//                cout << stream_index << ": OPR - FAILURE" << endl;
            }
            // Re-queue the buffer in the stream object
        } else {
            cout << stream_index << ": BAD RESULT!" << endl;
            // Retrieve buffer failure
            cout << lResult.GetCodeString().GetAscii() << "\n";
        }
    }
    cout << stream_index << ": Acquisition end with " << CurrentBlockID << endl;
}

void MergeThread(AcquisitionParameters &acq) {

    cv::Mat Frames[3], res;
    cuda::GpuMat cudaFSI;
    std::vector<cuda::GpuMat> cudaBGR(3), cudaFrames(3), cudaFrames_equalized(3), cudaFrames_normalized(3);
    std::vector<cv::Mat> images(3);
    int frame_count = 0;
    struct timespec max_wait = {0, 0};
    EnumeratedFrame *e_frames[3] = {new EnumeratedFrame, new EnumeratedFrame, new EnumeratedFrame};
    bool grabbed[3] = { false, false, false };

    cout << "MERGE THREAD START" << endl;
    sleep(1);
    pthread_cond_broadcast(&acq.GrabEvent);
    for (int frame_no = 1; acq.is_running; frame_no++) {
        for (int i = 0; i < 3; i++) {
            pthread_mutex_lock(&acq.grab_mtx);
            while (acq.MyStreamInfos[i]->Frames.empty() and !grabbed[i] and acq.is_running) {
                clock_gettime(CLOCK_REALTIME, &max_wait);
                max_wait.tv_sec += 1;
                const int timed_wait_rv = pthread_cond_timedwait(&acq.MergeFramesEvent[i], &acq.grab_mtx, &max_wait);
            }
            if (!acq.is_running) {
                pthread_mutex_unlock(&acq.grab_mtx);
                break;
            }
            if (!grabbed[i]) {
                e_frames[i] = acq.MyStreamInfos[i]->Frames.front();
                Frames[i] = e_frames[i]->frame;
                acq.MyStreamInfos[i]->Frames.pop();
            }
            grabbed[i] = false;
            pthread_mutex_unlock(&acq.grab_mtx);
        }
        if (!acq.is_running)
            break;
        if (e_frames[0]->BlockID != e_frames[1]->BlockID or e_frames[0]->BlockID != e_frames[2]->BlockID) {
            int max_id = std::max({e_frames[0]->BlockID, e_frames[1]->BlockID, e_frames[2]->BlockID});
            frame_drop_log_file << "MERGE DROP - AFTER FRAME NO. " << --frame_no << endl;
            for (int i = 0; i < 3; i++) {
                if (e_frames[i]->BlockID == max_id)
                    grabbed[i] = true;
            }
            continue;
        }
        cv::Mat res_fsi, res_bgr;
        // the actual bayer format we use is RGGB (or - BayerRG) but OpenCV reffers to it as BayerBG - https://github.com/opencv/opencv/issues/19629

        cudaFrames[0].upload(Frames[0]); // channel 0 = BayerBG8
        cudaFrames[2].upload(Frames[1]); // channel 1 = 800nm -> Red
        cudaFrames[1].upload(Frames[2]); // channel 2 = 975nm -> Green

        cv::cuda::demosaicing(cudaFrames[0], cudaFrames[0], cv::COLOR_BayerBG2BGR);
        cv::cuda::split(cudaFrames[0], cudaBGR);
        if (acq.video_conf->output_rgb)
            cudaFrames[0].download(res_bgr);

        cudaFrames[0] = cudaBGR[2]; // just pick the blue from the bayer

        for (int i = 0; i < 3; i++) {
            cv::cuda::equalizeHist(cudaFrames[i], cudaFrames_equalized[i]);
            cv::cuda::normalize(cudaFrames_equalized[i], cudaFrames_normalized[i], 0, 255, cv::NORM_MINMAX, CV_8U);
        }
        cv::cuda::merge(cudaFrames_normalized, cudaFSI);
        cudaFSI.download(res_fsi);

        frame_count++;
        if (frame_count % (acq.video_conf->FPS * 30) == 0)
            cout << endl << frame_count / (acq.video_conf->FPS * 60.0) << " minutes of video written" << endl << endl;

        if (acq.video_conf->output_fsi)
            acq.mp4_FSI.write(res_fsi);
        if (acq.video_conf->output_rgb)
            acq.mp4_BGR.write(res_bgr);
        if (acq.video_conf->output_800)
            acq.mp4_800.write(Frames[1]);
        if (acq.video_conf->output_975)
            acq.mp4_975.write(Frames[2]);
    }
}

bool connect_cameras(AcquisitionParameters &acq, int fps, int exposure_rgb, int exposure_800, int exposure_975,
                     string output_dir, bool output_fsi, bool output_rgb, bool output_800, bool output_975,
                     bool output_svo, bool view){
    PvStream *lStreams[3] = {NULL, NULL, NULL};
    int file_index;

    // check if this command is necessary
    PV_SAMPLE_INIT();

    pthread_cond_init(&acq.GrabEvent, NULL);
    acq.video_conf = parse_args(fps, exposure_rgb, exposure_800, exposure_975, output_dir, output_fsi, output_rgb,
                                output_800, output_975, output_svo, view);

    bool jai_success, zed_success;

    jai_success = setup_JAI(acq);
    zed_success = connect_ZED(acq);

    acq.is_connected = jai_success and zed_success;
    return acq.is_connected;
}


bool start_acquisition(AcquisitionParameters &acq) {
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStart;

    if (not acq.is_connected) {
        cout << "NOT CONNECTED" << endl;
        return false;
    }

    acq.is_running = true;
    MP4CreateFirstTime(acq);

    // Get device parameters need to control streaming - set acquisition start command
    lDeviceParams = acq.lDevice->GetParameters();
    lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));

    acq.zed_t = thread(ZedThread, ref(acq));
    lStart->Execute();
    acq.jai_t0 = thread(GrabThread, 0, ref(acq));
    acq.jai_t1 = thread(GrabThread, 1, ref(acq));
    acq.jai_t2 = thread(GrabThread, 2, ref(acq));
    acq.merge_t = thread(MergeThread, ref(acq));
    return true;
}

void stop_acquisition(AcquisitionParameters &acq) {
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStop;

    // Get device parameters need to control streaming - set acquisition stop command
    lDeviceParams = acq.lDevice->GetParameters();
    lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));

    acq.is_running = false;

    acq.jai_t0.join();
    acq.jai_t1.join();
    acq.jai_t2.join();
    acq.zed_t.join();
    acq.merge_t.join();

    if (acq.video_conf->output_fsi)
        acq.mp4_FSI.release();
    if (acq.video_conf->output_rgb)
        acq.mp4_BGR.release();
    if (acq.video_conf->output_800)
        acq.mp4_800.release();
    if (acq.video_conf->output_975)
        acq.mp4_975.release();


    // Tell the device to stop sending images + disable streaming
    lStop->Execute();
    acq.lDevice->StreamDisable();
}

void disconnect_cameras(AcquisitionParameters &acq){
    // Abort all buffers from the streams and dequeue
    cout << "Aborting buffers still in streams" << endl << "closing streams" << endl;
    for (int i = 0; i < 3; i++) {
        acq.MyStreamInfos[i]->aStream->AbortQueuedBuffers();

        // Empty the remaining frames that from the buffers
        while (acq.MyStreamInfos[i]->aStream->GetQueuedBufferCount() > 0) {
            PvBuffer *lBuffer = NULL;
            PvResult lOperationResult;
            acq.MyStreamInfos[i]->aStream->RetrieveBuffer(&lBuffer, &lOperationResult);
        }
        FreeStreamBuffers(&acq.lBufferLists[i]);
        acq.MyStreamInfos[i]->aStream->Close();
        PvStream::Free(acq.MyStreamInfos[i]->aStream);
    }

    // Disconnect the device
    acq.lDevice->Disconnect();

    PvDevice::Free(acq.lDevice);
    pthread_cond_destroy(&acq.GrabEvent);
    for (int i = 0; i < 3; i++) pthread_cond_destroy(&acq.MergeFramesEvent[i]);

    // check if necessary
    PV_SAMPLE_TERMINATE();
}

int main(int argc, char* argv[]) {
    int fps = 15;
    int exposure_rgb = 500, exposure_800 = 1000, exposure_975 = 2000;
    bool output_fsi = true, output_rgb = true, output_svo = true;
    bool output_800 = false, output_975 = false, view = false;
    string output_dir = "/home/mic-730ai/Desktop/BRN00106/row_1";

    AcquisitionParameters acq;
    cout << "TRYING TO CONNECT CAMERAS ACQUISITION" << endl;
    connect_cameras(acq, fps, exposure_rgb, exposure_800, exposure_975, output_dir, output_fsi, output_rgb, output_800,
                    output_975, output_svo, view);
    cout << "TRYING TO START ACQUISITION" << endl;
    start_acquisition(acq);
    cout << "ACQUISITION STARTED" << endl;
    cin.get();
    cout << "TRYING TO STOP ACQUISITION" << endl;
    stop_acquisition(acq);
    cout << "ACQUISITION STOPPED" << endl;
    cout << "TRYING TO DISCONNECT CAMERAS" << endl;
    disconnect_cameras(acq);
    cout << "CAMERAS DISCONNECTED" << endl;
}