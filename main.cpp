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
#include <nlohmann/json.hpp>
#include <dirent.h>

using namespace cv;
using namespace cuda;
using namespace sl;
using namespace std;
using json = nlohmann::json;

typedef map<string, PvDisplayWnd *> PvDisplayWndMap;

typedef list<PvBuffer *> BufferList;

PV_INIT_SIGNAL_HANDLER();


PvDevice *ConnectToDevice(const PvString &aConnectionID);

PvStream *OpenStream(const PvString &aConnectionID);

void CreateStreamBuffers(PvDevice *aDevice, PvStream *aStream, BufferList *aBufferList);

void FreeStreamBuffers(BufferList *aBufferList);

void GrabThread(void *StreamInfo);

void ZedThread(int file_index);

void MergeThread(void *_Frames);

int MP4CreateFirstTime(int height, int width, string output_dir);

void ConfigureStream(PvDevice *aDevice, PvStream *aStream, int channel);

typedef struct {
    cv::Mat frame;
    int BlockID;
} EnumeratedFrame;

typedef struct {
    PvStream *aStream;
    int StreamIndex;
    queue<EnumeratedFrame *> Frames;
} StreamInfo;

void set_output_dir();

bool SelectDeviceLocally(PvString *aConnectionID);

bool setup_JAI(StreamInfo *MyStreamInfos[3], PvDevice *&lDevice, BufferList lBufferLists[3]);

void setup_ZED(int file_index);

void parse_args(int argc, char* argv[]);

void sigterm_handler(int signum);


pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER, MergeFramesEvent[3];
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t tmp_mtx = PTHREAD_MUTEX_INITIALIZER;
VideoWriter mp4_FSI, mp4_BGR, mp4_800, mp4_975;
ofstream outfile;
json config;
string output_dir;

bool _abort = false, mp4_init = false;
float avg_coloring = 0;
int frame_count = 0;
PvDisplayWndMap mDisplays;
PvString mSource;
int FPS = 15;
bool view = false;
bool output_fsi = false, output_rgb = false, output_800 = false, output_975 = false, output_svo = false;
int64_t width, height;
Camera zed;
TickMeter t0;

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

void set_output_dir() {
    DIR *dir;
    dirent *ent;
    struct stat validator;
    int i = 0, output_i;
    output_dir = string("/home/mic-730ai/Desktop/JAI_Results");
    string target_dir;
    cout << "choose output device" << endl;
//    if ((dir = opendir(output_dir.c_str())) != NULL)
//        while ((ent = readdir(dir)) != NULL)
//            if (strcmp(ent->d_name, ".") & strcmp(ent->d_name, "..") != 0)
//                cout << "\t" << ++i << " - " << ent->d_name << endl;
//    i = 0;
//    cin >> output_i;
//    cout << "Enter final directory name (default is Acquisition)" << endl;
//    cin >> target_dir;
//    if (target_dir.empty())
//        target_dir = string("Acquisition");
//    target_dir = string("/") + target_dir;
//    if ((dir = opendir(output_dir.c_str())) != NULL)
//        while ((ent = readdir(dir)) != NULL)
//            if ((strcmp(ent->d_name, ".") & strcmp(ent->d_name, "..") != 0) and ++i == output_i) {
//                output_dir += string(ent->d_name) + target_dir;
//                break;
//            }

    if (stat(output_dir.c_str(), &validator) != 0) {
        mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        cout << output_dir << " Created!" << endl;
    }
}

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


bool setup_JAI(StreamInfo *MyStreamInfos[3], PvDevice *&lDevice, BufferList lBufferLists[3]) {
    PvString lConnectionID;
    bool test_streaming = true;
    if (SelectDeviceLocally(&lConnectionID)) {
        lDevice = ConnectToDevice(lConnectionID);
        if (lDevice != NULL) {
            PvGenParameterArray *lDeviceParams = lDevice->GetParameters();
            PvStream *lStreams[3] = {NULL, NULL, NULL};

            lDevice->GetParameters()->SetFloatValue("AcquisitionFrameRate", FPS);
            lDevice->GetParameters()->GetIntegerValue("Width", width);
            lDevice->GetParameters()->GetIntegerValue("Height", height);

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

            return test_streaming;
        }
    }
    return false;
}

void setup_ZED(int file_index) {
    char zed_filename[100];
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = FPS; // Set fps

    ERROR_CODE err = zed.open(init_params);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        exit(-1);
    }

    // Enable video recording
    sprintf(zed_filename, (output_dir + string("/ZED_%d.svo")).c_str(), file_index);

    RecordingParameters params(zed_filename, SVO_COMPRESSION_MODE::H265);
    if (output_svo) {
        err = zed.enableRecording(params);
        if (err != ERROR_CODE::SUCCESS) {
            std::cout << toString(err) << std::endl;
            exit(-1);
        }
    }
}

void parse_args(int argc, char *argv[]){
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--fps" or arg == "--FPS") FPS = std::stoi(argv[++i]);
        else if (arg == "--output-fsi") output_fsi = true;
        else if (arg == "--output-rgb") output_rgb = true;
        else if (arg == "--output-800") output_800 = true;
        else if (arg == "--output-975") output_975 = true;
        else if (arg == "--output-svo") output_svo = true;
        else if (arg == "--view") view = true;
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
}

void sigterm_handler(int signum) {
    _abort = true;
    cout << "sigterm handled" << endl;
}

int main(int argc, char* argv[]) {
    cout << "**************** CODEC: nvv4l2h265enc ****************" << endl;
    cout << "**************** bitrate: 15M ****************" << endl << endl;

    parse_args(argc, argv);
    signal(SIGTERM, sigterm_handler);

    PvDevice *lDevice = NULL;
    PvStream *lStreams[3] = {NULL, NULL, NULL};
    BufferList lBufferLists[3];
    StreamInfo *MyStreamInfos[3];
    ifstream config_file("/home/mic-730ai/fruitspec/JAI_Acquisition/config.json", std::ifstream::binary);

    config = json::parse(config_file);

    pthread_cond_init(&GrabEvent, NULL)

    PV_SAMPLE_INIT();

    set_output_dir();
    cout << "saving files to " << output_dir << endl;

    PvString lConnectionID;

    if (!setup_JAI(MyStreamInfos, lDevice, lBufferLists))
        exit(-1);
    _abort = false;
    mp4_init = false;

    // Get device parameters need to control streaming
    PvGenParameterArray *lDeviceParams = lDevice->GetParameters();

    // Map the GenICam AcquisitionStart and AcquisitionStop commands
    PvGenCommand *lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));
    PvGenCommand *lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));

    queue<EnumeratedFrame *> *Frames[3] = {&(MyStreamInfos[0]->Frames), &(MyStreamInfos[1]->Frames),
                                           &(MyStreamInfos[2]->Frames)};
    cout << 3 << endl;
    lDevice->StreamEnable();
    int file_index = MP4CreateFirstTime(height, width, output_dir);


    char log_filename[100];
    sprintf(log_filename, (output_dir + string("/frame_drop_%d.log")).c_str(), file_index);
    outfile.open(log_filename, std::ios_base::app); // append instead of overwrite

    // Enable streaming and send the AcquisitionStart command

    setup_ZED(file_index);

    cout << "Enabling streaming and sending AcquisitionStart command." << endl;
    t0.start();
    thread zed_t(ZedThread, file_index);

    lStart->Execute();
    thread jai_t1(GrabThread, (void *) MyStreamInfos[0]);
    thread jai_t2(GrabThread, (void *) MyStreamInfos[1]);
    thread jai_t3(GrabThread, (void *) MyStreamInfos[2]);
    thread merge_t(MergeThread, (void *) Frames);

    // it won't pass the joins until _abort will be set to true and this could only happen if SIGTERM is received

    jai_t1.join();
    jai_t2.join();
    jai_t3.join();
    zed_t.join();
    merge_t.join();

    if (output_fsi)
        mp4_FSI.release();
    if (output_rgb)
        mp4_BGR.release();
    if (output_800)
        mp4_800.release();
    if (output_975)
        mp4_975.release();


    // Tell the device to stop sending images + disable streaming
    lStop->Execute();
    lDevice->StreamDisable();

    // close frame drop log file
    outfile.close();

    // Abort all buffers from the streams and dequeue
    cout << "Aborting buffers still in streams" << endl << "closing streams" << endl;
    for (int i = 0; i < 3; i++) {
        MyStreamInfos[i]->aStream->AbortQueuedBuffers();
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
    cout << "Disconnecting device" << endl;
    lDevice->Disconnect();

    PvDevice::Free(lDevice);
    cout << endl;
    //cout << "<press a key to exit>" << endl;
    //PvWaitForKeyPress();
    // CloseHandle(GrabEvent);
    pthread_cond_destroy(&GrabEvent);
    for (int i = 0; i < 3; i++) pthread_cond_destroy(&MergeFramesEvent[i]);
    PV_SAMPLE_TERMINATE();
    return 0;
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

void ZedThread(int file_index) {
    cout << "ZED THREAD STARTED" << endl;
    // Grab data and write to SVO file

    for (int i = 1; !_abort; i++) {
        ERROR_CODE err = zed.grab();
        if (err != ERROR_CODE::SUCCESS)
            outfile << "ZED FRAME DROP - FRAME NO. " << i << endl;
    }

    if (output_svo)
        zed.disableRecording();
    zed.close();
    cout << "ZED ACQUISITION END" << endl;
}

void GrabThread(void *_StreamInfo) {
    uint64_t PrevBlockID = 0;
    StreamInfo *MyStreamInfo = (StreamInfo *) _StreamInfo;
    PvStream *lStream = (PvStream *) (MyStreamInfo->aStream);
    int StreamIndex = (MyStreamInfo->StreamIndex), height, width;
    int stream_fail_count = 0;
    PvBuffer *lBuffer = NULL;
    char str[200];


    PvResult lResult, lOperationResult;
    uint64_t CurrentBlockID = 0;

    // WaitForSingleObject(GrabEvent, INFINITE);
    pthread_mutex_lock(&tmp_mtx);
    pthread_cond_wait(&GrabEvent, &tmp_mtx);
    pthread_mutex_unlock(&tmp_mtx);
    cout << "JAI STREAM " << StreamIndex << " STARTED" << endl;
    while (!_abort) {
        PvBuffer *lBuffer = NULL;
        PvResult lOperationResult;

        lResult = lStream->RetrieveBuffer(&lBuffer, &lOperationResult, 1000);

        if (lResult.IsOK()) {
            if (lOperationResult.IsOK()) {
                // We now have a valid buffer. This is where you would typically process the buffer.

                CurrentBlockID = lBuffer->GetBlockID();
                if (CurrentBlockID != PrevBlockID + 1 and PrevBlockID != 0) {
                    outfile << "JAI STREAM " << StreamIndex << " - FRAME DROP - FRAME No. " << PrevBlockID << endl;
                    stream_fail_count -= CurrentBlockID - (PrevBlockID + 1);
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

void MergeThread(void *_Frames) {
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
            }
            grabbed[i] = false;
            t0.stop();
//            cout << "STREAM " << i << " POP " << fnum <<" AFTER " << t0.getTimeMilli() << endl;
            t0.start();
            (*(FramesQueue[i])).pop();
            pthread_mutex_unlock(&mtx);
        }
        if (_abort)
            break;
        if (e_frames[0]->BlockID != e_frames[1]->BlockID or e_frames[0]->BlockID != e_frames[2]->BlockID) {
            int max_id = std::max({e_frames[0]->BlockID, e_frames[1]->BlockID, e_frames[2]->BlockID});
            outfile << "MERGE DROP - AFTER FRAME NO. " << --frame_no << endl;
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

        frame_count++;
        if (frame_count % (FPS * 30) == 0)
            cout << endl << frame_count / (FPS * 60.0) << " minutes of video written" << endl << endl;

        TickMeter t;
        t.start();

        if (output_fsi)
            mp4_FSI.write(res_fsi);
        if (output_rgb)
            mp4_BGR.write(res_bgr);
        if (output_800)
            mp4_800.write(res_800);
        if (output_975)
            mp4_975.write(res_975);

        t.stop();
        elapsed = t.getTimeMilli();
        avg_coloring += elapsed;
    }
    cout << "MERGE END" << endl;
}

bool exists(char path[100]){
    struct stat buffer;
    return stat(path, &buffer) == 0;
}

int MP4CreateFirstTime(int height, int width, string output_dir) {
    if (!mp4_init) {
        bool is_exist = false;
        int i = 0;
        char stat_FSI[100], stat_RGB[100], stat_800[100], stat_975[100], stat_SVO[100];
        char f_fsi[100], f_rgb[100], f_800[100], f_975[100];

        if (not(save_FSI or save_RGB or save_800 or save_975))
            return -1;
        do {
            sprintf(stat_FSI, (output_dir + string("/Result_FSI_%d.mkv")).c_str(), ++i);
            sprintf(stat_RGB, (output_dir + string("/Result_RGB_%d.mkv")).c_str(), i);
            sprintf(stat_800, (output_dir + string("/Result_800_%d.mkv")).c_str(), i);
            sprintf(stat_975, (output_dir + string("/Result_975_%d.mkv")).c_str(), i);
            sprintf(stat_SVO, (output_dir + string("/ZED_%d.mkv")).c_str(), ++i);
            is_exist = exists(stat_FSI) || exists(stat_RGB) || exists(stat_800) || exists(stat_975) || exists(stat_SVO));
        } while (is_exist);

        string gst_3c = string("appsrc ! video/x-raw, format=BGR, width=(int)") + to_string(width) + string(", height=(int)") +
                        to_string(height) + string(", framerate=(fraction)") + to_string(FPS) +
                        string("/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! "
                               "nvv4l2h265enc bitrate=15000000 ! h265parse ! matroskamux ! filesink location=");
        // x265 encoder
//        string gst_3c = string("appsrc ! videoconvert ! x265enc ! h265parse ! matroskamux ! filesink location=");
        string gst_1c = string("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=");
        string gs_fsi = gst_3c + f_fsi, gs_rgb = gst_3c + f_rgb;
        string gs_800 = gst_1c + f_800, gs_975 = gst_1c + f_975;

        if (output_fsi) {
            sprintf(f_fsi, (string("\"") + output_dir + string("/Result_FSI_%d.mkv\"")).c_str(), i);
            mp4_FSI.open(gs_fsi, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height));
        }
        if (output_rgb) {
            sprintf(f_rgb, (string("\"") + output_dir + string("/Result_RGB_%d.mkv\"")).c_str(), i);
            mp4_BGR.open(gs_rgb, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height));
        }
        if (output_800) {
            sprintf(f_800, (string("\"") + output_dir + string("/Result_800_%d.mkv\"")).c_str(), i);
            mp4_800.open(gs_800, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height), false);
        }
        if (output_975) {
            sprintf(f_975, (string("\"") + output_dir + string("/Result_975_%d.mkv\"")).c_str(), i);
            mp4_975.open(gs_975, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height), false);
        }

        mp4_init = true;
        cout << endl << endl << "---- Video capturing started ----" << endl;

        return i;
    }
    return -1;
}
