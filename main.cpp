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

void Display(PvBuffer *aBuffer);

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

bool setup_JAI(StreamInfo **&MyStreamInfos, PvDevice *&lDevice, BufferList *&lBufferLists);

void setup_ZED(int file_index);

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
int64_t width, height;
Camera zed;
TickMeter t0;


std::string getCurrentTimestamp() {
    using std::chrono::system_clock;
    auto currentTime = std::chrono::system_clock::now();
    char buffer[80];

    auto transformed = currentTime.time_since_epoch().count() / 1000000;

    auto millis = transformed % 1000;

    std::time_t tt;
    tt = system_clock::to_time_t(currentTime);
    auto timeinfo = localtime(&tt);
    strftime(buffer, 80, "%F %H:%M:%S", timeinfo);
    sprintf(buffer, "%s:%03d", buffer, (int) millis);

    return std::string(buffer);
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

void set_output_dir() {
    DIR *dir;
    dirent *ent;
    struct stat validator;
    int i = 0, output_i;
    output_dir = string("/media/mic-730ai/");
    string target_dir;
    cout << "choose output device" << endl;
    if ((dir = opendir(output_dir.c_str())) != NULL)
        while ((ent = readdir(dir)) != NULL)
            if (strcmp(ent->d_name, ".") & strcmp(ent->d_name, "..") != 0)
                cout << "\t" << ++i << " - " << ent->d_name << endl;
    i = 0;
    cin >> output_i;
    cout << "Enter final directory name (default is Acquisition)" << endl;
    cin >> target_dir;
    if (target_dir.empty())
        target_dir = string("Acquisition");
    target_dir = string("/") + target_dir;
    if ((dir = opendir(output_dir.c_str())) != NULL)
        while ((ent = readdir(dir)) != NULL)
            if ((strcmp(ent->d_name, ".") & strcmp(ent->d_name, "..") != 0) and ++i == output_i) {
                output_dir += string(ent->d_name) + target_dir;
                break;
            }

    if (stat(output_dir.c_str(), &validator) != 0) {
        mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        cout << output_dir << " Created!" << endl;
    }
}

bool setup_JAI(StreamInfo *MyStreamInfos[3], PvDevice *&lDevice, BufferList lBufferLists[3]) {
    PvString lConnectionID;
    bool test_streaming = true;
    json jai_config = config["JAI"];
    if (PvSelectDevice(&lConnectionID)) {
        lDevice = ConnectToDevice(lConnectionID);
        if (lDevice != NULL) {
            PvGenParameterArray *lDeviceParams = lDevice->GetParameters();
            PvStream *lStreams[3] = {NULL, NULL, NULL};

            for (auto it = jai_config.begin(); it != jai_config.end(); ++it) {
                string key = it.key();
                auto value = it.value();
                if (value.is_number_integer()) {
                    int v = value.get<int>();
                    //                    lDeviceParams->SetIntegerValue(key, v);
                }
                if (value.is_number_float()) {
                    double v = value.get<double>();
                    //                    lDeviceParams->SetFloatValue(key, v);
                }
                if (value.is_string()) {
                    string v = value.get<string>();
                    //                    lDeviceParams->SetEnumValue(key, v);
                }
            }
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
    err = zed.enableRecording(params);
    if (err != ERROR_CODE::SUCCESS) {
        std::cout << toString(err) << std::endl;
        exit(-1);
    }


    zed.setCameraSettings(VIDEO_SETTINGS::GAIN, -1);
    zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, -1);
    cout << zed.getCameraSettings(VIDEO_SETTINGS::GAIN) << endl;
    cout << zed.getCameraSettings(VIDEO_SETTINGS::EXPOSURE) << endl;
}

int main() {
    cout << "**************** CODEC: nvv4l2h265enc ****************" << endl;
    cout << "**************** bitrate: 8M ****************" << endl << endl;

    PvDevice *lDevice = NULL;
    PvStream *lStreams[3] = {NULL, NULL, NULL};
    BufferList lBufferLists[3];
    StreamInfo *MyStreamInfos[3];
    ifstream config_file("/home/mic-730ai/fruitspec/JAI_Acquisition/config.json", std::ifstream::binary);

    config = json::parse(config_file);

    pthread_cond_init(&GrabEvent, NULL);

    PV_SAMPLE_INIT();

    set_output_dir();
    cout << "saving files to " << output_dir << endl;

    PvString lConnectionID;

    if (!setup_JAI(MyStreamInfos, lDevice, lBufferLists))
        exit(-1);
    while (1) {
        _abort = false;
        mp4_init = false;

        cout << "Press Enter to continue, press 'q' and then Enter to exit..." << endl;
        char c = cin.get();
        if (c == 'q')
            break;

        cout << 1 << endl;
        // Get device parameters need to control streaming
        PvGenParameterArray *lDeviceParams = lDevice->GetParameters();
        cout << 2 << endl;

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

        cout << "Press Enter to terminate acquisition" << endl;
        cin.get();

        _abort = true;

        jai_t1.join();
        jai_t2.join();
        jai_t3.join();
        cout << "JAI THREAD END" << endl;

        zed_t.join();
        cout << "ZED THREAD END" << endl;
        merge_t.join();
        cout << "MERGE THREAD END" << endl;

        mp4_FSI.release();
        mp4_BGR.release();
        mp4_800.release();
        mp4_975.release();

        cout << "WRITE: " << avg_coloring / frame_count << endl;

        // Tell the device to stop sending images.
        cout << "Sending AcquisitionStop command to the device" << endl;
        lStop->Execute();

        // Disable streaming on the device
        cout << "Disable streaming on the controller." << endl;
        lDevice->StreamDisable();
        outfile.close();
    }

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

    zed.disableRecording();
    zed.close();
    cout << "ZED ACQUISITION END" << endl;
}

void GrabThread(void *_StreamInfo) {
    uint64_t PrevBlockID = 0;
    StreamInfo *MyStreamInfo = (StreamInfo *) _StreamInfo;
    PvStream *lStream = (PvStream *) (MyStreamInfo->aStream);
    int StreamIndex = (MyStreamInfo->StreamIndex), height, width;
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
                }
                PrevBlockID = CurrentBlockID;
                height = lBuffer->GetImage()->GetHeight(), width = lBuffer->GetImage()->GetWidth();
                cv::Mat frame(height, width, CV_8U, lBuffer->GetImage()->GetDataPointer());
                lStream->QueueBuffer(lBuffer);
                EnumeratedFrame *curr_frame = new EnumeratedFrame;
                curr_frame->frame = frame;
                curr_frame->BlockID = CurrentBlockID;
                pthread_mutex_lock(&mtx);
                MyStreamInfo->Frames.push(curr_frame);
                pthread_cond_signal(&MergeFramesEvent[StreamIndex]);
                pthread_mutex_unlock(&mtx);
            } else {
                lStream->QueueBuffer(lBuffer);
//                cout << StreamIndex << ": OPR - FAILURE" << endl;
            }
            // Re-queue the buffer in the stream object
        } else {
            cout << StreamIndex << ": BAD RESULT!" << endl;
            // Retrieve buffer failure
            cout << lResult.GetCodeString().GetAscii() << "\n";
        }
//        pthread_cond_signal(&MergeFramesEvent[StreamIndex]);
        // SetEvent(MergeFramesEvent[StreamIndex]); // signals the MergeThread that the output from current stream is ready
    }
    cout << StreamIndex << ": Acquisition end with " << CurrentBlockID << endl;
}

int MP4CreateFirstTime(int height, int width, string output_dir) {
    if (!mp4_init) {
        int i = 0;
        char f_fsi[100], stat_name[100];
//        char f_rgb[100], f_800[100], f_975[100];
        struct stat buffer;
        do {
            i++;
            sprintf(stat_name, (output_dir + string("/Result_FSI_%d.mkv")).c_str(), i);
            sprintf(f_fsi, (string("\"") + output_dir + string("/Result_FSI_%d.mkv\"")).c_str(), i);
//            sprintf(f_rgb, (string("\"") + output_dir + string("/Result_RGB_%d.mkv\"")).c_str(), i);
//            sprintf(f_800, (string("\"") + output_dir + string("/Result_800_%d.mkv\"")).c_str(), i);
//            sprintf(f_975, (string("\"") + output_dir + string("/Result_975_%d.mkv\"")).c_str(), i);
        } while (stat(stat_name, &buffer) == 0);

        string gst_3c = string("appsrc ! video/x-raw, format=BGR, width=(int)") + to_string(width) + string(", height=(int)") +
                        to_string(height) + string(", framerate=(fraction)") + to_string(FPS) +
                        string("/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! "
                               "nvv4l2h265enc bitrate=8000000 ! h265parse ! matroskamux ! filesink location=");
        string gst_1c = string("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=");
        string gs_fsi = gst_3c + f_fsi, fs_rgb = gst_3c + f_rgb;
        string gs_800 = gst_1c + f_800, gs_975 = gst_1c + f_975;

        mp4_FSI.open(gs_fsi, VideoWriter::fourcc('H', '2', '6', '4'), FPS, cv::Size(width, height));
        mp4_BGR.open(gs_rgb, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height));
        mp4_800.open(gs_800, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height), false);
        mp4_975.open(gs_975, VideoWriter::fourcc('H', '2', '6', '5'), FPS, cv::Size(width, height), false);
        /*
        do {
            sprintf(filename, "%s:\\Temp\\Result_RED_%d.mkv", SelectedDrive.c_str(), i);
            i++;
        } while (stat(filename, &buffer) == 0);

        mp4_RED = VideoWriter(filename, VideoWriter::fourcc('h', '2', '6', '4'), FPS, cv::Size(width, height), false);
        */
        mp4_init = true;
        cout << endl << endl << "---- Video capturing started ----" << endl;

        return i;
    }
    return -1;
}

void MergeThread(void *_Frames) {
    queue<EnumeratedFrame *> **FramesQueue = (queue<EnumeratedFrame *> **) _Frames;
    EnumeratedFrame *e_frames[3];
    cv::Mat Frames[3], res;
    cuda::GpuMat cudaFSI;
    std::vector<cuda::GpuMat> cudaBGR(3), cudaFrames(3), cudaFrames_equalized(3), cudaFrames_normalized(3);
    std::vector<cv::Mat> images(3);
    uint64_t CurrentBlockID;
    int size0, size1, size2;
    double elapsed = 0;
    char filename[100];
    struct timespec max_wait = {0, 0};

    cout << "MERGE THREAD START" << endl;
    sleep(1);
    pthread_cond_broadcast(&GrabEvent);
    for (CurrentBlockID = 1; !_abort; CurrentBlockID++) {
        int fnum;
        for (int i = 0; i < 3; i++) {
            e_frames[i] = new EnumeratedFrame;
            pthread_mutex_lock(&mtx);
            while ((*(FramesQueue[i])).empty() and !_abort) {
                clock_gettime(CLOCK_REALTIME, &max_wait);
                max_wait.tv_sec += 1;
                const int timed_wait_rv = pthread_cond_timedwait(&MergeFramesEvent[i], &mtx, &max_wait);
            }
            if (_abort) {
                pthread_mutex_unlock(&mtx);
                break;
            }
            e_frames[i] = (*(FramesQueue[i])).front();
            Frames[i] = e_frames[i]->frame;
            fnum = e_frames[i]->BlockID;
            t0.stop();
//            cout << "STREAM " << i << " POP " << fnum <<" AFTER " << t0.getTimeMilli() << endl;
            t0.start();
            (*(FramesQueue[i])).pop();
            pthread_mutex_unlock(&mtx);
        }
        if (_abort)
            break;
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

        mp4_FSI.write(res_fsi);
        mp4_BGR.write(res_bgr);
        mp4_800.write(res_800);
        mp4_975.write(res_975);

        t.stop();
        elapsed = t.getTimeMilli();
        avg_coloring += elapsed;
    }
    cout << "MERGE END" << endl;
}

PvDisplayWnd *GetDisplay(const string &aName) {
    PvDisplayWndMap::iterator lIt = mDisplays.find(aName);
    if (lIt != mDisplays.end()) {
        return lIt->second;
    }

    // Create new display
    PvDisplayWnd *lDisplay = new PvDisplayWnd;
    if (lDisplay == NULL) {
        return NULL;
    }

    // Store display in map
    mDisplays[aName] = lDisplay;

    // Setup display
    // lDisplay->ShowModeless();
    lDisplay->SetTitle(aName.c_str());

    return lDisplay;
}

// Attempts to display the content of a buffer
void Display(PvBuffer *aBuffer) {
    stringstream lSS1;
    lSS1 << mSource.GetAscii();
    string lName = lSS1.str();

    PvDisplayWnd *lDisplay = GetDisplay(lName);
    lDisplay->Display(aBuffer->GetImage());
    // lDisplay->DoEvents();
}
