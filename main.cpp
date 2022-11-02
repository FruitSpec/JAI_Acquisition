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

using namespace cv;
using namespace cuda;
using namespace sl;
using namespace std;

typedef map<string, PvDisplayWnd*> PvDisplayWndMap;

typedef list<PvBuffer*> BufferList;

PV_INIT_SIGNAL_HANDLER();


PvDevice* ConnectToDevice(const PvString& aConnectionID);
PvStream* OpenStream(const PvString& aConnectionID);
void CreateStreamBuffers(PvDevice* aDevice, PvStream* aStream, BufferList* aBufferList);
void FreeStreamBuffers(BufferList* aBufferList);
void GrabThread(void* StreamInfo);
void ZedThread(int file_index);
void MergeThread(void* _Frames);
int MP4CreateFirstTime(int height, int width);
void Display(PvBuffer* aBuffer);
void ConfigureStream( PvDevice *aDevice, PvStream *aStream, int channel );

pthread_cond_t GrabEvent = PTHREAD_COND_INITIALIZER, MergeFramesEvent[3];
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t tmp_mtx = PTHREAD_MUTEX_INITIALIZER;
//mutex mtx;
VideoWriter mp4_FSI, mp4_RED;
cv::Ptr<cv::cudacodec::VideoWriter> cuda_mp4_FSI;

bool _abort = false, mp4_init = false;
float avg_coloring = 0;
int frame_count = 0;
PvDisplayWndMap mDisplays;
PvString mSource;
int FPS = 30;
int64_t width, height;
Camera zed;
TickMeter t0;

typedef struct
{
    cv::Mat frame;
    int BlockID;
} EnumeratedFrame;

typedef struct
{
    PvStream* aStream;
    int StreamIndex;
    queue<EnumeratedFrame*> Frames;
} StreamInfo;


std::string getCurrentTimestamp()
{
    using std::chrono::system_clock;
    auto currentTime = std::chrono::system_clock::now();
    char buffer[80];

    auto transformed = currentTime.time_since_epoch().count() / 1000000;

    auto millis = transformed % 1000;

    std::time_t tt;
    tt = system_clock::to_time_t(currentTime);
    auto timeinfo = localtime(&tt);
    strftime(buffer, 80, "%F %H:%M:%S", timeinfo);
    sprintf(buffer, "%s:%03d", buffer, (int)millis);

    return std::string(buffer);
}

void ConfigureStream( PvDevice *aDevice, PvStream *aStream, int channel )
{
    // If this is a GigE Vision device, configure GigE Vision specific streaming parameters
    PvDeviceGEV* lDeviceGEV = dynamic_cast<PvDeviceGEV *>( aDevice );
    if ( lDeviceGEV != NULL )
    {
        PvStreamGEV *lStreamGEV = static_cast<PvStreamGEV *>( aStream );

        // Negotiate packet size
        lDeviceGEV->NegotiatePacketSize(channel);

        // Configure device streaming destination
        lDeviceGEV->SetStreamDestination( lStreamGEV->GetLocalIPAddress(), lStreamGEV->GetLocalPort(), channel);
    }
}


int main()
{
    PvDevice* lDevice = NULL;
    PvStream* lStreams[3] = {NULL, NULL, NULL};
    BufferList lBufferLists[3];
    StreamInfo* MyStreamInfos[3];

    pthread_cond_init(&GrabEvent, NULL);
    for (int i = 0; i < 3; i++) {
        MyStreamInfos[i] = new StreamInfo;
        MergeFramesEvent[i] = PTHREAD_COND_INITIALIZER;
        pthread_cond_init(&MergeFramesEvent[i], NULL);
    }

    PV_SAMPLE_INIT();

//    cout << "Please enter Frame Rate:" << endl;
//    cin >> FPS_string;
//    FPS = stoi(FPS_string);

    PvString lConnectionID;
    if (PvSelectDevice(&lConnectionID))
    {
        lDevice = ConnectToDevice(lConnectionID);
        if (lDevice != NULL)
        {
            lDevice->GetParameters()->SetFloatValue("AcquisitionFrameRate", FPS);
            lDevice->GetParameters()->GetIntegerValue("Width", width);
            lDevice->GetParameters()->GetIntegerValue("Height", height);
            bool test_streaming = true;
            for (int i = 0; i < 3; i++){
                lStreams[i] = OpenStream(lConnectionID);
                if (lStreams[i] != NULL) {
                    ConfigureStream(lDevice, lStreams[i], i);
                    CreateStreamBuffers(lDevice, lStreams[i], &lBufferLists[i]);
                    MyStreamInfos[i]->aStream = lStreams[i];
                    MyStreamInfos[i]->StreamIndex = i;
                }
                else
                    test_streaming = false;
            }

            if (test_streaming) {
                // Get device parameters need to control streaming
                PvGenParameterArray *lDeviceParams = lDevice->GetParameters();

                // Map the GenICam AcquisitionStart and AcquisitionStop commands
                PvGenCommand *lStart = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStart"));
                PvGenCommand *lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));

                queue<EnumeratedFrame*> *Frames[3] = {&(MyStreamInfos[0]->Frames), &(MyStreamInfos[1]->Frames),
                                         &(MyStreamInfos[2]->Frames)};
                lDevice->StreamEnable();
                int file_index = MP4CreateFirstTime(height, width);

                // Enable streaming and send the AcquisitionStart command

                char zed_filename[100];
                InitParameters init_params;
                init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
                init_params.camera_fps = FPS; // Set fps at 30

                ERROR_CODE err = zed.open(init_params);
                if (err != ERROR_CODE::SUCCESS) {
                    std::cout << toString(err) << std::endl;
                    exit(-1);
                }

                // Enable video recording
                sprintf(zed_filename, "/home/mic-730ai/Counter/ZED_%d.svo", file_index);
                err = zed.enableRecording(RecordingParameters(zed_filename, SVO_COMPRESSION_MODE::H265));
                if (err != ERROR_CODE::SUCCESS) {
                    std::cout << toString(err) << std::endl;
                    exit(-1);
                }

                lStart->Execute();
                cout << "Enabling streaming and sending AcquisitionStart command." << endl;
                t0.start();
                thread zed_t(ZedThread, file_index);
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
                mp4_RED.release();

                cout << "WRITE: " << avg_coloring / frame_count << endl;

                // Tell the device to stop sending images.
                cout << "Sending AcquisitionStop command to the device" << endl;
                lStop->Execute();

                // Disable streaming on the device
                cout << "Disable streaming on the controller." << endl;
                lDevice->StreamDisable();

                // Abort all buffers from the streams and dequeue
                cout << "Aborting buffers still in streams" << endl << "closing streams" << endl;
                for (int i = 0; i < 3; i++){
                    lStreams[i]->AbortQueuedBuffers();
                    while (lStreams[i]->GetQueuedBufferCount() > 0) {
                        PvBuffer *lBuffer = NULL;
                        PvResult lOperationResult;
                        lStreams[i]->RetrieveBuffer(&lBuffer, &lOperationResult);
                    }
                    FreeStreamBuffers(&lBufferLists[i]);
                    lStreams[i]->Close();
                    PvStream::Free(lStreams[i]);
                }
            }

            // Disconnect the device
            cout << "Disconnecting device" << endl;
            lDevice->Disconnect();

            PvDevice::Free(lDevice);
        }
    }

    cout << endl;
    //cout << "<press a key to exit>" << endl;
    //PvWaitForKeyPress();
    // CloseHandle(GrabEvent);
    pthread_cond_destroy(&GrabEvent);
    for (int i = 0; i < 3; i++) pthread_cond_destroy(&MergeFramesEvent[i]);
    PV_SAMPLE_TERMINATE();
    return 0;
}

PvDevice* ConnectToDevice(const PvString& aConnectionID)
{
    PvDevice* lDevice;
    PvResult lResult;

    // Connect to the GigE Vision or USB3 Vision device
    cout << "Connecting to device." << endl;
    lDevice = PvDevice::CreateAndConnect(aConnectionID, &lResult);
    if (lDevice == NULL)
    {
        cout << "Unable to connect to device: " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

    return lDevice;
}

PvStream* OpenStream(const PvString& aConnectionID)
{
    PvStream* lStream;
    PvResult lResult;

    // Open stream to the GigE Vision or USB3 Vision device
    cout << "Opening stream from device." << endl;
    lStream = PvStream::CreateAndOpen(aConnectionID, &lResult);
    if (lStream == NULL)
    {
        cout << "Unable to stream from device. " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

    return lStream;
}

void CreateStreamBuffers( PvDevice *aDevice, PvStream *aStream, BufferList *aBufferList )
{
    // Reading payload size from device
    uint32_t lSize = aDevice->GetPayloadSize();

    // Use BUFFER_COUNT or the maximum number of buffers, whichever is smaller
    uint32_t lBufferCount = aStream->GetQueuedBufferMaximum();
//    ( aStream->GetQueuedBufferMaximum() < BUFFER_COUNT ) ?
//                            aStream->GetQueuedBufferMaximum() :
//                            BUFFER_COUNT;

    // Allocate buffers
    for ( uint32_t i = 0; i < lBufferCount; i++ )
    {
        // Create new buffer object
        PvBuffer *lBuffer = new PvBuffer;

        // Have the new buffer object allocate payload memory
        lBuffer->Alloc( static_cast<uint32_t>( lSize ) );

        // Add to external list - used to eventually release the buffers
        aBufferList->push_back( lBuffer );
    }

    // Queue all buffers in the stream
    BufferList::iterator lIt = aBufferList->begin();
    while ( lIt != aBufferList->end() )
    {
        aStream->QueueBuffer( *lIt );
        lIt++;
    }
}

void FreeStreamBuffers(BufferList* aBufferList)
{
    // Go through the buffer list
    BufferList::iterator lIt = aBufferList->begin();
    while (lIt != aBufferList->end())
    {
        delete* lIt;
        lIt++;
    }

    // Clear the buffer list
    aBufferList->clear();
}

void ZedThread(int file_index){
    cout << "ZED THREAD STARTED" << endl;

    // Grab data and write to SVO file
    while (!_abort)
        zed.grab();

    zed.disableRecording();
    zed.close();
    cout << "ZED ACQUISITION END" << endl;
}

void GrabThread(void* _StreamInfo)
{
    uint64_t PrevBlockID = 0;
    StreamInfo* MyStreamInfo = (StreamInfo*)_StreamInfo;
    PvStream* lStream = (PvStream*)(MyStreamInfo->aStream);
    int StreamIndex = (MyStreamInfo->StreamIndex), height, width;
    PvBuffer* lBuffer = NULL;
    char str[200];


    PvResult lResult, lOperationResult;
    uint64_t CurrentBlockID = 0;

    // WaitForSingleObject(GrabEvent, INFINITE);
    pthread_mutex_lock(&tmp_mtx);
    pthread_cond_wait(&GrabEvent, &tmp_mtx);
    pthread_mutex_unlock(&tmp_mtx);
    cout << "JAI STREAM " << StreamIndex << " STARTED" << endl;
    while (!_abort)
    {
        PvBuffer *lBuffer = NULL;
        PvResult lOperationResult;

        lResult = lStream->RetrieveBuffer(&lBuffer, &lOperationResult, 1000);

        if (lResult.IsOK())
        {
            if (lOperationResult.IsOK())
            {
                // We now have a valid buffer. This is where you would typically process the buffer.

                CurrentBlockID = lBuffer->GetBlockID();
//                if (CurrentBlockID != PrevBlockID + 1 and PrevBlockID != 0)
//                    cout << "STREAM " << StreamIndex << " MISSED BLOCK. OLD: " << PrevBlockID <<
//                    ", NEW: " << CurrentBlockID << endl;
//                PrevBlockID = CurrentBlockID;
                height = lBuffer->GetImage()->GetHeight(), width = lBuffer->GetImage()->GetWidth();
                cv::Mat frame(height, width, CV_8U, lBuffer->GetImage()->GetDataPointer());
                lStream->QueueBuffer(lBuffer);
                EnumeratedFrame* curr_frame = new EnumeratedFrame;
                curr_frame->frame = frame;
                curr_frame->BlockID = CurrentBlockID;
                pthread_mutex_lock(&mtx);
//                if (StreamIndex == 0) {
//                    t0.stop();
//                    cout << CurrentBlockID << " AFTER " << t0.getTimeMilli() << endl;
//                    t0.start();
//                }
                MyStreamInfo->Frames.push(curr_frame);
                pthread_cond_signal(&MergeFramesEvent[StreamIndex]);
                pthread_mutex_unlock(&mtx);
            }
            else{
                lStream->QueueBuffer(lBuffer);
//                cout << StreamIndex << ": OPR - FAILURE" << endl;
            }
            // Re-queue the buffer in the stream object
        }
        else
        {
            cout << StreamIndex << ": BAD RESULT!" << endl;
            // Retrieve buffer failure
            cout << lResult.GetCodeString().GetAscii() << "\n";
        }
//        pthread_cond_signal(&MergeFramesEvent[StreamIndex]);
        // SetEvent(MergeFramesEvent[StreamIndex]); // signals the MergeThread that the output from current stream is ready
    }
    cout << StreamIndex << ": Acquisition end with " << CurrentBlockID << endl;
}

int MP4CreateFirstTime(int height, int width) {
    if (!mp4_init)
    {
        int i = 0;
        char filename[100];
        struct stat buffer;
        do {
            i++;
            sprintf(filename, "/home/mic-730ai/Counter/Result_FSI_%d.mkv", i);
            // sprintf(filename, "%s:\\Temp\\Result_FSI_%d.mkv", SelectedDrive.c_str(), i);
        } while (stat(filename, &buffer) == 0);

//        mp4_FSI = VideoWriter(filename, VideoWriter::fourcc('h', '2', '6', '4'), FPS, cv::Size(width, height), true);
        string gstreamer = string("appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=")
                + filename;
        mp4_FSI.open(gstreamer, 0, FPS, cv::Size(width, height));
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

void MergeThread(void* _Frames) {
    queue<EnumeratedFrame*>** FramesQueue = (queue<EnumeratedFrame*> **)_Frames;
    EnumeratedFrame* e_frames[3];
    cv::Mat Frames[3], res;
    cuda::GpuMat cudaFSI;
    std::vector<GpuMat> cudaFrames(3);
    std::vector<cv::Mat> images(3);
    uint64_t CurrentBlockID;
    int size0, size1, size2;
    double elapsed = 0;
    char filename[100];
    struct timespec max_wait = {0, 0};

    cout << "MERGE THREAD START" << endl;
    sleep(1);
    pthread_cond_broadcast(&GrabEvent);
    for (CurrentBlockID = 1; !_abort; CurrentBlockID++)
    {
        int fnum;
        for (int i = 0; i < 3; i++) {
            e_frames[i] = new EnumeratedFrame;
            pthread_mutex_lock(&mtx);
            while ((*(FramesQueue[i])).empty() and !_abort) {
                clock_gettime(CLOCK_REALTIME, &max_wait);
                max_wait.tv_sec += 1;
                const int timed_wait_rv = pthread_cond_timedwait(&MergeFramesEvent[i], &mtx, &max_wait);
            }
            if (_abort)
                break;
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
        cv::Mat frame_BGR, res_FSI, blue, red;
        // the actual bayer format we use is RGGB but OpenCV reffers to it as BayerBG - https://github.com/opencv/opencv/issues/19629

        cudaFrames[0].upload(Frames[0]); // channel 0 = BayerBG8
        cudaFrames[1].upload(Frames[2]); // channel 2 = 975nm -> Green
        cudaFrames[2].upload(Frames[1]); // channel 1 = 800nm -> Red

//        cv::cuda::demosaicing(cudaFrames[0], cudaFrames[0], cv::COLOR_BayerBG2BGR);
//        cv::cuda::split(cudaFrames[0], cudaBGR);
//        cudaFrames[0] = cudaBGR[0]; // final blue extraction

//        cv::merge(Frames, 3, res_FSI); // without CUDA
        cv::cuda::merge(cudaFrames, cudaFSI); // with CUDA
//        cv::cuda::cvtColor(cudaFSI, cudaFSI_LAB, COLOR_BGR2Lab);
//        cv::cuda::split(cudaFSI_LAB, cudaLAB);
//        cv::Ptr<cv::CLAHE> clahe = cv::cuda::createCLAHE();
//        clahe->setClipLimit(4);
//        clahe->apply(cudaLAB[0], cudaLAB_equalized[0]);
//        cudaLAB_equalized[1] = cudaLAB[1];
//        cudaLAB_equalized[2] = cudaLAB[2];
//        cv::cuda::merge(cudaLAB_equalized, cudaFSI_LAB);
//        cv::cuda::cvtColor(cudaFSI_LAB, cudaFSI, COLOR_Lab2BGR);
        cudaFSI.download(res_FSI);
        // cudaBGR[2].download(red);
        frame_count++;
        if (frame_count % (FPS * 30) == 0) cout << endl << frame_count / (FPS * 60.0) << " minutes of video written" << endl << endl;

        TickMeter t;
//        cout << "WRITE START " << fnum << endl;
        t.start();
        mp4_FSI.write(res_FSI);
        t.stop();
//        cout << "WRITE END " << fnum << " - " << t.getTimeMilli() << endl;
        // mp4_RED.write(red);
        elapsed = t.getTimeMilli();
        avg_coloring += elapsed;
    }
    cout << "MERGE END" << endl;
}

PvDisplayWnd* GetDisplay(const string& aName)
{
    PvDisplayWndMap::iterator lIt = mDisplays.find(aName);
    if (lIt != mDisplays.end())
    {
        // Return existing display
        return lIt->second;
    }

    // Create new display
    PvDisplayWnd* lDisplay = new PvDisplayWnd;
    if (lDisplay == NULL)
    {
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
void Display(PvBuffer* aBuffer)
{
    stringstream lSS1;
    lSS1 << mSource.GetAscii();
    string lName = lSS1.str();

    PvDisplayWnd* lDisplay = GetDisplay(lName);
    lDisplay->Display(aBuffer->GetImage());
    // lDisplay->DoEvents();
}
