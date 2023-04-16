#include "acquisition.hpp"

using namespace cv;
using namespace cuda;
using namespace sl;
using namespace std;

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

PvDevice *ConnectToDevice(const PvString &aConnectionID, bool debug) {
    PvDevice *lDevice;
    PvResult lResult;

    // Connect to the GigE Vision or USB3 Vision device
    if (debug)
        cout << "Connecting to device" << endl;
    lDevice = PvDevice::CreateAndConnect(aConnectionID, &lResult);
    if (lDevice == nullptr and debug) {
        cout << "Unable to connect to device: " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

    return lDevice;
}

PvStream *OpenStream(const PvString &aConnectionID, bool debug) {
    PvStream *lStream;
    PvResult lResult;

    // Open stream to the GigE Vision or USB3 Vision device
    lStream = PvStream::CreateAndOpen(aConnectionID, &lResult);
    if (lStream == NULL and debug) {
        cout << "Unable to stream from device. " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

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

VideoConfig * parse_args(short bit_depth, short fps, short exposure_rgb = 500, short exposure_800 = 1000, short exposure_975 = 3000,
                         const string & output_dir = string("/home/mic-730ai/Desktop/JAI_Results"), bool output_fsi = false,
                         bool output_rgb = false, bool output_800 = false, bool output_975 = false,
                         bool output_svo = false, bool view = false, bool use_clahe_stretch = false,
                         bool debug_mode = false){
    auto *video_conf = new VideoConfig;

    video_conf->bit_depth = bit_depth;
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
    if (bit_depth != 8)
        video_conf->use_clahe_stretch = true;
    else
        video_conf->use_clahe_stretch = use_clahe_stretch;


    if (debug_mode) {
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
    }
    return video_conf;
}

void set_acquisition_parameters(AcquisitionParameters &acq){
    PvGenParameterArray *lDeviceParams = acq.lDevice->GetParameters();
    PvStream *lStreams[3] = {nullptr, nullptr, nullptr};

    PvString mono = "Mono8", bayer = "BayerRG8";
    if (acq.video_conf->bit_depth == 8) {
        mono = "Mono12";
        bayer = "BayerRG12";
    }

    lDeviceParams->SetEnumValue("SourceSelector", "Source0");
    lDeviceParams->SetEnumValue("PixelFormat", bayer);
    lDeviceParams->SetFloatValue("ExposureAutoControlMax", acq.video_conf->exposure_rgb);
    lDeviceParams->SetEnumValue("SourceSelector", "Source1");
    lDeviceParams->SetEnumValue("PixelFormat", mono);
    lDeviceParams->SetFloatValue("ExposureAutoControlMax", acq.video_conf->exposure_800);
    lDeviceParams->SetEnumValue("SourceSelector", "Source2");
    lDeviceParams->SetEnumValue("PixelFormat", mono);
    lDeviceParams->SetFloatValue("ExposureAutoControlMax", acq.video_conf->exposure_975);
    lDeviceParams->SetEnumValue("SourceSelector", "Source0");
    lDeviceParams->SetFloatValue("AcquisitionFrameRate", acq.video_conf->FPS);
    lDeviceParams->GetIntegerValue("Width", acq.video_conf->width);
    lDeviceParams->GetIntegerValue("Height", acq.video_conf->height);
}

bool setup_JAI(AcquisitionParameters &acq) {
    PvString lConnectionID;
    bool test_streaming = true;
    if (SelectDeviceLocally(&lConnectionID)) {
        acq.lDevice = ConnectToDevice(lConnectionID, acq.debug);
        if (acq.lDevice != NULL) {
            PvStream *lStreams[3] = {NULL, NULL, NULL};
            for (int i = 0; i < 3; i++) {
                acq.MyStreamInfos[i] = new StreamInfo;
                acq.MergeFramesEvent[i] = PTHREAD_COND_INITIALIZER;
                pthread_cond_init(&acq.MergeFramesEvent[i], NULL);

                lStreams[i] = OpenStream(lConnectionID, acq.debug);
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

void MP4CreateFirstTime(AcquisitionParameters &acq){
    bool is_exist = false;
    short i = 0;
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

    char log_filename[100];
    sprintf(log_filename, (acq.video_conf->output_dir + string("/frame_drop_%d.log")).c_str(), i);
    acq.frame_drop_log_file.open(log_filename, std::ios_base::app); // append instead of overwrite

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

string gs_sink_builder(int file_index, const string& output_type_name, const string& output_dir){
    string gs_loc = string("\"") + output_dir;
    gs_loc += string("/Result_") + output_type_name + "_" + to_string(file_index) + string(".mkv\"");
    return gs_loc;
}

bool exists(char path[100]){
    struct stat buffer{};
    return stat(path, &buffer) == 0;
}

bool connect_ZED(AcquisitionParameters &acq, int fps){
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = fps; // Set fps
    ERROR_CODE err = acq.zed.open(init_params);
    return err == ERROR_CODE::SUCCESS;
}

void ZedThread(AcquisitionParameters &acq) {
    // Grab ZED data and write to SVO file

    for (int zed_frame_number = 0; acq.is_running; zed_frame_number++) {
        EnumeratedZEDFrame zed_frame;
        ERROR_CODE err = acq.zed.grab();
        if (err != ERROR_CODE::SUCCESS) {
            if (acq.debug)
                acq.frame_drop_log_file << "ZED FRAME DROP - FRAME NO. " << zed_frame_number << endl;
        }
        auto frame_loc = static_cast<short>(zed_frame_number % BATCH_SIZE);
        acq.zed.retrieveMeasure(zed_frame.frame, MEASURE::XYZRGBA);
        zed_frame.BlockID = zed_frame_number;
        acq.jz_streamer.push_zed(zed_frame);
    }

    if (acq.video_conf->output_svo)
        acq.zed.disableRecording();
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
    if (acq.debug)
        cout << "JAI STREAM " << stream_index << " STARTED" << endl;
    short cv_bit_depth = acq.video_conf->bit_depth == 8 ? CV_8U : CV_16U;
    while (acq.is_running) {
        PvBuffer *lBuffer = nullptr;
        lResult = lStream->RetrieveBuffer(&lBuffer, &lOperationResult, 1000);

        if (lResult.IsOK()) {
            if (lOperationResult.IsOK()) {
                // We now have a valid buffer. This is where you would typically process the buffer.

                CurrentBlockID = lBuffer->GetBlockID();
                if (CurrentBlockID != PrevBlockID + 1 and PrevBlockID != 0) {
                    acq.frame_drop_log_file << "JAI STREAM " << stream_index << " - FRAME DROP - FRAME No. " << PrevBlockID << endl;
                }
                PrevBlockID = CurrentBlockID;
                height = lBuffer->GetImage()->GetHeight(), width = lBuffer->GetImage()->GetWidth();
                cv::Mat frame(height, width, cv_bit_depth, lBuffer->GetImage()->GetDataPointer());
                lStream->QueueBuffer(lBuffer);
                EnumeratedJAIFrame *curr_frame = new EnumeratedJAIFrame;
                curr_frame->frame = frame;
                curr_frame->BlockID = CurrentBlockID;
                pthread_mutex_lock(&acq.grab_mtx);
                MyStreamInfo->Frames.push(curr_frame);
                pthread_cond_signal(&acq.MergeFramesEvent[stream_index]);
                pthread_mutex_unlock(&acq.grab_mtx);
            } else {
                lStream->QueueBuffer(lBuffer);
                if (acq.debug)
                    cout << stream_index << ": OPR - FAILURE" << endl;
            }
            // Re-queue the buffer in the stream object
        } else if (acq.debug) {
            cout << stream_index << ": BAD RESULT!" << endl;
            // Retrieve buffer failure
            cout << lResult.GetCodeString().GetAscii() << "\n";
        }
    }
    if (acq.debug)
        cout << stream_index << ": Acquisition end with " << CurrentBlockID << endl;
}

void stretch(cuda::GpuMat& channel, double lower = 0.02, double upper = 0.98, double min_int = 25, double max_int = 235) {
    const int hist_size = 256;
    int lower_threshold = 0, upper_threshold = 255;
    double percentile = 0.0;

    cuda::normalize(channel, channel, 0, 255, NORM_MINMAX, CV_8U);
    cuda::GpuMat gpu_hist;
    cv::Mat cpu_hist;

    cuda::calcHist(channel, gpu_hist);
    gpu_hist.download(cpu_hist);
    cv::Scalar total = cuda::sum(gpu_hist);

    // Calculate lower and upper threshold indices for gain

    for (int i = 0; i < hist_size; i++) {
        percentile += cpu_hist.at<int>(0, i) / total[0];
        if (percentile >= lower) {
            lower_threshold = i;
            break;
        }
    }

    percentile = 0.0;
    for (int i = hist_size - 1; i > 0; i--) {
        percentile += cpu_hist.at<int>(0, i) / total[0];
        if (percentile <= upper) {
            upper_threshold = i;
            break;
        }
    }

    double gain = (max_int - min_int) / upper_threshold;
    double offset = min_int;

    cuda::multiply(channel, gain, channel);
    cuda::add(channel, offset, channel);

    cv::Scalar lower_scalar(0);
    cv::Scalar upper_scalar(255);

    // Clipping to range [0, 255]
    cv::cuda::max(lower_scalar, channel, channel);
    cv::cuda::min(upper_scalar, channel, channel);
}

void stretch_and_clahe(cuda::GpuMat& channel, const Ptr<cuda::CLAHE>& clahe, double lower = 0.02, double upper = 0.98, double min_int = 25,
                       double max_int = 235) {
    stretch(channel, lower, upper, min_int, max_int);
    clahe->apply(channel, channel);
}

void fsi_from_channels(cv::Ptr<cuda::CLAHE> clahe, cuda::GpuMat& blue, cuda::GpuMat& c_800, cuda::GpuMat& c_975, cuda::GpuMat &fsi,
                       double lower = 0.02, double upper = 0.98, double min_int = 25, double max_int = 235) {
    stretch_and_clahe(blue, clahe, lower, upper, min_int, max_int);
    stretch_and_clahe(c_800, clahe, lower, upper, min_int, max_int);
    stretch_and_clahe(c_975, clahe, lower, upper, min_int, max_int);

    std::vector<cv::cuda::GpuMat> channels;
    channels.push_back(blue);
    channels.push_back(c_800);
    channels.push_back(c_975);

    cuda::merge(channels, fsi);
}

void MergeThread(AcquisitionParameters &acq) {

    cv::Mat Frames[3], res;
    cuda::GpuMat cudaFSI;
    std::vector<cuda::GpuMat> cudaBGR(3), cudaFrames(3);
    std::vector<cv::Mat> images(3);
    int frame_count = 0;
    struct timespec max_wait = {0, 0};
    EnumeratedJAIFrame *e_frames[3] = {new EnumeratedJAIFrame, new EnumeratedJAIFrame, new EnumeratedJAIFrame};
    bool grabbed[3] = { false, false, false };

    if (acq.debug)
        cout << "MERGE THREAD START" << endl;
    sleep(1);
    pthread_cond_broadcast(&acq.GrabEvent);
    cv::Ptr<cuda::CLAHE> clahe = cv::cuda::createCLAHE(5, cv::Size(10, 10));
    for (int frame_no = 0; acq.is_running; frame_no++) {
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
            acq.frame_drop_log_file << "MERGE DROP - AFTER FRAME NO. " << --frame_no << endl;
            for (int i = 0; i < 3; i++) {
                if (e_frames[i]->BlockID == max_id)
                    grabbed[i] = true;
            }
            continue;
        }
        cv::Mat res_fsi, res_bgr;
        // the actual bayer format we use is RGGB (or - BayerRG) but OpenCV refers to it as BayerBG
        // for more info look at - https://github.com/opencv/opencv/issues/19629

        cudaFrames[0].upload(Frames[0]); // channel 0 = BayerBG8
        cudaFrames[2].upload(Frames[1]); // channel 1 = 800nm -> Red
        cudaFrames[1].upload(Frames[2]); // channel 2 = 975nm -> Green

        cv::cuda::demosaicing(cudaFrames[0], cudaFrames[0], cv::COLOR_BayerBG2BGR);
        cv::cuda::split(cudaFrames[0], cudaBGR);
        if (acq.video_conf->output_rgb)
            cudaFrames[0].download(res_bgr);

        cudaFrames[0] = cudaBGR[2]; // just pick the blue from the bayer

        if (acq.video_conf->use_clahe_stretch){
            fsi_from_channels(clahe, cudaFrames[0], cudaFrames[1], cudaFrames[2], cudaFSI);
        }
        else {
            for (int i = 0; i < 3; i++) {
                cv::cuda::equalizeHist(cudaFrames[i], cudaFrames[i]);
                cv::cuda::normalize(cudaFrames[i], cudaFrames[i], 0, 255, cv::NORM_MINMAX, CV_8U);
            }
            cv::cuda::merge(cudaFrames, cudaFSI);
        }

        cudaFSI.download(res_fsi);

        int BlockID = e_frames[0]->BlockID;
        EnumeratedJAIFrame e_frame_fsi = {res_fsi, BlockID};
        acq.jz_streamer.push_jai(e_frame_fsi);

        frame_count++;
        if (frame_count % (acq.video_conf->FPS * 30) == 0 and acq.debug)
            cout << endl << frame_count / (acq.video_conf->FPS * 60.0) << " minutes of video written" << endl << endl;

        if (acq.video_conf->output_fsi)
            acq.mp4_FSI.write(cudaFSI);
        if (acq.video_conf->output_rgb)
            acq.mp4_BGR.write(res_bgr);
        if (acq.video_conf->output_800)
            acq.mp4_800.write(Frames[1]);
        if (acq.video_conf->output_975)
            acq.mp4_975.write(Frames[2]);
    }
}

JaiZedStatus connect_cameras(AcquisitionParameters &acq, int fps){
    PvStream *lStreams[3] = {NULL, NULL, NULL};

    // check if this command is necessary
    PV_SAMPLE_INIT();

    pthread_cond_init(&acq.GrabEvent, NULL);

    JaiZedStatus jzs {
        setup_JAI(acq),
        connect_ZED(acq, fps)
    };

    if (acq.debug){
        cout << (jzs.jai_connected ? "JAI CONNECTED" : "JAI NOT CONNECTED") << endl;
        cout << (jzs.zed_connected ? "ZED CONNECTED" : "ZED NOT CONNECTED") << endl;
    }

    acq.is_connected = jzs.jai_connected and jzs.zed_connected;
    return jzs;
}

void start_acquisition(AcquisitionParameters &acq) {
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStart;

    if (acq.is_connected) {
        set_acquisition_parameters(acq);

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
    }
    else if (acq.debug)
        cout << "NOT CONNECTED" << endl;
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
    // disconnect ZED
    acq.zed.close();

    // Abort all buffers from the streams and dequeue
    if (acq.debug)
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

    acq.is_connected = false;
    // check if necessary
    PV_SAMPLE_TERMINATE();
}