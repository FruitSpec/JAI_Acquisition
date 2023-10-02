#include "acquisition.hpp"


bool SelectDeviceLocally(PvString *aConnectionID)
{
    PvResult lResult;
    const PvDeviceInfo *lSelectedDI = nullptr;
    PvSystem lSystem;

        lSystem.Find();

        // Detect, select device
        vector<const PvDeviceInfo *> lDIVector;
        for ( uint32_t i = 0; i < lSystem.GetInterfaceCount(); i++ ) {
            const auto *lInterface = dynamic_cast<const PvInterface *>( lSystem.GetInterface( i ) );
            if ( lInterface != nullptr ) {
                for ( uint32_t j = 0; j < lInterface->GetDeviceCount(); j++ ) {
                    const auto *lDI = dynamic_cast<const PvDeviceInfo *>( lInterface->GetDeviceInfo( j ) );
                    if ( lDI != nullptr ){
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
    if (lStream == nullptr and debug) {
        cout << "Unable to stream from device. " << lResult.GetCodeString().GetAscii()
             << " (" << lResult.GetDescription().GetAscii() << ")" << endl;
    }

    return lStream;
}

void ConfigureStream(PvDevice *&aDevice, PvStream *aStream, int channel) {
    // If this is a GigE Vision device, configure GigE Vision specific streaming parameters
    auto *lDeviceGEV = dynamic_cast<PvDeviceGEV *>( aDevice );
    if (lDeviceGEV != nullptr) {
        auto *lStreamGEV = dynamic_cast<PvStreamGEV *>( aStream );

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
        auto *lBuffer = new PvBuffer;

        // Have the new buffer object allocate payload memory
        lBuffer->Alloc(static_cast<uint32_t>( lSize ));

        // Add to external list - used to eventually release the buffers
        aBufferList->push_back(lBuffer);
    }

    // Queue all buffers in the stream
    auto lIt = aBufferList->begin();
    while (lIt != aBufferList->end()) {
        aStream->QueueBuffer(*lIt);
        lIt++;
    }
}

void FreeStreamBuffers(BufferList *aBufferList) {
    // Go through the buffer list
    auto lIt = aBufferList->begin();
    while (lIt != aBufferList->end()) {
        delete *lIt;
        lIt++;
    }

    // Clear the buffer list
    aBufferList->clear();
}

VideoConfig * parse_args(short fps, short exposure_rgb, short exposure_800, short exposure_975,
                         const string& output_dir, bool output_clahe_fsi, bool output_equalize_hist_fsi,
                         bool output_rgb, bool output_800, bool output_975, bool output_svo, bool output_zed_gray,
                         bool output_zed_depth, bool output_zed_pc, bool view, bool transfer_data,
                         bool pass_clahe_stream, bool debug_mode, std::vector<PvString> alc_true_areas,
                         std::vector<PvString> alc_false_areas) {
    auto *video_conf = new VideoConfig;

    video_conf->FPS = fps;
    video_conf->exposure_rgb = exposure_rgb;
    video_conf->exposure_800 = exposure_800;
    video_conf->exposure_975 = exposure_975;
    video_conf->output_dir = output_dir;
    video_conf->output_clahe_fsi = output_clahe_fsi;
    video_conf->output_equalize_hist_fsi = output_equalize_hist_fsi;
    video_conf->output_rgb = output_rgb;
    video_conf->output_800 = output_800;
    video_conf->output_975 = output_975;
    video_conf->output_svo = output_svo;
    video_conf->output_zed_gray = output_zed_gray;
    video_conf->output_zed_depth = output_zed_depth;
    video_conf->output_zed_pc = output_zed_pc;
    video_conf->view = view;
    video_conf->transfer_data = transfer_data;
    video_conf->pass_clahe_stream = pass_clahe_stream;
    video_conf->alc_true_areas = alc_true_areas;
    video_conf->alc_false_areas = alc_false_areas;

    if (debug_mode) {
        std::cout << "FPS: " << video_conf->FPS << std::endl;
        if (video_conf->view)
            std::cout << "view mode: on" << std::endl;
        else {
            std::cout << "view mode: off" << std::endl;
            std::cout << "output-clahe-fsi: " << std::boolalpha << video_conf->output_clahe_fsi << std::endl;
            std::cout << "output-equalize-hist-fsi: " << std::boolalpha << video_conf->output_equalize_hist_fsi << std::endl;
            std::cout << "output-rgb: " << std::boolalpha << video_conf->output_rgb << std::endl;
            std::cout << "output-800: " << std::boolalpha << video_conf->output_800 << std::endl;
            std::cout << "output-975: " << std::boolalpha << video_conf->output_975 << std::endl;
            std::cout << "output-svo: " << std::boolalpha << video_conf->output_svo << std::endl;
            std::cout << "output-zed-rgb: " << std::boolalpha << video_conf->output_zed_gray << std::endl;
            std::cout << "output-zed-depth: " << std::boolalpha << video_conf->output_zed_depth << std::endl;
            std::cout << "output-zed-point-cloud: " << std::boolalpha << video_conf->output_zed_pc << std::endl;
            std::cout << "output-dir: " << std::boolalpha << video_conf->output_dir << std::endl;
        }
    }
    return video_conf;
}

void set_parameters_per_source(PvGenParameterArray *&lDeviceParams, const PvString& source, int auto_exposure_max,
                               const PvString &pixel_format, std::vector<PvString> alc_true_areas,
                               std::vector<PvString> alc_false_areas){
//    const PvString alc_true_areas[4] = {"HighMidLeft", "LowMidLeft", "MidHighMidLeft", "MidLowMidLeft"};
//    const PvString alc_false_areas[12] = {"HighRight", "HighMidRight", "HighLeft",
//                                         "MidHighRight", "MidHighMidRight", "MidHighLeft",
//                                         "MidLowRight", "MidLowMidRight", "MidLowLeft",
//                                         "LowRight", "LowMidRight", "LowLeft"
//    };

    lDeviceParams->SetEnumValue("SourceSelector", source);
    lDeviceParams->SetEnumValue("PixelFormat", pixel_format);

    lDeviceParams->SetEnumValue("ExposureAuto", "Continuous");
    lDeviceParams->SetFloatValue("ExposureAutoControlMax", auto_exposure_max);
    lDeviceParams->SetBooleanValue("ALCAreaEnableAll", false);

    for (const PvString& alc_true_area : alc_true_areas) {
        lDeviceParams->SetEnumValue("ALCAreaSelector", alc_true_area);
        lDeviceParams->SetBooleanValue("ALCAreaEnable", true);
    }

    for (const PvString& alc_false_area : alc_false_areas) {
        lDeviceParams->SetEnumValue("ALCAreaSelector", alc_false_area);
        lDeviceParams->SetBooleanValue("ALCAreaEnable", false);
    }
}

void set_acquisition_parameters(AcquisitionParameters &acq) {
    PvGenParameterArray *lDeviceParams = acq.lDevice->GetParameters();
    PvStream *lStreams[3] = {nullptr, nullptr, nullptr};
    const PvString source_0 = "Source0", source_1 = "Source1", source_2 = "Source2";
    const PvString color = "BayerRG8", mono = "Mono8";

    set_parameters_per_source(lDeviceParams, source_0, acq.video_conf->exposure_rgb, color,
                              acq.video_conf->alc_true_areas, acq.video_conf->alc_false_areas);
    set_parameters_per_source(lDeviceParams, source_1, acq.video_conf->exposure_800, mono,
                              acq.video_conf->alc_true_areas, acq.video_conf->alc_false_areas);
    set_parameters_per_source(lDeviceParams, source_2, acq.video_conf->exposure_975, mono,
                              acq.video_conf->alc_true_areas, acq.video_conf->alc_false_areas);

    lDeviceParams->SetFloatValue("AcquisitionFrameRate", acq.video_conf->FPS);
    lDeviceParams->GetIntegerValue("Width", acq.video_conf->width);
    lDeviceParams->GetIntegerValue("Height", acq.video_conf->height);
}

bool setup_JAI(AcquisitionParameters &acq) {
    PvString lConnectionID;
    bool test_streaming = true;
    if (SelectDeviceLocally(&lConnectionID)) {
        acq.lDevice = ConnectToDevice(lConnectionID, acq.debug);
        if (acq.lDevice != nullptr) {
            PvStream *lStreams[3] = {nullptr, nullptr, nullptr};
            for (int i = 0; i < 3; i++) {
                acq.MyStreamInfos[i] = new StreamInfo;
                acq.MergeFramesEvent[i] = PTHREAD_COND_INITIALIZER;
                pthread_cond_init(&acq.MergeFramesEvent[i], nullptr);

                lStreams[i] = OpenStream(lConnectionID, acq.debug);
                if (lStreams[i] != nullptr) {
                    ConfigureStream(acq.lDevice, lStreams[i], i);
//                    CreateStreamBuffers(acq.lDevice, lStreams[i], &acq.lBufferLists[i]);
                    acq.MyStreamInfos[i]->aStream = lStreams[i];
                    acq.MyStreamInfos[i]->stream_index = i;
                } else
                    test_streaming = false;
            }

//            acq.lDevice->StreamEnable();
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
    short file_index = 0;
    string f_fsi_clahe, f_fsi_equalize_hist, f_rgb, f_800, f_975, f_zed_rgb, f_zed_depth, f_zed_X, f_zed_Y, f_zed_Z;
    string width_s, height_s, FPS_s, file_index_s;
    string frame_drop_log_path, imu_log_path, jai_acquisition_log_path;
    string zed_svo_filename;

    acq.video_conf->file_index = - 1;
    if (acq.video_conf->output_clahe_fsi or acq.video_conf->output_equalize_hist_fsi or acq.video_conf->output_rgb or acq.video_conf->output_800 or acq.video_conf->output_975)
    {
        width_s = to_string(acq.video_conf->width);
        height_s = to_string(acq.video_conf->height);
        FPS_s = to_string(acq.video_conf->FPS);

        acq.video_conf->file_index = file_index;

        jai_acquisition_log_path = acq.video_conf->output_dir + "/jai_acquisition.log";
        frame_drop_log_path = acq.video_conf->output_dir + "/frame_drop.log";
        imu_log_path = acq.video_conf->output_dir + "/imu.log";

        acq.jai_acquisition_log.open(jai_acquisition_log_path, ios_base::app);
        acq.frame_drop_log_file.open(frame_drop_log_path, ios_base::app);
        acq.imu_log_file.open(imu_log_path, ios_base::app);

        string gst_3c = "appsrc ! video/x-raw, format=BGR, width=(int)" + width_s + ", height=(int)" + height_s;
        gst_3c += ", framerate=(fraction)" + FPS_s + "/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv !";
        gst_3c += "nvv4l2h265enc bitrate=15000000 ! h265parse ! matroskamux ! filesink location=";

        string gst_1c = "appsrc ! autovideoconvert ! omxh265enc ! matroskamux ! filesink location=";

        cv::Size jai_frame_size(acq.video_conf->width, acq.video_conf->height);
        cv::Size zed_frame_size(1920, 1080);

        int four_c = VideoWriter::fourcc('H', '2', '6', '5');
        if (acq.video_conf->output_clahe_fsi) {
            f_fsi_clahe = gs_sink_builder("FSI_CLAHE", acq.video_conf->output_dir);
            string gs_clahe_fsi = gst_3c + f_fsi_clahe;
            acq.mp4_clahe_FSI.open(gs_clahe_fsi, four_c, acq.video_conf->FPS, jai_frame_size);
        }
        if (acq.video_conf->output_equalize_hist_fsi) {
            f_fsi_equalize_hist = gs_sink_builder("Result_FSI", acq.video_conf->output_dir);
            string gs_equalize_hist_fsi = gst_3c + f_fsi_equalize_hist;
            acq.mp4_equalize_hist_FSI.open(gs_equalize_hist_fsi, four_c, acq.video_conf->FPS, jai_frame_size);
        }
        if (acq.video_conf->output_rgb) {
            f_rgb = gs_sink_builder("Result_RGB", acq.video_conf->output_dir);
            string gs_rgb = gst_3c + f_rgb;
            acq.mp4_BGR.open(gs_rgb, four_c, acq.video_conf->FPS, jai_frame_size);
        }
        if (acq.video_conf->output_800) {
            f_800 = gs_sink_builder("Result_800", acq.video_conf->output_dir);
            string gs_800 = gst_1c + f_800;
            acq.mp4_800.open(gs_800, four_c, acq.video_conf->FPS, jai_frame_size, false);
        }
        if (acq.video_conf->output_975) {
            f_975 = gs_sink_builder("Result_975", acq.video_conf->output_dir);
            string gs_975 = gst_1c + f_975;
            acq.mp4_975.open(gs_975, four_c, acq.video_conf->FPS, jai_frame_size, false);
        }
        if (acq.video_conf->output_svo) {
            zed_svo_filename = (acq.video_conf->output_dir + string("/ZED.svo"));
            RecordingParameters params(zed_svo_filename.c_str(), SVO_COMPRESSION_MODE::H265);
            acq.zed.enableRecording(params);
        }
        if (acq.video_conf->output_zed_gray) {
            f_zed_rgb = gs_sink_builder("ZED", acq.video_conf->output_dir);
            string gs_zed_rgb = gst_1c + f_zed_rgb;
            acq.mp4_zed_rgb.open(gs_zed_rgb, four_c, 15, zed_frame_size, false);
        }
        if (acq.video_conf->output_zed_depth) {
            f_zed_depth = gs_sink_builder("DEPTH", acq.video_conf->output_dir);
            string gs_zed_depth = gst_1c + f_zed_depth;
            acq.mp4_zed_depth.open(gs_zed_depth, four_c, 15, zed_frame_size, false);
        }
        if (acq.video_conf->output_zed_pc){
            f_zed_X = gs_sink_builder("ZED_X", acq.video_conf->output_dir);
            f_zed_Y = gs_sink_builder("ZED_Y", acq.video_conf->output_dir);
            f_zed_Z = gs_sink_builder("ZED_Z", acq.video_conf->output_dir);
            string gs_zed_X = gst_1c + f_zed_X;
            string gs_zed_Y = gst_1c + f_zed_Y;
            string gs_zed_Z = gst_1c + f_zed_Z;
            acq.mp4_zed_X.open(gs_zed_X, four_c, 15, zed_frame_size, false);
            acq.mp4_zed_Y.open(gs_zed_Y, four_c, 15, zed_frame_size, false);
            acq.mp4_zed_Z.open(gs_zed_Z, four_c, 15, zed_frame_size, false);
        }
    }
}

string gs_sink_builder(const string& output_type_name, const string& output_dir){
    string gs_loc = "\"" + output_dir;
    gs_loc += "/" + output_type_name + ".mkv\"";
    return gs_loc;
}

bool connect_ZED(AcquisitionParameters &acq, int fps){
    InitParameters init_params;
    init_params.camera_resolution = RESOLUTION::HD1080; // Use HD1080 video mode
    init_params.camera_fps = 15; // Set fps
    init_params.depth_mode = DEPTH_MODE::QUALITY;
    init_params.coordinate_units = UNIT::METER;
    init_params.depth_minimum_distance = 0.5f;
    init_params.depth_maximum_distance = 8.0f;
    init_params.depth_stabilization = true;
    ERROR_CODE err = acq.zed.open(init_params);
    acq.zed_connected = err == ERROR_CODE::SUCCESS;
    return acq.zed_connected;
}

void ZedThread(AcquisitionParameters &acq) {
    sl::SensorsData sensors_data;
    sl::Mat zed_gpu_rgb, zed_gpu_depth, zed_gpu_point_cloud;
    cv::cuda::GpuMat cuda_gpu_rgb, cuda_gpu_gray, cuda_gpu_depth, cuda_gpu_point_cloud;
    cv::Mat cuda_gray, cuda_depth;
    std::vector<cuda::GpuMat> cudaXYZ(3);
    cv::Mat cvXYZ[3];
    EnumeratedZEDFrame zed_frame;
    cuda::Stream stream_rgb, stream_depth, stream_pc[3];

    bool first = true;
    int width = 1920, height = 1080;
    ERROR_CODE err;

    if (acq.debug)
        cout << "ZED THREAD STARTED" << endl;

    for (int zed_frame_number = 0; acq.is_running; zed_frame_number++) {
        err = acq.zed.grab();
        zed_frame.timestamp = get_current_time();
        if (err != ERROR_CODE::SUCCESS) {
            if (acq.debug)
                acq.frame_drop_log_file << "ZED FRAME DROP - FRAME NO. " << zed_frame_number << endl;
        }
        if (acq.video_conf->transfer_data) {
            acq.zed.retrieveImage(zed_frame.rgb, VIEW::LEFT);
            acq.zed.getSensorsData(sensors_data, TIME_REFERENCE::IMAGE);
            acq.zed.retrieveMeasure(zed_frame.point_cloud, MEASURE::XYZ);
            acq.zed.retrieveMeasure(zed_frame.depth, MEASURE::DEPTH);
            zed_frame.BlockID = zed_frame_number;
            zed_frame.imu = sensors_data.imu;
            std::stringstream imu;
            imu << get_current_time() << endl << zed_frame.imu.angular_velocity << endl
                << zed_frame.imu.linear_acceleration;
            acq.imu_log_file << imu.str() << endl << endl;
            acq.jz_streamer.push_zed(zed_frame);
        }
        if (acq.video_conf->output_zed_gray) {
            acq.zed.retrieveImage(zed_gpu_rgb, VIEW::LEFT, MEM::GPU);
            auto rgb_ptr = zed_gpu_rgb.getPtr<sl::uchar1>(MEM::GPU);
            auto rgb_step = zed_gpu_rgb.getStepBytes(MEM::GPU);

            cuda_gpu_rgb = cuda::GpuMat(height, width, CV_8UC4, rgb_ptr, rgb_step);
            cv::cuda::cvtColor(cuda_gpu_rgb, cuda_gpu_gray, cv::COLOR_RGBA2GRAY, 0, stream_rgb);

            cuda_gpu_gray.download(cuda_gray, stream_rgb);

            stream_rgb.waitForCompletion();
            acq.mp4_zed_rgb.write(cuda_gray);
        }
        if (acq.video_conf->output_zed_depth){
            acq.zed.retrieveMeasure(zed_gpu_depth, MEASURE::DEPTH, MEM::GPU);
            auto depth_ptr = zed_gpu_depth.getPtr<sl::uchar1>(MEM::GPU);
            auto depth_step = zed_gpu_depth.getStepBytes(MEM::GPU);

            cuda_gpu_depth = cuda::GpuMat(height, width, CV_32FC1, depth_ptr, depth_step);
            cuda::min(cuda_gpu_depth, 8.0f, cuda_gpu_depth, stream_depth);
            cv::cuda::multiply(cuda_gpu_depth, 255.0f / 8.0f, cuda_gpu_depth, 1, CV_32FC1, stream_depth);
            cuda_gpu_depth.convertTo(cuda_gpu_depth, CV_8UC1, stream_depth);

            cuda_gpu_depth.download(cuda_depth, stream_depth);

            stream_depth.waitForCompletion();
            acq.mp4_zed_depth.write(cuda_depth);
        }
        if (acq.video_conf->output_zed_pc){
            acq.zed.retrieveMeasure(zed_gpu_point_cloud, MEASURE::XYZ, MEM::GPU);

            auto pc_ptr = zed_gpu_point_cloud.getPtr<sl::uchar1>(MEM::GPU);
            auto pc_step = zed_gpu_point_cloud.getStepBytes(MEM::GPU);

            cuda_gpu_point_cloud = cuda::GpuMat(height, width, CV_32FC3, pc_ptr, pc_step);
            cv::cuda::split(cuda_gpu_point_cloud, cudaXYZ);
            for (int i = 0; i < 3; i++) {
                auto cuda_axis = cudaXYZ[i];
                cuda::min(cuda_axis, 8.0f, cuda_axis, stream_pc[i]);
                cv::cuda::multiply(cuda_axis, 255.0f / 8.0f, cuda_axis, 1, CV_32FC1, stream_pc[i]);
                cuda_axis.convertTo(cuda_axis, CV_8UC1, stream_pc[i]);
                cuda_axis.download(cvXYZ[i], stream_pc[i]);
            }
            for (auto &spc : stream_pc) spc.waitForCompletion();
            acq.mp4_zed_X.write(cvXYZ[0]);
            acq.mp4_zed_Y.write(cvXYZ[1]);
            acq.mp4_zed_Z.write(cvXYZ[2]);
        }
    }

    if (acq.debug)
        cout << "ZedThread end" << endl;
}

void GrabThread(int stream_index, AcquisitionParameters &acq) {
    StreamInfo *MyStreamInfo = acq.MyStreamInfos[stream_index];
    auto *lStream = (PvStream *) (MyStreamInfo->aStream);
    PvResult lResult, lOperationResult;
    uint64_t CurrentBlockID = 0, PrevBlockID = 0;
    int height, width;

    pthread_mutex_lock(&acq.acq_start_mtx);
    pthread_cond_wait(&acq.GrabEvent, &acq.acq_start_mtx);
    pthread_mutex_unlock(&acq.acq_start_mtx);
    if (acq.debug)
        cout << "JAI STREAM " << stream_index << " STARTED" << endl;
    short cv_bit_depth = CV_8U;
    while (acq.is_running) {
        PvBuffer *lBuffer = nullptr;
        auto *curr_frame = new SingleJAIChannel;
        lResult = lStream->RetrieveBuffer(&lBuffer, &lOperationResult, 1000);
        curr_frame->timestamp = get_current_time();
        if (lResult.IsOK()) {
            if (lOperationResult.IsOK()) {
                // We now have a valid buffer. This is where you would typically process the buffer.

                CurrentBlockID = lBuffer->GetBlockID();
                if (CurrentBlockID != PrevBlockID + 1 and PrevBlockID != 0) {
                    acq.frame_drop_log_file << "JAI STREAM " << stream_index << " - FRAME DROP - FRAME No. " << PrevBlockID << endl;
                }
                PrevBlockID = CurrentBlockID;
                height = lBuffer->GetImage()->GetHeight(), width = lBuffer->GetImage()->GetWidth();
                lStream->QueueBuffer(lBuffer);
                curr_frame->frame = cv::Mat(height, width, cv_bit_depth, lBuffer->GetImage()->GetDataPointer());
                curr_frame->BlockID = CurrentBlockID;
                pthread_mutex_lock(&acq.grab_mtx);
                string s = "JAI STREAM" + to_string(stream_index) + " - PUSH FRAME " + to_string(CurrentBlockID);
                acq.jai_acquisition_log << s << endl;
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
            acq.is_running = false;
            acq.jai_connected = false;
            if (acq.debug) {
                cout << stream_index << ": BAD RESULT!" << endl;
                // Retrieve buffer failure
                cout << lResult.GetCodeString().GetAscii() << "\n";
            }
        }
    }
    if (acq.debug)
        cout << stream_index << ": Acquisition end with " << CurrentBlockID << endl;
}

void stretch(cuda::GpuMat& channel, cuda::Stream &stream, double lower = 0.005, double upper = 0.995, double min_int = 25, double max_int = 235) {
    const int hist_size = 256;
    int lower_threshold = 0, upper_threshold = 255;
    double percentile = 0.0;

    cuda::GpuMat gpu_hist;
    cv::Mat cpu_hist;

    cuda::calcHist(channel, gpu_hist, stream);
    gpu_hist.download(cpu_hist, stream);
    cv::Scalar total = cuda::sum(gpu_hist);

    // Calculate lower and upper threshold indices for gain
    
    stream.waitForCompletion();

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

    double gain = (max_int - min_int) / (upper_threshold - lower_threshold);
    double offset = min_int;

    cuda::multiply(channel, gain, channel, 1, -1, stream);
    cuda::add(channel, offset, channel, noArray(), -1, stream);

    cv::Scalar lower_scalar(0);
    cv::Scalar upper_scalar(255);

    // Clipping to range [0, 255]
    cv::cuda::max(lower_scalar, channel, channel, stream);
    cv::cuda::min(upper_scalar, channel, channel, stream);
}

void stretch_and_clahe(cuda::GpuMat& channel, cuda::Stream &stream, const Ptr<cuda::CLAHE>& clahe, double lower = 0.005, double upper = 0.995, double min_int = 25,
                       double max_int = 235) {
    stretch(channel, stream, lower, upper, min_int, max_int);
    clahe->apply(channel, channel, stream);
}

void fsi_from_channels(const cv::Ptr<cuda::CLAHE>& clahe, cuda::GpuMat& blue, cuda::Stream &stream_blue,
                       cuda::GpuMat& c_800, cuda::Stream &stream_800, cuda::GpuMat& c_975, cuda::Stream &stream_975,
                       cuda::GpuMat &fsi, cuda::Stream &stream_fsi, double lower = 0.005, double upper = 0.995,
                       double min_int = 25, double max_int = 235) {
    
    stretch_and_clahe(blue, stream_blue, clahe, lower, upper, min_int, max_int);
    stretch_and_clahe(c_800, stream_800, clahe, lower, upper, min_int, max_int);
    stretch_and_clahe(c_975, stream_975, clahe, lower, upper, min_int, max_int);

    std::vector<cv::cuda::GpuMat> channels;
    
    stream_blue.waitForCompletion();
    channels.push_back(blue);
    
    stream_800.waitForCompletion();
    channels.push_back(c_800);
    
    stream_975.waitForCompletion();
    channels.push_back(c_975);

    cuda::merge(channels, fsi, stream_fsi);
}

void MergeThread(AcquisitionParameters &acq) {

    cv::Mat Frames[3], res;
    cuda::GpuMat cudaFSI_clahe, cudaFSI_equalized_hist;
    std::vector<cuda::GpuMat> cudaBGR(3), cudaFrames(3), cudaFrames_equalized(3);
    cuda::Stream streams[3], stream_fsi_equalize_hist, stream_fsi_clahe;
    int frame_count = 0;
    struct timespec max_wait = {0, 0};
    SingleJAIChannel *e_frames[3] = {new SingleJAIChannel, new SingleJAIChannel, new SingleJAIChannel};
    bool grabbed[3] = { false, false, false };

    if (acq.debug)
        cout << "MERGE THREAD START" << endl;
    sleep(1);
    pthread_cond_broadcast(&acq.GrabEvent);
    cv::Ptr<cuda::CLAHE> clahe = cv::cuda::createCLAHE(2, cv::Size(10, 10));
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
                string s = "MERGE POP FRAME " + to_string(e_frames[i]->BlockID) + " FROM STREAM " + to_string(i);
                acq.jai_acquisition_log << s << endl;
            }
            grabbed[i] = false;
            pthread_mutex_unlock(&acq.grab_mtx);
        }
        if (!acq.is_running) {
            break;
        }
        if (e_frames[0]->BlockID != e_frames[1]->BlockID or e_frames[0]->BlockID != e_frames[2]->BlockID) {
            int max_id = std::max({e_frames[0]->BlockID, e_frames[1]->BlockID, e_frames[2]->BlockID});
            acq.frame_drop_log_file << "MERGE DROP - AFTER FRAME NO. " << --frame_no << endl;
            for (int i = 0; i < 3; i++) {
                if (e_frames[i]->BlockID == max_id)
                    grabbed[i] = true;
            }
            continue;
        }

        cv::Mat res_clahe_fsi, res_equalize_hist_fsi;
        // the actual bayer format we use is RGGB (or - BayerRG) but OpenCV refers to it as BayerBG
        // for more info look at - https://github.com/opencv/opencv/issues/19629

        cudaFrames[0].upload(Frames[0], streams[0]); // channel 0 = BayerBG8
        cudaFrames[2].upload(Frames[1], streams[2]); // channel 1 = 800nm -> Red
        cudaFrames[1].upload(Frames[2], streams[1]); // channel 2 = 975nm -> Green

        cv::cuda::demosaicing(cudaFrames[0], cudaFrames[0], cv::COLOR_BayerBG2BGR, -1, streams[0]);
        if (acq.video_conf->output_rgb)
            cudaFrames[0].download(Frames[0], streams[0]);
        cv::cuda::split(cudaFrames[0], cudaBGR, streams[0]);
        cudaFrames[0] = cudaBGR[2]; // just pick the blue from the bayer
        for (int i = 0; i < 3; i++) {
            cv::cuda::normalize(cudaFrames[i], cudaFrames[i], 0, 255, cv::NORM_MINMAX, CV_8U, noArray(), streams[i]);
        }
        if (acq.video_conf->output_800)
            cudaFrames[2].download(Frames[1], streams[2]);
        if (acq.video_conf->output_975)
            cudaFrames[1].download(Frames[2], streams[1]);

        if (acq.video_conf->output_equalize_hist_fsi or (not acq.video_conf->pass_clahe_stream)) {
            for (int i = 0; i < 3; i++) {
                cv::cuda::equalizeHist(cudaFrames[i], cudaFrames_equalized[i], streams[i]);
            }
            
            cv::cuda::merge(cudaFrames_equalized, cudaFSI_equalized_hist, stream_fsi_equalize_hist);
            cudaFSI_equalized_hist.download(res_equalize_hist_fsi, stream_fsi_equalize_hist);
        }
        if (acq.video_conf->output_clahe_fsi or acq.video_conf->pass_clahe_stream) {
            fsi_from_channels(clahe, cudaFrames[0], streams[0], cudaFrames[1], streams[1], cudaFrames[2], streams[2], cudaFSI_clahe, stream_fsi_clahe);
            cudaFSI_clahe.download(res_clahe_fsi, stream_fsi_clahe);
        }

        string timestamp = e_frames[0]->timestamp;
        int BlockID = e_frames[0]->BlockID;
        string s = "MERGE SYNC FRAME " + to_string(e_frames[0]->BlockID);
        acq.jai_acquisition_log << s << endl;

        EnumeratedJAIFrame e_frame_fsi;
        if (acq.video_conf->transfer_data) {
            if (acq.video_conf->pass_clahe_stream) {
                stream_fsi_clahe.waitForCompletion();
                e_frame_fsi = {timestamp, res_clahe_fsi, Frames[0], BlockID};
            } else {
                stream_fsi_equalize_hist.waitForCompletion();
                e_frame_fsi = {timestamp, res_equalize_hist_fsi, Frames[0], BlockID};
            }
        }
        acq.jz_streamer.push_jai(e_frame_fsi);

        frame_count++;
        if (frame_count % (acq.video_conf->FPS * 30) == 0 and acq.debug)
            cout << endl << frame_count / (acq.video_conf->FPS * 60.0) << " minutes of video written" << endl << endl;

        if (acq.video_conf->output_clahe_fsi) {
            stream_fsi_clahe.waitForCompletion();
            acq.mp4_clahe_FSI.write(res_clahe_fsi);
        }
        if (acq.video_conf->output_equalize_hist_fsi) {
            stream_fsi_equalize_hist.waitForCompletion();
            acq.mp4_equalize_hist_FSI.write(res_equalize_hist_fsi);
        }
        if (acq.video_conf->output_rgb) {
            streams[0].waitForCompletion();
            acq.mp4_BGR.write(Frames[0]);
        }
        if (acq.video_conf->output_800) {
            streams[2].waitForCompletion();
            acq.mp4_800.write(Frames[1]);
        }
        if (acq.video_conf->output_975) {
            streams[1].waitForCompletion();
            acq.mp4_975.write(Frames[2]);
        }
    }

    if (acq.debug)
        cout << "MergeThread end" << endl;

}

string get_current_time() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() % 1000000;
    auto current_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss_now;
    ss_now << std::put_time(std::localtime(&current_time), "%Y-%m-%d %H:%M:%S.") << std::setfill('0')
    << std::setw(6) << microseconds;
    return ss_now.str();
}

JaiZedStatus connect_cameras(AcquisitionParameters &acq, int fps){

    pthread_cond_init(&acq.GrabEvent, nullptr);
    bool jai_connected = false, zed_connected = false;

    while (true) {
        if (not jai_connected)
            jai_connected = setup_JAI(acq);
        if (not zed_connected)
            zed_connected = connect_ZED(acq, fps);
        if (jai_connected and zed_connected)
            break;
        if (acq.debug) {
            cout << (jai_connected ? "JAI CONNECTED" : "JAI NOT CONNECTED") << endl;
            cout << (zed_connected ? "ZED CONNECTED" : "ZED NOT CONNECTED") << endl;
            cout << "Retrying..." << endl;
        }
        sleep(10);
    }
    JaiZedStatus jzs = {jai_connected, zed_connected};

    if (acq.debug) {
        cout << (jzs.jai_connected ? "JAI CONNECTED" : "JAI NOT CONNECTED") << endl;
        cout << (jzs.zed_connected ? "ZED CONNECTED" : "ZED NOT CONNECTED") << endl;
    }

    acq.jai_connected = jzs.jai_connected;
    acq.zed_connected = jzs.zed_connected;
    return jzs;
}

bool start_acquisition(AcquisitionParameters &acq) {
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStart;

    if (acq.jai_connected and acq.zed_connected) {
        set_acquisition_parameters(acq);
        for (int i = 0; i < 3; i++) {
            CreateStreamBuffers(acq.lDevice, acq.MyStreamInfos[i]->aStream, &acq.lBufferLists[i]);
        }
        acq.lDevice->StreamEnable();

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
//        pthread_cond_wait(&acq.GrabEvent, &acq.acq_start_mtx);
    }
    else if (acq.debug)
        cout << "NOT CONNECTED" << endl;
    return acq.is_running;
}

void stop_acquisition(AcquisitionParameters &acq) {
    PvGenParameterArray *lDeviceParams;
    PvGenCommand *lStop;

    if (acq.debug)
        cout << "ACQUISITION STOP RECEIVED" << endl;

    // Get device parameters need to control streaming - set acquisition stop command
    lDeviceParams = acq.lDevice->GetParameters();
    lStop = dynamic_cast<PvGenCommand *>(lDeviceParams->Get("AcquisitionStop"));
    acq.is_running = false;


    acq.jai_t0.join();
    acq.jai_t1.join();
    acq.jai_t2.join();

    lStop->Execute();

    acq.zed_t.join();
    acq.merge_t.join();

    for (auto & MyStreamInfo : acq.MyStreamInfos) {
        MyStreamInfo->aStream->AbortQueuedBuffers();

        // Empty the remaining frames that from the buffers
        while (MyStreamInfo->aStream->GetQueuedBufferCount() > 0) {
            PvBuffer *lBuffer = nullptr;
            PvResult lOperationResult;
            MyStreamInfo->aStream->RetrieveBuffer(&lBuffer, &lOperationResult);
        }
        while (not MyStreamInfo->Frames.empty()) {
            MyStreamInfo->Frames.pop();
        }
    }

    acq.jai_acquisition_log.close();
    acq.frame_drop_log_file.close();
    acq.imu_log_file.close();

    if (acq.video_conf->output_clahe_fsi)
        acq.mp4_clahe_FSI.release();
    if (acq.video_conf->output_equalize_hist_fsi)
        acq.mp4_equalize_hist_FSI.release();
    if (acq.video_conf->output_rgb)
        acq.mp4_BGR.release();
    if (acq.video_conf->output_800)
        acq.mp4_800.release();
    if (acq.video_conf->output_975)
        acq.mp4_975.release();
    if (acq.video_conf->output_svo)
        acq.zed.disableRecording();
    if (acq.video_conf->output_zed_gray) {
        acq.mp4_zed_rgb.release();
    }
    if (acq.video_conf->output_zed_depth) {
        acq.mp4_zed_depth.release();
    }
    if (acq.video_conf->output_zed_pc) {
        acq.mp4_zed_X.release();
        acq.mp4_zed_Y.release();
        acq.mp4_zed_Z.release();
    }

    // Tell the device to stop sending images + disable streaming
    acq.lDevice->StreamDisable();

    if (acq.debug)
        cout << "ACQUISITION STOPPED SUCCESSFULLY" << endl;

}

void disconnect_zed(AcquisitionParameters &acq){
    acq.zed.close();
    acq.zed_connected = false;
}

void disconnect_jai(AcquisitionParameters &acq) {

    if (acq.debug)
        cout << "Aborting buffers still in streams" << endl << "closing streams" << endl;
    for (int i = 0; i < 3; i++) {
        acq.MyStreamInfos[i]->aStream->AbortQueuedBuffers();

        // Empty the remaining frames that from the buffers
        while (acq.MyStreamInfos[i]->aStream->GetQueuedBufferCount() > 0) {
            PvBuffer *lBuffer = nullptr;
            PvResult lOperationResult;
            acq.MyStreamInfos[i]->aStream->RetrieveBuffer(&lBuffer, &lOperationResult);
        }
        FreeStreamBuffers(&acq.lBufferLists[i]);
        acq.MyStreamInfos[i]->aStream->Close();
        PvStream::Free(acq.MyStreamInfos[i]->aStream);
    }
    acq.lDevice->StreamDisable();
    // Disconnect the device
    cout << acq.jai_connected << endl;
    if (acq.jai_connected) {
        acq.lDevice->Disconnect();
        PvDevice::Free(acq.lDevice);
    }

    pthread_cond_destroy(&acq.GrabEvent);
    for (int i = 0; i < 3; i++)
        pthread_cond_destroy(&acq.MergeFramesEvent[i]);

    acq.jai_connected = false;
}

void disconnect_cameras(AcquisitionParameters &acq){
    disconnect_zed(acq);
    disconnect_jai(acq);
    if (acq.debug)
        cout << "cameras disconnected" << endl;
}
