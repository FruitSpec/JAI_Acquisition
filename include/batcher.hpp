#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudacodec.hpp>
#include <sl/Camera.hpp>

#define BATCH_SIZE 4
#define BUFFER_SIZE 15


using namespace cv;

struct FramesBatch {
    bool jai_set = false, zed_set = false;
    std::array<cv::Mat, BATCH_SIZE> jai_frames;
    std::array<sl::Mat, BATCH_SIZE> zed_frames;
};

class BatchQueue {
public:
    BatchQueue();

    ~BatchQueue();

    void push_jai(const std::array<cv::Mat, BATCH_SIZE>& jai_frames, short frame_number);

    void push_zed(const std::array<sl::Mat, BATCH_SIZE>& zed_frames, short frame_number);

    FramesBatch pop();

private:
    short pointer = 0;
    pthread_mutex_t m_mutex{};
    pthread_cond_t m_cv{};
    std::array<FramesBatch, BUFFER_SIZE> m_buffer;
};
