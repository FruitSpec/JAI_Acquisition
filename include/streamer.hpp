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

using namespace cv;
using namespace std;

struct SingleJAIChannel {
    string timestamp;
    cv::Mat frame;
    int BlockID;
};

struct EnumeratedJAIFrame {
    string timestamp;
    cv::Mat fsi_frame, rgb_frame;
    int BlockID;
};

struct EnumeratedZEDFrame {
    string timestamp;
    sl::Mat point_cloud, rgb;
    sl::SensorsData::IMUData imu;
    int BlockID;
};

class JaiZedStream {
public:
    JaiZedStream();

    ~JaiZedStream();

    void push_jai(EnumeratedJAIFrame& jai_frame);

    void push_zed(EnumeratedZEDFrame& zed_frame);

    EnumeratedJAIFrame pop_jai();

    EnumeratedZEDFrame pop_zed();

private:
    pthread_mutex_t jai_mutex{}, zed_mutex{};
    pthread_cond_t jai_cv{}, zed_cv{};
    std::queue<EnumeratedZEDFrame> zed_frames;
    std::queue<EnumeratedJAIFrame> jai_frames;
};
