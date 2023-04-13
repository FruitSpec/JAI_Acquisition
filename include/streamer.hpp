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
using namespace std;

struct EnumeratedJAIFrame {
    cv::Mat frame;
    int BlockID;
};

struct EnumeratedZEDFrame {
    sl::Mat frame;
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
    pthread_mutex_t m_mutex{};
    pthread_cond_t m_cv{};
    std::queue<EnumeratedZEDFrame> zed_frames;
    std::queue<EnumeratedJAIFrame> jai_frames;
};
