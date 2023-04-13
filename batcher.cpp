#include "batcher.hpp"

using namespace cv;
using namespace std;

BatchQueue::BatchQueue() {
    pthread_mutex_init(&m_mutex, nullptr);
    pthread_cond_init(&m_cv, nullptr);
}

BatchQueue::~BatchQueue() {
    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy(&m_cv);
}

void BatchQueue::push_jai(const std::array<cv::Mat, BATCH_SIZE>& jai_frames, short batch_number) {
    pthread_mutex_lock(&m_mutex);
    batch_number = static_cast<short>(batch_number % BUFFER_SIZE);
    m_buffer[batch_number].jai_frames = jai_frames;
    m_buffer[batch_number].jai_set = true;
    pthread_cond_signal(&m_cv);
    pthread_mutex_unlock(&m_mutex);
}

void BatchQueue::push_zed(const std::array<sl::Mat, BATCH_SIZE>& zed_frames, short batch_number) {
    pthread_mutex_lock(&m_mutex);
    batch_number = static_cast<short>(batch_number % BUFFER_SIZE);
    m_buffer[batch_number].zed_frames = zed_frames;
    m_buffer[batch_number].zed_set = true;
    pthread_cond_signal(&m_cv);
    pthread_mutex_unlock(&m_mutex);
}

FramesBatch BatchQueue::pop() {
    pthread_mutex_lock(&m_mutex);
    while (not (m_buffer[pointer].jai_set and m_buffer[pointer].zed_set)) {
        pthread_cond_wait(&m_cv, &m_mutex);
    }
    FramesBatch fb = m_buffer[pointer];
    m_buffer[pointer].jai_set = false;
    m_buffer[pointer].zed_set = false;
    pointer = static_cast<short>((pointer + 1) % BUFFER_SIZE);
    pthread_mutex_unlock(&m_mutex);
    return fb;
}
