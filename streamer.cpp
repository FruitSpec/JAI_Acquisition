#include "streamer.hpp"


JaiZedStream::JaiZedStream() {
    pthread_mutex_init(&jai_mutex, nullptr);
    pthread_cond_init(&jai_cv, nullptr);
    pthread_mutex_init(&zed_mutex, nullptr);
    pthread_cond_init(&zed_cv, nullptr);
}

JaiZedStream::~JaiZedStream() {
    pthread_mutex_destroy(&jai_mutex);
    pthread_cond_destroy(&jai_cv);
    pthread_mutex_destroy(&zed_mutex);
    pthread_cond_destroy(&zed_cv);
}

void JaiZedStream::push_jai(EnumeratedJAIFrame &jai_frame) {
    pthread_mutex_lock(&jai_mutex);
    jai_frames.push(jai_frame);
    pthread_cond_signal(&jai_cv);
    pthread_mutex_unlock(&jai_mutex);
}

void JaiZedStream::push_zed(EnumeratedZEDFrame &zed_frame) {
    pthread_mutex_lock(&zed_mutex);
    zed_frames.push(zed_frame);
    pthread_cond_signal(&zed_cv);
    pthread_mutex_unlock(&zed_mutex);
}

EnumeratedJAIFrame JaiZedStream::pop_jai() {
    struct timespec max_wait = {0, 0};
    pthread_mutex_lock(&jai_mutex);
    if (jai_frames.empty()) {
        clock_gettime(CLOCK_REALTIME, &max_wait);
        max_wait.tv_sec += 1;
        const int timed_wait_rv = pthread_cond_timedwait(&jai_cv, &jai_mutex, &max_wait);
    }
    if (!acq.is_runing) {
        for (int i = 0; i < 100; i++)
            cout << "ACQ IS NOT RUNNING BUT ASKED FOR JAI FRAME!!!" << endl;
    }
    EnumeratedJAIFrame frame = jai_frames.front();
    jai_frames.pop();
    pthread_mutex_unlock(&jai_mutex);
    return frame;
}

EnumeratedZEDFrame JaiZedStream::pop_zed() {
    struct timespec max_wait = {0, 0};
    pthread_mutex_lock(&zed_mutex);
    if (zed_frames.empty()) {
        clock_gettime(CLOCK_REALTIME, &max_wait);
        max_wait.tv_sec += 1;
        const int timed_wait_rv = pthread_cond_timedwait(&zed_cv, &zed_mutex, &max_wait);
    }
    if (!acq.is_runing) {
        for (int i = 0; i < 100; i++)
            cout << "ACQ IS NOT RUNNING BUT ASKED FOR ZED FRAME!!!" << endl;
    }
    EnumeratedZEDFrame frame = zed_frames.front();
    zed_frames.pop();
    pthread_mutex_unlock(&zed_mutex);
    return frame;
}
