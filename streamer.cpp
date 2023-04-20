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
    pthread_mutex_lock(&jai_mutex);
    if (jai_frames.empty()) {
        pthread_cond_wait(&jai_cv, &jai_mutex);
    }
    EnumeratedJAIFrame frame = jai_frames.front();
    cout << "frame no. " << frame.BlockID << endl;
    jai_frames.pop();
    pthread_mutex_unlock(&jai_mutex);
    return frame;
}

EnumeratedZEDFrame JaiZedStream::pop_zed() {
    pthread_mutex_lock(&zed_mutex);
    if (zed_frames.empty()) {
        pthread_cond_wait(&zed_cv, &zed_mutex);
    }
    EnumeratedZEDFrame frame = zed_frames.front();
    zed_frames.pop();
    pthread_mutex_unlock(&zed_mutex);
    return frame;
}
