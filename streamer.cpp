#include "streamer.hpp"


JaiZedStream::JaiZedStream() {
    pthread_mutex_init(&m_mutex, nullptr);
    pthread_cond_init(&m_cv, nullptr);
}

JaiZedStream::~JaiZedStream() {
    pthread_mutex_destroy(&m_mutex);
    pthread_cond_destroy(&m_cv);
}

void JaiZedStream::push_jai(EnumeratedJAIFrame &jai_frame) {
    pthread_mutex_lock(&m_mutex);
    jai_frames.push(jai_frame);
    pthread_cond_signal(&m_cv);
    pthread_mutex_unlock(&m_mutex);
}

void JaiZedStream::push_zed(EnumeratedZEDFrame &zed_frame) {
    pthread_mutex_lock(&m_mutex);
    zed_frames.push(zed_frame);
    pthread_cond_signal(&m_cv);
    pthread_mutex_unlock(&m_mutex);
}

EnumeratedJAIFrame JaiZedStream::pop_jai() {
    pthread_mutex_lock(&m_mutex);
    if (jai_frames.empty()) {
        cout << "C++" << endl;
        pthread_cond_wait(&m_cv, &m_mutex);
    }
    EnumeratedJAIFrame frame = jai_frames.front();
    jai_frames.pop();
    pthread_mutex_unlock(&m_mutex);
    return frame;
}

EnumeratedZEDFrame JaiZedStream::pop_zed() {
    pthread_mutex_lock(&m_mutex);
    if (zed_frames.empty()) {
        cout << "C++" << endl;
        pthread_cond_wait(&m_cv, &m_mutex);
    }
    EnumeratedZEDFrame frame = zed_frames.front();
    zed_frames.pop();
    pthread_mutex_unlock(&m_mutex);
    return frame;
}
