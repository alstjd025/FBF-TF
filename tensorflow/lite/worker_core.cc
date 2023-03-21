#include "tensorflow/lite/worker_core.h"
#include "tensorflow/lite/interpreter.h"

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

namespace tflite{
  Worker::Worker(){
    
  };

  Worker::Worker(ResourceType wType, int w_id, Interpreter* interpreter){
    type = wType;
    worker_id = w_id;
    state = WorkerState::INIT_WORK;
    interpreter_ = interpreter; 
  };

  void Worker::GiveJob(tflite::Job* new_job){
    jobs.push_back(new_job);
  }

  bool Worker::TryLock(){
    if(mtx_lock.try_lock()){
      return true;
    }else{
      return false;
    }
  }

  void Worker::Unlock(){
    mtx_lock.unlock();
  }

  void Worker::WakeWorker(){
    worker_cv.notify_one();
  }

  void Worker::Work(){
    std::cout << "Worker [" << worker_id << "] started" << "\n";
    while(stop_working){
      std::unique_lock<std::mutex> lock(mtx_lock);
      worker_cv.wait(lock, [&] { return state == WorkerState::WORKING; });
      
    }
  }

  Worker::~Worker(){

  };



} //namespace tflite