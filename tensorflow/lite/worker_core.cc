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

  Worker::Worker(WorkerType wType, int w_id, Interpreter* interpreter){
    type = wType;
    worker_id = w_id;
    interpreter_ = interpreter; 
  };

  bool Worker::JobTryLock(){
    if(mtx_lock.try_lock()){
      return true;
    }else{
      return false;
    }
  }

  void Worker::JobUnlock(){
    mtx_lock.unlock();
  }

  void Worker::Work(){
    
  }

  Worker::~Worker(){

  };



} //namespace tflite