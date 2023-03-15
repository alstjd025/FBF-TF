#include "tensorflow/lite/worker_core.h"

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define TFLITE_WORKER_MSG(x)

namespace tflite{
  Worker::Worker(){
    
  };

  Worker::Worker(workerType wType, int w_id){
    type = wType;
    worker_id = w_id;
    
  };

  Worker::~Worker(){

  };



} //namespace tflite