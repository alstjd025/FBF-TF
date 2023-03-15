#include "tensorflow/lite/scheduler.h"


namespace tflite{
  Scheduler::Scheduler(){

  };

  Scheduler::~Scheduler(){

  };


  Profiler::Profiler(){

  };

  TfLiteStatus Profiler::CreateAndProfileModel(const char* model, \
                                              std::shared_ptr<Scheduler> scheduler){

  }

  TfLiteStatus Profiler::GiveJobtoInterperter(std::shared_ptr<Scheduler> scheduler){

  }

  Profiler::~Profiler(){

  };



} //namespace tflite