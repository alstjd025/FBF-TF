#include "tensorflow/lite/workframe.h"

namespace tflite{

  WorkFrame::WorkFrame(){
  };

  WorkFrame::WorkFrame(const char* model_name, workerType wType){

    // Creates a new scheduler
    CreateProfilerandScheduler(model_name);

    // WorkFrame has at least one CPU worker
    if(workers.size() < 1){
      CreateWorker(wType, 1);
    }

  };


  TfLiteStatus WorkFrame::CreateProfilerandScheduler(const char* model){
    profiler_ = Profiler();
  };



  TfLiteStatus WorkFrame::CreateWorker(workerType wType, int cpu_num){
    // Creates a worker of given workerType.
    // If given worker type is NONE, will create a default worker.
    // A default worker only uses single CPU.
    if(wType == workerType::NONE){
      int new_id;
      new_id = worker_ids.size();
      Worker* new_worker = new Worker(wType, new_id);
      worker_ids.push_back(new_id);
      workers.push_back(new_worker);
    }else if(wType == workerType::CO_EX){ 

    }else if(wType == workerType::GPU){

    }else if(wType == workerType::CPU){
      if(cpu_num < 1){ // TODO : check available cpu numbers here
      
      }
    }
  };

  TfLiteStatus WorkFrame::CreateAndGiveJob(const char* model){

  };

  TfLiteStatus WorkFrame::AllocateTask(){

  };

  TfLiteStatus WorkFrame::TestInvoke(){

  };

  WorkFrame::~WorkFrame(){
    
  };




} //namespace tflite