#include "tensorflow/lite/worker_api.h"

namespace tflite{

  WorkerAPI::WorkerAPI(){
    CreateWorker(workerType::NONE);
  };

  WorkerAPI::WorkerAPI(const char* model_name){

    // Creates a new interpreterBuilder for worker

    // WorkerAPI has at least one CPU worker
    CreateWorker(workerType::CPU);
  };



  TfLiteStatus WorkerAPI::CreateWorker(workerType wType){
    int new_id;
    new_id = worker_ids.size();
    Worker* new_worker = new Worker(wType, new_id);
    worker_ids.push_back(new_id);
    workers.push_back(new_worker);
  };


  TfLiteStatus WorkerAPI::AllocateTask(){

  };

  WorkerAPI::~WorkerAPI(){

  };




} //namespace tflite