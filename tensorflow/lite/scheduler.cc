#include "tensorflow/lite/scheduler.h"

namespace tflite{

  Scheduler::Scheduler(){
    state = SchedulerState::INIT_SCHED;
    interpreter_ = nullptr;
  };

  Scheduler::Scheduler(std::shared_ptr<tflite::Interpreter> interpreter){
    state = SchedulerState::INIT_SCHED;
    interpreter_ = interpreter;
    interpreter_->CreateWorker(workerType::CPU_WORKER, 1);
    scheduler_thread = std::thread(&Scheduler::SchedulerSpin, this);
  };

  void Scheduler::SchedulerSpin(){
    std::cout << "Scheduler started" << "\n";
    while(state != SchedulerState::STOP){
      std::unique_lock<std::mutex> lock(scheduler_lock);
      interpreter_->LockJobs();
      scheduler_cv_.wait(lock, [&] { return interpreter_->IsJobEmpty(); });
      std::cout << "Schedule started" << "\n";
      if(interpreter_->IsJobEmpty()){
        lock.unlock();
      }else{
        
        if(CheckSchedulability()){ // Schedulable
          state = SchedulerState::SCHEDULABLE;
          if(Reschedule() != kTfLiteOk){
            std::cout << "Rescheudling ERROR" << "\n";
            exit(-1);
          }
        }
        else{ // Not schedulable
          state = SchedulerState::BLOCKED;
        }

      }
      interpreter_->UnlockJobs();
    }
  };

  Scheduler::~Scheduler(){
      
  };


  bool Scheduler::CheckSchedulability(){
    // NEEDS IMPL
    // interpreter->CheckSchedulability()
    /////
    return true;
  }
  
  TfLiteStatus Scheduler::ReadyWorkers(){
    return kTfLiteOk;
  }

  TfLiteStatus Scheduler::Reschedule(){
    if(ReadyWorkers() != kTfLiteOk){
      std::cout << "ReadyWorkers EROOR" << "\n";
      exit(-1);
    }
    interpreter_->GiveJob();
    return kTfLiteOk;
  };

  void Scheduler::notify(){
    scheduler_cv_.notify_one();
  };

  // ModelFactory codes //
  ModelFactory::ModelFactory(){
    interpreter_ = nullptr;
  };

  ModelFactory::ModelFactory(std::shared_ptr<tflite::Interpreter> interpreter){
    interpreter_ = interpreter;
  };

  TfLiteStatus ModelFactory::CreateAndProfileModel(const char* model_name){
    std::unique_ptr<tflite::FlatBufferModel>* model;
    
    model = new std::unique_ptr<tflite::FlatBufferModel>\
    (tflite::FlatBufferModel::BuildFromFile(model_name));

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver* resolver;
    resolver = new tflite::ops::builtin::BuiltinOpResolver;
    
    // Experimental API
    // Giving a model number from recieved model size?
    int model_number = builder_and_id.size();
    #ifdef DEBUG
      std::cout << "Currently have " << model_number << " on runtime" << "\n";
    #endif

    tflite::InterpreterBuilder* new_builder = \
             new tflite::InterpreterBuilder(**model, *resolver, model_name, \
                                                        model_number);

    builder_and_id.insert({model_number, new_builder});
    
    // Now creates an invokable origin subgraph from new model.
    if(new_builder->CreateSubgraphFromFlatBuffer(interpreter_) != kTfLiteOk){
      std::cout << "CreateSubgraphFromFlatBuffer returned Error" << "\n";
      exit(-1);
    }
    // And schedule it for latency profiling
    
  };

  void ModelFactory::ModelFactorySpin(){

  };

  TfLiteStatus ModelFactory::GiveSubgraphtoInterperter(\
                                      std::shared_ptr<Scheduler> scheduler){

  };

  ModelFactory::~ModelFactory(){

  };



} //namespace tflite