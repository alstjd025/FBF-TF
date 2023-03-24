#include "tensorflow/lite/scheduler.h"

namespace tflite{

  Scheduler::Scheduler(){
    state = SchedulerState::INIT_SCHED;
    interpreter_ = nullptr;
  };

  Scheduler::Scheduler(std::shared_ptr<tflite::Interpreter> interpreter){
    state = SchedulerState::INIT_SCHED;
    interpreter_ = interpreter;
    interpreter_->CreateWorker(ResourceType::CPU, 1);
    scheduler_thread = std::thread(&Scheduler::SchedulerSpin, this);
    scheduler_thread.detach();
  };

  void Scheduler::SchedulerSpin(){
    std::cout << "Scheduler : Scheduler spin!" << "\n";
    while(true){
      std::unique_lock<std::mutex> lock(scheduler_lock);
      std::cout << "scheduler_stop " << scheduler_stop << "\n";
      scheduler_cv_.wait(lock, [&] { return state != SchedulerState::STOP; });
      std::cout << "Scheduler : woke up" << "\n";
      interpreter_->LockJobs();
      std::cout << "Scheduler : start scheduling" << "\n";

      if(interpreter_->IsJobEmpty()){
        std::cout << "Scheduler : scheduler woke up but no jobs in interpreter"
                  << ". wait for 100ms. " <<"\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      if(need_reschedule){ // if new job is allocated or SLO violated?
        if(CheckSchedulability()){ // Schedulable
          state = SchedulerState::SCHEDULABLE;
          if(Reschedule() != kTfLiteOk){
            std::cout << "Rescheudling ERROR" << "\n";
            exit(-1);
          }
        }
        else{ // Not schedulable
          state = SchedulerState::NSCHEDULABLE;
        }
        need_reschedule = false;
        interpreter_->UnlockJobs();
      }
      if(DoInvoke() != kTfLiteOk){
        std::cout << "Invoke returned Error on scheduler" << "\n";
        std::cout << "Stop scheduler" << "\n";
        state = SchedulerState::STOP;
      }
    }
  };

  Scheduler::~Scheduler(){
      
  };

  TfLiteStatus Scheduler::DoInvoke(){
    if(interpreter_->DoInvoke() != kTfLiteOk){
      std::cout << "Interpreter invoke returned ERROR" << "\n";
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

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

  void Scheduler::NeedReschedule(bool flag){
    need_reschedule = flag;
  };

  void Scheduler::ChangeState(SchedulerState new_state){
    state = new_state;
  };

  void Scheduler::Join(){
    scheduler_thread.join();
  }

  void Scheduler::notify(){
    std::unique_lock<std::mutex> lock(scheduler_lock);
    scheduler_cv_.notify_all();
  };

  // ModelFactory codes //
  ModelFactory::ModelFactory(){
    interpreter_ = nullptr;
  };

  ModelFactory::ModelFactory(std::shared_ptr<tflite::Interpreter> interpreter){
    interpreter_ = interpreter;
  };

  TfLiteStatus ModelFactory::CreateProfileModel(const char* model_name){
    std::unique_ptr<tflite::FlatBufferModel>* model;
    
    model = new std::unique_ptr<tflite::FlatBufferModel>\
    (tflite::FlatBufferModel::BuildFromFile(model_name));

    // Build the interpreter with the InterpreterBuilder.
    tflite::ops::builtin::BuiltinOpResolver* resolver;
    resolver = new tflite::ops::builtin::BuiltinOpResolver;
    
    // Experimental API
    // Giving a model number from recieved model size?
    int model_number = builders_created;
    builders_created++;

    std::cout << "Factory: Currently have " << model_number << " on runtime" << "\n";


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
    return kTfLiteOk;
  };

  void ModelFactory::ModelFactorySpin(){

  };

  TfLiteStatus ModelFactory::GiveSubgraphtoInterperter(\
                                      std::shared_ptr<Scheduler> scheduler){

  };

  ModelFactory::~ModelFactory(){

  };



} //namespace tflite