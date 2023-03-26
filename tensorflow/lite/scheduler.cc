#include "tensorflow/lite/scheduler.h"

namespace tflite{

//legacy
  Scheduler::Scheduler(){
    state = SchedulerState::INIT_SCHED;
    interpreter_ = nullptr;
  };

//legacy
  Scheduler::Scheduler(std::shared_ptr<tflite::Interpreter> interpreter){
    state = SchedulerState::INIT_SCHED;
    interpreter_ = interpreter;
    interpreter_->CreateWorker(ResourceType::CPU, 1);
    scheduler_thread = std::thread(&Scheduler::SchedulerSpin, this);
    scheduler_thread.detach();
  };

//legacy
  void Scheduler::SchedulerSpin(){
    std::cout << "Scheduler : Scheduler spin!" << "\n";
    while(true){
      std::unique_lock<std::mutex> lock(scheduler_lock);
      scheduler_cv_.wait(lock, [&] { return state != SchedulerState::STOP &&
                                      state != SchedulerState::INIT_SCHED; });
      std::cout << "Scheduler : woke up" << "\n";
      std::cout << "Scheduler : start scheduling" << "\n";
      if(interpreter_->IsJobEmpty()){
        std::cout << "Scheduler : scheduler woke up but no jobs in interpreter"
                  << ". wait for 100ms. " <<"\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        continue;
      }
      if(need_reschedule){ // if new job is allocated or SLO violated?
        if(CheckSchedulability()){ // Schedulable
          if(Reschedule() != kTfLiteOk){
            std::cout << "Rescheudling ERROR" << "\n";
            exit(-1);
          }
        }
        else{ // Not schedulable
          state = SchedulerState::NSCHEDULABLE;
        }
        std::cout << "sda " << "\n";
        need_reschedule = false;
      }
      std::cout << "invoke" << "\n";
      if(DoInvoke() != kTfLiteOk){
        std::cout << "Invoke returned Error on scheduler" << "\n";
        std::cout << "Stop scheduler" << "\n";
        state = SchedulerState::STOP;
      }
      state = SchedulerState::STOP;
    }
  };

//legacy
  Scheduler::~Scheduler(){
    std::cout << "Scheduler destructor called" << "\n";
  };

//legacy
  TfLiteStatus Scheduler::DoInvoke(){
    if(interpreter_->DoInvoke() != kTfLiteOk){
      std::cout << "Interpreter invoke returned ERROR" << "\n";
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

//legacy
  bool Scheduler::CheckSchedulability(){
    std::cout << "Scheduler : needs rescheduling.. checks scehdulability" << "\n";
    // NEEDS IMPL
    // interpreter->CheckSchedulability()
    /////
    return true;
  }

//legacy
  TfLiteStatus Scheduler::Reschedule(){
    if(interpreter_->GiveJob() != kTfLiteOk){
      std::cout << "GiveJob ERROR" << "\n";
      return kTfLiteError;
    }
    return kTfLiteOk;
  };

//legacy
  void Scheduler::NeedReschedule(bool flag){
    scheduler_lock.lock();
    need_reschedule = flag;
    scheduler_lock.unlock();
  };

//legacy
  void Scheduler::ChangeState(SchedulerState new_state){
    scheduler_lock.lock();
    state = new_state;
    scheduler_lock.unlock();
  };

//legacy
  void Scheduler::Join(){
    scheduler_thread.join();
  }

//legacy
  void Scheduler::notify(){
    std::unique_lock<std::mutex> lock(scheduler_lock);
    scheduler_cv_.notify_all();
  };

  // ModelFactory codes // //legacy
  ModelFactory::ModelFactory(){
    interpreter_ = nullptr;
  };

//legacy
  ModelFactory::ModelFactory(std::shared_ptr<tflite::Interpreter> interpreter){
    interpreter_ = interpreter;
  };

//legacy
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

//legacy
  void ModelFactory::ModelFactorySpin(){

  };

//legacy
  TfLiteStatus ModelFactory::GiveSubgraphtoInterperter(\
                                      std::shared_ptr<Scheduler> scheduler){

  };

//legacy
  ModelFactory::~ModelFactory(){

  };



} //namespace tflite