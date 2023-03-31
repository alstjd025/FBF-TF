#include "tensorflow/lite/lite_scheduler.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h" 
#include "tensorflow/lite/core/subgraph.h" 

namespace tflite{
// new
LiteScheduler::LiteScheduler(){
  state = SchedulerState::INIT_SCHED;
  scheduler_thread = std::thread(&LiteScheduler::SchedulerSpin, this);
  scheduler_thread.detach();
};


//legacy
LiteScheduler::LiteScheduler(Interpreter* interpreter){
  state = SchedulerState::INIT_SCHED;
  need_reschedule = true;
  interpreter_ = interpreter;
  scheduler_thread = std::thread(&LiteScheduler::SchedulerSpin, this);
  scheduler_thread.detach();
};

LiteScheduler::~LiteScheduler() {};


void LiteScheduler::Wake(){
  std::unique_lock<std::mutex> lock(scheduler_lock);
  scheduler_cv_.notify_all();
}

void LiteScheduler::Join(){
  scheduler_thread.join();
}

void LiteScheduler::NeedReschedule(){
  std::unique_lock<std::mutex> lock(scheduler_lock);
  need_reschedule = true;
}


TfLiteStatus LiteScheduler::RegisterInterpreterBuilder(InterpreterBuilder* builder){
  for(auto builder_ : builders){
    if(builder_->GetModelid() == builder->GetModelid()){
      std::cout << "Model id [" << builder->GetModelid() << "] already exists" << "\n";
      return kTfLiteError;
    }
  }
  builders.push_back(builder);
  return kTfLiteOk;
}

// new
void LiteScheduler::SchedulerSpin(){
  std::cout << "Scheduler: Scheduler spin!" << "\n";
  while(true){
    //wait for notify ()
    std::unique_lock<std::mutex> lock(scheduler_lock);
    scheduler_cv_.wait(lock, [&] { return !stop_scheduler; });    
    if(need_reschedule){
      if(interpreter_->IsJobEmpty()){
        std::cout << "Scheduler : scheduler woke up but no jobs in interpreter"
                  << ". Stops scheduler" <<"\n";
        stop_scheduler = true;
        continue;
      }  
      // profile unprofiled models
      // Some profiling logic here
      // Create workers if needed
      // schedule jobs to workers and wake all  
      Profile();

      std::cout << "Scheduler: Creates worker" << "\n";
      interpreter_->CreateWorker(ResourceType::CPU, 1);
      //interpreter_->CreateWorker(ResourceType::CPU, 2);
      interpreter_->GiveJob();          
    }
    // schedule jobs with scheduling algorithm.
  };
}

void LiteScheduler::Profile(){
  // Debugging code 
  for(auto builder : builders){
    if(builder->GetModelid() == 0){
      Subgraph* original_graph_profiled = 
            interpreter_->returnProfiledOriginalSubgraph(0);
      if(original_graph_profiled == nullptr){
        std::cout << "Model id " << builder->GetModelid() << " no subgraph. \n"; 
        continue;
      }
      if(builder->CreateSubgraphsFromProfiling(original_graph_profiled, 
                    std::make_shared<tflite::Interpreter>(interpreter_))
          != kTfLiteOk){
            std::cout << "CreateSubgraphsFromProfiling returned ERROR" << "\n";
      }
    }
  }
}

TfLiteStatus LiteScheduler::RebuildSubgraphsAndJobs(){

};

} //namespace tflite