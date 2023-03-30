#include "tensorflow/lite/lite_scheduler.h"
#include "tensorflow/lite/interpreter.h"

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
  interpreter_ = interpreter;
  scheduler_thread = std::thread(&LiteScheduler::SchedulerSpin, this);
  scheduler_thread.detach();
};

LiteScheduler::~LiteScheduler() {};


void LiteScheduler::Wake(){
  std::unique_lock<std::mutex> lock(scheduler_lock);
  scheduler_cv_.notify_all();
}

// new
void LiteScheduler::SchedulerSpin(){
  std::cout << "Scheduler: Scheduler spin!" << "\n";
  while(true){
    //wait for notify ()
    std::unique_lock<std::mutex> lock(scheduler_lock);
    scheduler_cv_.wait(lock, [&] { return !stop_scheduler; });
    if(interpreter_->IsJobEmpty()){
      std::cout << "Scheduler : scheduler woke up but no jobs in interpreter"
                << ". Stops scheduler" <<"\n";
      stop_scheduler = true;
      continue;
    }      
    if(need_reschedule){
      // profile unprofiled models
      // Some profiling logic here
      // Create workers if needed
      // schedule jobs to workers and wake all  
      std::cout << "Scheduler: Creates worker" << "\n";
      interpreter_->CreateWorker(ResourceType::CPU, 1);
      interpreter_->CreateWorker(ResourceType::CPU, 2);
      interpreter_->GiveJob();               
    }
    // schedule jobs
    
  };
}

} //namespace tflite