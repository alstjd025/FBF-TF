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
  interpreter_->CreateWorker(ResourceType::CPU, 1);
  scheduler_thread = std::thread(&LiteScheduler::SchedulerSpin, this);
  scheduler_thread.detach();
};

LiteScheduler::~LiteScheduler() {};

// new
void LiteScheduler::SchedulerSpin(){
  std::cout << "Scheduler: Scheduler spin!" << "\n";
  while(true){
    //wait for notify ()
    std::unique_lock<std::mutex> lock(scheduler_lock);
    scheduler_cv_.wait(lock, [&] { return need_reschedule && !stop_scheduler; });
    std::cout << "Scheduler: Woke up \n";
    if(interpreter_->IsJobEmpty()){
      std::cout << "Scheduler : scheduler woke up but no jobs in interpreter"
                << ". Stops scheduler" <<"\n";
      stop_scheduler = true;
      continue;
    }      
    if(need_reschedule){
      // profile unprofiled models
      // Some profiling logic here
    }
    // schedule jobs to workers and wake all                 
  };
}

} //namespace tflite