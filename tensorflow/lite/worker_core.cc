#include "tensorflow/lite/worker_core.h"
#include "tensorflow/lite/interpreter.h"

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

namespace tflite{
  Worker::Worker(){
    
  };

  Worker::Worker(ResourceType wType, int w_id, Interpreter* interpreter){
    type = wType;
    worker_id = w_id;
    state = WorkerState::INIT_WORK;
    interpreter_ = interpreter; 
  };

  void Worker::ChangeStateTo(WorkerState new_state){
    state = new_state;
  }

  void Worker::GiveJob(tflite::Job* new_job){
    have_job = true;
    jobs.push_back(new_job);
  }

  bool Worker::TryLock(){
    if(mtx_lock.try_lock()){
      return true;
    }else{
      return false;
    }
  }

  void Worker::Unlock(){
    mtx_lock.unlock();
  }

  void Worker::WakeWorker(){
    worker_cv.notify_one();
  }

  void Worker::Work(){
    std::cout << "Worker [" << worker_id << "] started" << "\n";
    while(true){
      std::unique_lock<std::mutex> lock(mtx_lock);
      worker_cv.wait(lock, [&] { return state == WorkerState::WORKING; });
      for(int i=0; i<jobs.size(); ++i){
        Subgraph* working_graph; 
        if(jobs[i]->resource_type == type){
          int graphs_to_invoke = jobs[i]->subgraphs.size();
          for(int j=0; j<graphs_to_invoke; ++j){
            working_graph = interpreter_->subgraph(jobs[i]->subgraphs[j].first);
            if(working_graph->Invoke() != kTfLiteOk){
              std::cout << "Invoke returned Error" << "\n";
            }
            std::cout << "Worker " << worker_id << " job "
                        << jobs[i]->job_id << " done" << "\n";
          }
        }
      }
    }
  }

  Worker::~Worker(){

  };



} //namespace tflite