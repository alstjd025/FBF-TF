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

//legacy
  Worker::Worker(ResourceType wType, int w_id, Interpreter* interpreter){
    type = wType;
    worker_id = w_id;
    state = WorkerState::INIT_WORK;
    interpreter_ = interpreter; 
    working_thread = std::thread(&Worker::Work, this);
    working_thread.detach();
  };

//legacy
  void Worker::ChangeStateTo(WorkerState new_state){
    std::cout << " changed worker state1 " << "\n";
    std::unique_lock<std::mutex> lock(worker_lock);
    state = new_state;
    std::cout << " changed worker state 2" << "\n";
  }

  void Worker::DeleteJob(int job_id){
    std::unique_lock<std::mutex> lock(worker_lock);
    for(size_t i=0; jobs.size(); ++i){
      if(jobs[i]->job_id == job_id)
        jobs.erase(jobs.begin() + i);
    }
  }

//legacy
  void Worker::GiveJob(tflite::Job* new_job){
    std::unique_lock<std::mutex> lock(worker_lock);
    std::cout << "got job1" << "\n";
    if(!have_job)
      have_job = true;
    std::cout << "got job2" << "\n";
    jobs.push_back(new_job);
    std::cout << "got job3" << "\n";
  }

//legacy
  void Worker::WakeWorker(){
    worker_cv.notify_all();
  }

//legacy
  void Worker::Work(){
    std::cout << "Worker [" << worker_id << "] started" << "\n";
    while(true){
      std::unique_lock<std::mutex> lock(worker_lock);
      worker_cv.wait(lock, [&] { return state == WorkerState::WORKING; });
      std::cout << "Worker [" << worker_id << "] woke up" << "\n";
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
            interpreter_->LockJobs();
            jobs[i]->state == JobState::DONE;
            interpreter_->UnlockJobs();
          }
        }
      }
    }
  }

//legacy
  Worker::~Worker(){
    std::cout << "Worker destuctor called " << "\n";
  };



} //namespace tflite