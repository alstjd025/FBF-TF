#include "tensorflow/lite/workframe.h"

namespace tflite{

  WorkFrame::WorkFrame(){
  };

  WorkFrame::WorkFrame(const char* model_name){

    // Creates a new scheduler
    CreateProfilerandScheduler(model_name);
  };


  TfLiteStatus WorkFrame::CreateProfilerandScheduler(const char* model){
    // NEED DEBUG: use of make_shared here is quite ambigious.
    std::shared_ptr<std::queue<tflite::Job*>> new_job_queue = \
                          std::make_shared<std::queue<tflite::Job*>>();
    interpreter = std::make_shared<tflite::Interpreter>();
    scheduler_ = std::make_shared<tflite::Scheduler>(interpreter);
    factory_ = std::make_shared<tflite::ModelFactory>(interpreter);
    #ifdef DEBUG
      std::cout << "interpreter count :" << interpreter.use_count() << "\n";
    #endif
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