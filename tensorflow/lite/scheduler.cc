#include "tensorflow/lite/scheduler.h"

namespace tflite{

  Scheduler::Scheduler(){
    state = SchedulerState::INIT;
    interpreter_ = nullptr;
  };

  Scheduler::Scheduler(std::shared_ptr<tflite::Interpreter> interpreter){
    state = SchedulerState::INIT;
    interpreter_ = interpreter;
  }

  Scheduler::~Scheduler(){
      
  };

  TfLiteStatus Scheduler::Reschedule(){
    
  };


  // Profiler codes //
  ModelFactory::ModelFactory(){
    interpreter_ = nullptr;
  };

  ModelFactory::ModelFactory(std::shared_ptr<tflite::Interpreter> interpreter){
    interpreter_ = interpreter;
  }

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

  TfLiteStatus ModelFactory::GiveSubgraphtoInterperter(\
                                      std::shared_ptr<Scheduler> scheduler){

  };

  ModelFactory::~ModelFactory(){

  };



} //namespace tflite