#include "tensorflow/lite/lite_runtime.h"




namespace tflite{

TfLiteRuntime::TfLiteRuntime(){
  interpreter = std::make_shared<tflite::Interpreter>();
};

TfLiteRuntime::~TfLiteRuntime(){

};

TfLiteStatus TfLiteRuntime::CreateTfLiteRuntime(){
  return kTfLiteOk;
};

TfLiteStatus TfLiteRuntime::AddModelToRuntime(const char* model){
  std::unique_ptr<tflite::FlatBufferModel>* model_;
  
  model_ = new std::unique_ptr<tflite::FlatBufferModel>\
  (tflite::FlatBufferModel::BuildFromFile(model));

  // Build the interpreter with the InterpreterBuilder.
  tflite::ops::builtin::BuiltinOpResolver* resolver;
  resolver = new tflite::ops::builtin::BuiltinOpResolver;
  
  // Experimental API
  // Giving a model number from recieved model size?
  int model_number = builders_created;
  builders_created++;

  std::cout << "RUNTIME: Currently have " << model_number << " on runtime" << "\n";

  ProfileData dummy_profile;
  // fill dummy profile here //

  // end //

  tflite::InterpreterBuilder* new_builder = \
            new tflite::InterpreterBuilder(**model_, *resolver, model, \
                                                      model_number, true, dummy_profile);

  builder_and_id.insert({model_number, new_builder});
  
  // Now creates an invokable origin subgraph from new model.
  if(new_builder->CreateSubgraphFromFlatBuffer(interpreter) != kTfLiteOk){
    std::cout << "CreateSubgraphFromFlatBuffer returned Error" << "\n";
    exit(-1);
  }
  // And schedule it for latency profiling
  

  return kTfLiteOk;
};


TfLiteStatus TfLiteRuntime::DebugInvoke(){
  if(interpreter->DebugInvoke() != kTfLiteOk){
    return kTfLiteError;
  }
  return kTfLiteOk;
};

}