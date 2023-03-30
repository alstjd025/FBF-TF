#include "tensorflow/lite/lite_runtime.h"




namespace tflite{

TfLiteRuntime::TfLiteRuntime(){
  interpreter = std::make_shared<tflite::Interpreter>();

  TfLiteDelegate *MyDelegate = NULL;
  const TfLiteGpuDelegateOptionsV2 options = {
      .is_precision_loss_allowed = 0, 
      .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
      //.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED,
      .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
      //.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
      .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .experimental_flags = 1,
      .max_delegated_partitions = 1000,
  };
  MyDelegate = TfLiteGpuDelegateV2Create(&options);
  interpreter->RegisterDelegate(MyDelegate);
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

void TfLiteRuntime::WakeScheduler(){
  interpreter->WakeScheduler();
  interpreter->JoinScheduler();
}

TfLiteStatus TfLiteRuntime::DebugInvoke(){
  if(interpreter->DebugInvoke() != kTfLiteOk){
    return kTfLiteError;
  }
  return kTfLiteOk;
};

}