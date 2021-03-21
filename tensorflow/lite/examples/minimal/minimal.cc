/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <iostream>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#ifdef GPU_DELEGATE
	#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif
// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Intrepter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

//
#ifdef GPU_DELEGATE
    TfLiteDelegate *MyDelegate = NULL;

    const TfLiteGpuDelegateOptionsV2 options = {
        .is_precision_loss_allowed = 1, //FP16,
        .inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
        .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
        .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
        .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    };
    MyDelegate = TfLiteGpuDelegateV2Create(&options);

    if(interpreter->ModifyGraphWithDelegate(MyDelegate) != kTfLiteOk) {
        cerr << "ERROR: Unable to use delegate" << endl;
        return 0;
    }
#endif


  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  //TESTESTESTSETESTSETEST
//  std::cout <<"\n=============================================="<<std::endl;

 /* int In = interpreter->inputs()[0];
  int model_height = interpreter->tensor(In)->dims->data[1];
  int model_width = interpreter->tensor(In)->dims->data[2];
  std::cout << "height : " << model_height << std::endl;
  std::cout << "width : " << model_width << std::endl; 
    
 
  GetInput(interpreter->typed_tensor<float>(interpreter->inputs()[0]));
  std::cout <<"FSDAF";  



  const float* test = interpreter->tensor(interpreter->outputs()[0])->data.f;
  std::cout << *test << std::endl;

# */
// interpreter->typed_input_tensor<int>(0)[0] = 20;
//  std::cout <<"\n=============================================="<<std::endl;
  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  //printf("\ninterpreter->typed_input_tensor()[]\n");
  printf("=====set_input======\n\n");
  int input1 = 10;
  printf("input : %d\n\n", input1);
  interpreter->typed_input_tensor<int>(0)[0] = input1;
  //interpreter->typed_input_tensor<int>(0)[1] = 20;
	
  // Run inference
  printf("\n=====START Invoke=====\n\n");
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n=====End Invoke=====\n\n");
  //printf("\n\n=== Post-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());


  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  printf("=====get_output=====\n\n");
  float result = interpreter->typed_output_tensor<float>(0)[0];
  //float result2 = interpreter->typed_output_tensor<float>(0)[1];
  printf("\noutput : %f\n", result);


   return 0;
}
