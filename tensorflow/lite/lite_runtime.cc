#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/lite_scheduler.h"

void PrintTensor(TfLiteTensor& tensor){
  std::cout << "[Print Tensor]" << "\n";
  int tensor_data_dims_size = tensor.dims->size-1;
  int tensor_data_ch_size = tensor.dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< tensor.dims->size; i++){
    if(i == 1){
      tensor_axis = tensor.dims->data[i];
    }
    tensor_data_size *= tensor.dims->data[i]; 
  }
  std::cout << " Nunber of data : " << tensor_data_size << "\n";
  std::cout << " Tensor DATA " << "\n";
  if(tensor.type == TfLiteType::kTfLiteFloat32){
    std::cout << "[FLOAT32 TENSOR]" << "\n";
    auto data_st = (float*)tensor.data.data;
    for(int i=0; i<tensor_data_ch_size; i++){
      std::cout << "CH [" << i << "] \n";
      for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
        float data = *(data_st+(i+j*tensor_data_ch_size));
        if (data == 0) {
          printf("%0.6f ", data);
        }
        else if (data != 0) {
          printf("%s%0.6f%s ", C_GREN, data, C_NRML);
        }
        if (j % tensor_axis == tensor_axis-1) {
          printf("\n");
        }
      }
      std::cout << "\n";
    }
  }
}


namespace tflite{

TfLiteRuntime::TfLiteRuntime(){
  interpreter = new tflite::Interpreter;

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
  scheduler = interpreter->GetScheduler();
};

TfLiteRuntime::~TfLiteRuntime(){
  std::cout << "TfLiteRuntime destructor called" << "\n";
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
  
  ProfileData dummy_profile;
  // fill dummy profile here //
  
  // end //

  tflite::InterpreterBuilder* new_builder = \
            new tflite::InterpreterBuilder(**model_, *resolver, interpreter, model, \
                                                  0, true, dummy_profile);
  
  // Now creates an invokable origin subgraph from new model.
  if(new_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk){
    std::cout << "CreateSubgraphFromFlatBuffer returned Error" << "\n";
    exit(-1);
  }
  
  scheduler->RegisterInterpreterBuilder(new_builder);
  // And schedule it for latency profiling

  return kTfLiteOk;
};

void TfLiteRuntime::FeedInputToInterpreter(std::vector<cv::Mat>& mnist,
                                          std::vector<cv::Mat>& imagenet){
  interpreter->mnist_input = mnist;
  interpreter->imagenet_input = imagenet;
}

void TfLiteRuntime::FeedInputToModel(const char* model,
                                    std::vector<cv::Mat>& input,
                                    INPUT_TYPE input_type){
  TfLiteTensor* input_tensor = nullptr;
  input_tensor = interpreter->input_tensor_of_model(0);

  if(input_tensor == nullptr){
    std::cout << "TfLiteRuntime : cannont get pointer to input tensor, model["
              << model << "]" << "\n";
    return;
  }
  auto input_pointer = (float*)input_tensor->data.data;
  switch (input_type)
  {
  case INPUT_TYPE::MNIST :
    for (int i=0; i<28; ++i){
      for (int j=0; j<28; ++j){
        input_pointer[i*28 + j] = ((float)input[0].at<uchar>(i, j)/255.0);          
      }
    }   
    break;
  case INPUT_TYPE::IMAGENET224 :
    memcpy(input_pointer, input[0].data, input[0].total() * input[0].elemSize());
    // for (int i=0; i < 224; ++i){
    //   for(int j=0; j < 224; ++j){
    //     input_pointer[i*224 + j*3] = (float)input[0].at<cv::Vec3b>(i, j)[0] /255.0;    
    //     input_pointer[i*224 + j*3 + 1] = (float)input[0].at<cv::Vec3b>(i, j)[1] /255.0;
    //     input_pointer[i*224 + j*3 + 2] = (float)input[0].at<cv::Vec3b>(i, j)[2] /255.0;
    //   }
    // }
    break;
  case INPUT_TYPE::IMAGENET300 :
    for (int i=0; i < 300; ++i){
      for(int j=0; j < 300; ++j){
        input_pointer[i*300 + j*3] = \
          ((float)input[0].at<cv::Vec3b>(i, j)[0]);    
        input_pointer[i*300 + j*3 + 1] = \
          ((float)input[0].at<cv::Vec3b>(i, j)[1]);
        input_pointer[i*300 + j*3 + 2] = \
          ((float)input[0].at<cv::Vec3b>(i, j)[2]);
      }
    }   
    break;
  default:
    break;
  }
  //PrintTensor(*input_tensor);
} 

void TfLiteRuntime::WakeScheduler(){
  interpreter->WakeScheduler();
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

void TfLiteRuntime::JoinScheduler(){
  interpreter->JoinScheduler();
}

TfLiteStatus TfLiteRuntime::DebugInvoke(){
  if(interpreter->DebugInvoke() != kTfLiteOk){
    return kTfLiteError;
  }
  return kTfLiteOk;
};



}