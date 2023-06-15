#include "tensorflow/lite/lite_runtime.h"

#include "tensorflow/lite/lite_scheduler.h"

void PrintTensor(TfLiteTensor& tensor) {
  std::cout << "[Print Tensor]"
            << "\n";
  int tensor_data_dims_size = tensor.dims->size - 1;
  int tensor_data_ch_size = tensor.dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for (int i = 0; i < tensor.dims->size; i++) {
    if (i == 1) {
      tensor_axis = tensor.dims->data[i];
    }
    tensor_data_size *= tensor.dims->data[i];
  }
  std::cout << " Nunber of data : " << tensor_data_size << "\n";
  std::cout << " Tensor DATA "
            << "\n";
  if (tensor.type == TfLiteType::kTfLiteFloat32) {
    std::cout << "[FLOAT32 TENSOR]"
              << "\n";
    auto data_st = (float*)tensor.data.data;
    for (int i = 0; i < tensor_data_ch_size; i++) {
      std::cout << "CH [" << i << "] \n";
      for (int j = 0; j < tensor_data_size / tensor_data_ch_size; j++) {
        float data = *(data_st + (i + j * tensor_data_ch_size));
        if (data == 0) {
          printf("%0.6f ", data);
        } else if (data != 0) {
          printf("%s%0.6f%s ", C_GREN, data, C_NRML);
        }
        if (j % tensor_axis == tensor_axis - 1) {
          printf("\n");
        }
      }
      std::cout << "\n";
    }
  }
}

// cascade operator overloading for debug message.
std::ostream& operator<<(std::ostream& out, const tflite::RuntimeState value){
  const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
  switch(value){
    PROCESS_VAL(tflite::INITIALIZE);     
    PROCESS_VAL(tflite::NEED_PROFILE);     
    PROCESS_VAL(tflite::SUBGRAPH_CREATE);
    PROCESS_VAL(tflite::INVOKE_);
  }
#undef PROCESS_VAL
  return out << s;
}

namespace tflite {

TfLiteRuntime::TfLiteRuntime(char* uds_runtime, char* uds_scheduler,
                                     const char* model, INPUT_TYPE type) {
  interpreter = new tflite::Interpreter(true);
  quantized_interpreter = nullptr;
  quantized_builder = nullptr;
  interpreter->SetInputType(type);
  state = RuntimeState::INITIALIZE;
  uds_runtime_filename = uds_runtime;
  uds_scheduler_filename = uds_scheduler;
  TfLiteDelegate* MyDelegate = NULL;
  const TfLiteGpuDelegateOptionsV2 options = {
      .is_precision_loss_allowed = 0,
      .inference_preference =
          TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
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
  if(InitializeUDS() != kTfLiteOk){
    std::cout << "UDS socker init ERROR" << "\n";
    exit(-1);
  }
  if(AddModelToRuntime(model) != kTfLiteOk){
    std::cout << "Model registration to runtime ERROR" << "\n";
    exit(-1);
  }
  if(RegisterModeltoScheduler() != kTfLiteOk){
    std::cout << "Model registration to scheduler ERROR" << "\n";
    exit(-1);
  }
  if(PartitionSubgraphs() != kTfLiteOk){
    std::cout << "Model partitioning ERROR" << "\n";
    exit(-1);
  }
  
};

TfLiteRuntime::TfLiteRuntime(char* uds_runtime, char* uds_scheduler,
                      const char* f_model, const char* i_model, INPUT_TYPE type) {
  co_execution = true;
  interpreter = new tflite::Interpreter(true);
  quantized_interpreter = new tflite::Interpreter(true);
  quantized_builder = nullptr;
  interpreter->SetInputType(type);
  quantized_interpreter->SetInputType(type);
  state = RuntimeState::INITIALIZE;
  uds_runtime_filename = uds_runtime;
  uds_scheduler_filename = uds_scheduler;
  TfLiteDelegate* MyDelegate = NULL;
  const TfLiteGpuDelegateOptionsV2 options = {
      .is_precision_loss_allowed = 0,
      .inference_preference =
          TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
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
  if(InitializeUDS() != kTfLiteOk){
    std::cout << "UDS socker init ERROR" << "\n";
    exit(-1);
  }
  if(AddModelToRuntime(f_model, i_model) != kTfLiteOk){
    std::cout << "Model registration to runtime ERROR" << "\n";
    exit(-1);
  }
  if(RegisterModeltoScheduler() != kTfLiteOk){
    std::cout << "Model registration to scheduler ERROR" << "\n";
    exit(-1);
  }
  if(PartitionCoSubgraphs() != kTfLiteOk){
    std::cout << "Model partitioning ERROR" << "\n";
    exit(-1);
  }
  
  InitLogFile();
};

TfLiteRuntime::~TfLiteRuntime() {
  std::cout << "TfLiteRuntime destructor called"
            << "\n";
};

void TfLiteRuntime::InitLogFile(){
  logFile.open("latency.txt");
  logFile_.open("latency_.txt");
  return;
}

void TfLiteRuntime::WriteVectorLog(std::vector<double>& log, int n){
  if(logFile.is_open() && n == 0){
    for(int i=0; i<log.size(); ++i){
      logFile << log[i] << " ";
    }
    logFile << "\n";
  }else if(logFile_.is_open() && n == 1){
    for(int i=0; i<log.size(); ++i){
      logFile_ << log[i] << " ";
    }
    logFile_ << "\n";
  }
  else{
    std::cout << "Log file not open ERROR" << "\n";
    return;
  }
}

TfLiteStatus TfLiteRuntime::InitializeUDS(){
  // Delete runtime socket if already exists.
  if(access(uds_runtime_filename, F_OK) == 0)
    unlink(uds_runtime_filename);
  
  // Create a UDS socket for TFruntime.
  runtime_sock = socket(PF_FILE, SOCK_DGRAM, 0);
  if(runtime_sock == -1){
    std::cout << "Socket create ERROR" << "\n";
    return kTfLiteError;
  }

  memset(&runtime_addr, 0, sizeof(runtime_addr));
  runtime_addr.sun_family = AF_UNIX; // unix domain socket
  strcpy(runtime_addr.sun_path, uds_runtime_filename);

  memset(&scheduler_addr, 0, sizeof(scheduler_addr));
  scheduler_addr.sun_family = AF_UNIX; // unix domain socket
  strcpy(scheduler_addr.sun_path, uds_scheduler_filename);
  addr_size = sizeof(scheduler_addr);

  // Bind runtime socket for TX,RX with scheduler
  if(bind(runtime_sock, (struct sockaddr*)&runtime_addr, sizeof(runtime_addr))
        == -1){
    std::cout << "Socket bind ERROR" << "\n";
    return kTfLiteError;
  }
  tf_packet new_packet;
  memset(&new_packet, 0, sizeof(tf_packet));
  new_packet.runtime_current_state = 0;
  new_packet.runtime_id = -1;


  if(SendPacketToScheduler(new_packet) != kTfLiteOk){
    std::cout << "Sending Hello to scheduler FAILED" << "\n";
    return kTfLiteError;
  }
  std::cout << "Send runtime register request to scheduler" << "\n";

  tf_packet recv_packet;
  if(ReceivePacketFromScheduler(recv_packet) != kTfLiteOk){
    std::cout << "Receiving packet from scheduler FAILED" << "\n";
    return kTfLiteError;
  }

  runtime_id = recv_packet.runtime_id;
  std::cout << "Got runtime ID " << runtime_id << " from scheduler" << "\n";
  
  if(ChangeStatewithPacket(recv_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::SendPacketToScheduler(tf_packet& tx_p){
  if(sendto(runtime_sock, (void *)&tx_p, sizeof(tf_packet), 0,
            (struct sockaddr*)&scheduler_addr, sizeof(scheduler_addr)) == -1){
    std::cout << "Sending packet to scheduler FAILED" << "\n";
    return kTfLiteError;
  }
  return kTfLiteOk; 
}

TfLiteStatus TfLiteRuntime::ReceivePacketFromScheduler(tf_packet& rx_p){
  if(recvfrom(runtime_sock, &rx_p, sizeof(tf_packet), 0 , NULL, 0) == -1){
    std::cout << "Receiving packet from scheduler FAILED" << "\n";
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::ChangeStatewithPacket(tf_packet& rx_p){
  std::cout << "================================================" << "\n";
  if(rx_p.runtime_next_state != state){
    std::cout << "runtime_next_state : " << rx_p.runtime_next_state << "\n";
    state = static_cast<RuntimeState>(rx_p.runtime_next_state);
    std::cout << "Runtime " << runtime_id << " state changed to " << state << "\n";
    std::cout << "================================================" << "\n";
    return kTfLiteOk;
  }else{
    std::cout << "Runtime " << runtime_id << " state no change." << "\n";
    std::cout << "================================================" << "\n";
    return kTfLiteOk;
  }
}

TfLiteStatus TfLiteRuntime::AddModelToRuntime(const char* model) {
  std::unique_ptr<tflite::FlatBufferModel>* model_ = 
        new std::unique_ptr<tflite::FlatBufferModel>(
            tflite::FlatBufferModel::BuildFromFile(model));

  // Build the interpreter with the InterpreterBuilder.
  tflite::ops::builtin::BuiltinOpResolver* resolver =
           new tflite::ops::builtin::BuiltinOpResolver;

  interpreter_builder = new tflite::InterpreterBuilder(
      **model_, *resolver, interpreter, model, 0, false);

  // Now creates an invokable origin subgraph from new model.
  if (interpreter_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk) {
    std::cout << "CreateSubgraphFromFlatBuffer returned Error"
              << "\n";
    exit(-1);
  }
  interpreter->PrintSubgraphInfo();
  // scheduler->RegisterInterpreterBuilder(new_builder);

  return kTfLiteOk;
};

TfLiteStatus TfLiteRuntime::AddModelToRuntime(const char* f_model,
                                               const char* i_model) {
  std::unique_ptr<tflite::FlatBufferModel>* float_model = 
        new std::unique_ptr<tflite::FlatBufferModel>(
            tflite::FlatBufferModel::BuildFromFile(f_model));

  std::unique_ptr<tflite::FlatBufferModel>* int_model = 
        new std::unique_ptr<tflite::FlatBufferModel>(
            tflite::FlatBufferModel::BuildFromFile(i_model));

  // An opresolver for float interpreter
  tflite::ops::builtin::BuiltinOpResolver* float_resolver =
           new tflite::ops::builtin::BuiltinOpResolver;

  // An opresolver for int interpreter
  tflite::ops::builtin::BuiltinOpResolver* int_resolver =
           new tflite::ops::builtin::BuiltinOpResolver;

  // Build InterpreterBuilder for float model
  interpreter_builder = new tflite::InterpreterBuilder(
      **float_model, *float_resolver, interpreter, f_model, 0, false);

  // Build IntpertereBuilder for int model
  quantized_builder = new tflite::InterpreterBuilder(
      **int_model, *int_resolver, quantized_interpreter, i_model, 0, true);

  // Now creates an invokable (float)origin subgraph from new model.
  if (interpreter_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk) {
    std::cout << "CreateSubgraphFromFlatBuffer returned Error"
              << "\n";
    exit(-1);
  }

  // Now creates an invokable (int)origin subgraph from new model.
  if (quantized_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk) {
    std::cout << "CreateSubgraphFromFlatBuffer returned Error"
              << "\n";
    exit(-1);
  }
  std::cout << "============================" << "\n";
  std::cout << "Full precision interpreter" << "\n";
  PrintInterpreterStateV2(interpreter);
  std::cout << "============================" << "\n";
  std::cout << "Minimal precision interpreter" << "\n";
  PrintInterpreterStateV2(quantized_interpreter);
  //interpreter->PrintSubgraphInfo();
  
  return kTfLiteOk;
};

TfLiteStatus TfLiteRuntime::RegisterModeltoScheduler(){
  if(state != RuntimeState::NEED_PROFILE){
    std::cout << "State is " << state << ". RegisterModeltoScheduler must called in "
             << RuntimeState::NEED_PROFILE << "\n";
    return kTfLiteError;
  }
  tf_packet tx_packet;
  memset(&tx_packet, 0, sizeof(tf_packet));
  tx_packet.runtime_current_state = state;
  tx_packet.runtime_id = runtime_id;

  //////////////////////////////////////
  // Some profiling logic here. later. //
  //////////////////////////////////////

  int layers = interpreter->nodes_size(0);
  for(int i=0; i<layers; ++i){
    tx_packet.latency[i] = -1.0; // means that this is a dummy latency profile.
  }
  if(SendPacketToScheduler(tx_packet) != kTfLiteOk){
    std::cout << "Sending profile packet to scheduler failed" << "\n";
    return kTfLiteError;
  }

  tf_packet rx_packet;
  if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
    std::cout << "Receiving partitioning plan packet from scheduler Failed" << "\n";
    return kTfLiteError;
  }

  // copy the partitioning plan from scheduler.
  memcpy(partitioning_plan, rx_packet.partitioning_plan, sizeof(int)*1000*4);

  if(ChangeStatewithPacket(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  std::cout << "Successfully registered model to scheduler" << "\n";
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::PartitionSubgraphs(){
  std::vector<std::vector<int>> raw_plan;
  for(int i=0; i<TF_P_PLAN_LENGTH; ++i){
    raw_plan.push_back(std::vector<int>());
    if(partitioning_plan[i][TF_P_IDX_START] == TF_P_END_PLAN){
      raw_plan[i].push_back(TF_P_END_PLAN);
      break;
    }
    for(int j=0; j<TF_P_PLAN_SIZE; ++j){ // third idx means processor.
      raw_plan[i].push_back(partitioning_plan[i][j]);
    }
  } 
  interpreter_builder->CopyRawPartitioningPlan(raw_plan);
  Subgraph* origin_subgraph = interpreter->returnProfiledOriginalSubgraph(0);
  if(origin_subgraph == nullptr){
    std::cout << "Model id " << interpreter_builder->GetModelid() << " no subgraph. \n"; 
    return kTfLiteError;
  }
  if(interpreter_builder->CreateSubgraphsFromProfiling(origin_subgraph)
      != kTfLiteOk){
    std::cout << "CreateSubgraphsFromProfiling returned ERROR" << "\n";
    return kTfLiteError;
  }
  
  tf_packet tx_packet;
  memset(&tx_packet, 0, sizeof(tf_packet));
  tx_packet.runtime_id = runtime_id;
  tx_packet.runtime_current_state = state;
  if(SendPacketToScheduler(tx_packet) != kTfLiteOk){
    return kTfLiteError;
  }

  tf_packet rx_packet;
  if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  // At this point, scheduler will send INVOKE state.
  if(ChangeStatewithPacket(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  interpreter->PrintSubgraphInfo();
  PrintInterpreterStateV2(interpreter);
  std::cout << "Successfully partitioned subgraph" << "\n";
  std::cout << "Ready to invoke" << "\n";
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::PartitionCoSubgraphs(){
  std::vector<std::vector<int>> raw_plan;
  for(int i=0; i<TF_P_PLAN_LENGTH; ++i){
    raw_plan.push_back(std::vector<int>());
    if(partitioning_plan[i][TF_P_IDX_START] == TF_P_END_PLAN){
      raw_plan[i].push_back(TF_P_END_PLAN);
      break;
    }
    for(int j=0; j<TF_P_PLAN_SIZE; ++j){ // third idx means processor.
      raw_plan[i].push_back(partitioning_plan[i][j]);
    }
  }  
  // Create subgraphs of float model
  interpreter_builder->CopyRawPartitioningPlan(raw_plan);
  Subgraph* origin_subgraph = interpreter->returnProfiledOriginalSubgraph(0);
  if(origin_subgraph == nullptr){
    std::cout << "Model id " << interpreter_builder->GetModelid() << " no subgraph. \n"; 
    return kTfLiteError;
  }
  if(interpreter_builder->CreateSubgraphsFromProfiling(origin_subgraph)
      != kTfLiteOk){
    std::cout << "CreateSubgraphsFromProfiling returned ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "===============================" << "\n";
  std::cout << "Full precision subgraph created" << "\n";
  std::cout << "===============================" << "\n";
  // Create subgraphs of quantized model
  quantized_builder->CopyRawPartitioningPlan(raw_plan);
  Subgraph* origin_quantized_subgraph = quantized_interpreter->returnProfiledOriginalSubgraph(0);
  if(origin_quantized_subgraph == nullptr){
    std::cout << "Model id " << interpreter_builder->GetModelid() << " no subgraph. \n"; 
    return kTfLiteError;
  }
  if(quantized_builder->CreateSubgraphsFromProfiling(origin_quantized_subgraph)
      != kTfLiteOk){
    std::cout << "CreateSubgraphsFromProfiling returned ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "===============================" << "\n";
  std::cout << "Minimal precision subgraph created" << "\n";
  std::cout << "===============================" << "\n";
  tf_packet tx_packet;
  memset(&tx_packet, 0, sizeof(tf_packet));
  tx_packet.runtime_id = runtime_id;
  tx_packet.runtime_current_state = state;
  if(SendPacketToScheduler(tx_packet) != kTfLiteOk){
    return kTfLiteError;
  }

  tf_packet rx_packet;
  if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  // At this point, scheduler will send INVOKE state.
  if(ChangeStatewithPacket(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  //interpreter->PrintSubgraphInfo();
  if(PrepareCoExecution() != kTfLiteOk){
    std::cout << "PrepareCoExecution returned ERROR" << "\n";
    return kTfLiteError;
  }
  std::cout << "=====================" << "\n";
  std::cout << "GPU Interpreter state" << "\n";
  PrintInterpreterStateV2(interpreter);
  std::cout << "=====================" << "\n";
  std::cout << "CPU Interpreter state" << "\n";
  PrintInterpreterStateV2(quantized_interpreter);
  std::cout << "Successfully partitioned subgraph" << "\n";
  std::cout << "Ready to invoke" << "\n";
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::PrepareCoExecution(){
  if(quantized_interpreter == nullptr){
    std::cout << "PrepareCoExecution ERROR" << "\n";
    std::cout << "minimal precision interpreter nullptr" << "\n";
    return kTfLiteError;
  }
  int co_subgraph_idx = 0;
  for(int subgraph_idx=0; subgraph_idx < interpreter->subgraphs_size();
        subgraph_idx++){
    Subgraph* subgraph = interpreter->subgraph(subgraph_idx);
    if(subgraph->GetResourceType() == ResourceType::CO_GPU){
      if(quantized_interpreter->subgraphs_size() < 1){
        std::cout << "PrepareCoExecution ERROR" << "\n";
        std::cout << "minimal precision interpreter has no subgraph" << "\n";
        return kTfLiteError;
      }
      Subgraph* co_subgraph = quantized_interpreter->subgraph(co_subgraph_idx);
      std::vector<int> inputs = subgraph->inputs();
      std::vector<int> outputs = subgraph->outputs();
      for(int i=0; i<inputs.size(); ++i){
        co_subgraph->PushToInputs(inputs[i]);
      }
      for(int i=0; i<outputs.size(); ++i){
        co_subgraph->PushToOutputs(outputs[i]);
      }
      co_subgraph_idx++;
    } 
  }
  return kTfLiteOk;
}

void TfLiteRuntime::FeedInputToInterpreter(std::vector<cv::Mat>& mnist,
                                           std::vector<cv::Mat>& imagenet) {
  interpreter->mnist_input = mnist;
  interpreter->imagenet_input = imagenet;
  if(quantized_interpreter != nullptr){
    quantized_interpreter->mnist_input = mnist;
    quantized_interpreter->imagenet_input = imagenet;
  }
}

void TfLiteRuntime::FeedInputToModel(const char* model,
                                     std::vector<cv::Mat>& input,
                                     INPUT_TYPE input_type) {
  TfLiteTensor* input_tensor = nullptr;
  input_tensor = interpreter->input_tensor_of_model(0);

  if (input_tensor == nullptr) {
    std::cout << "TfLiteRuntime : cannont get pointer to input tensor, model["
              << model << "]"
              << "\n";
    return;
  }
  auto input_pointer = (float*)input_tensor->data.data;
  switch (input_type) {
    case INPUT_TYPE::MNIST:
      for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
          input_pointer[i * 28 + j] = ((float)input[0].at<uchar>(i, j) / 255.0);
        }
      }
      break;
    case INPUT_TYPE::IMAGENET224:
      memcpy(input_pointer, input[0].data,
             input[0].total() * input[0].elemSize());
      break;
    case INPUT_TYPE::IMAGENET300:
      for (int i = 0; i < 300; ++i) {
        for (int j = 0; j < 300; ++j) {
          input_pointer[i * 300 + j * 3] =
              ((float)input[0].at<cv::Vec3b>(i, j)[0]);
          input_pointer[i * 300 + j * 3 + 1] =
              ((float)input[0].at<cv::Vec3b>(i, j)[1]);
          input_pointer[i * 300 + j * 3 + 2] =
              ((float)input[0].at<cv::Vec3b>(i, j)[2]);
        }
      }
      break;
    default:
      break;
  }
  // PrintTensor(*input_tensor);
}

// TODO : Impl global input tensor sharing
void TfLiteRuntime::FeedInputToModelDebug(const char* model,
                                     cv::Mat& input,
                                     INPUT_TYPE input_type) {
  // for (int i = 0; i < 28; ++i) {
  //   for (int j = 0; j < 28; ++j) {
  //     printf("%.6f ", ((float)input.at<uchar>(i,j) / 255.0));
  //   }
  //   std::cout << "\n";
  // }
  bool use_two_interpreter = false;
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* quant_input_tensor = nullptr;
  input_tensor = interpreter->input_tensor_of_model(0);
  if(quantized_interpreter != nullptr && quantized_interpreter->subgraphs_size() > 0){
    quant_input_tensor = quantized_interpreter->input_tensor_of_model(0);
    if(quant_input_tensor == nullptr) {
      std::cout << "TfLiteRuntime : cannont get pointer to quant input tensor, model["
              << model << "]"
              << "\n";
      return;
    }
    use_two_interpreter = true;
  }
  
  if(input_tensor == nullptr) {
    std::cout << "TfLiteRuntime : cannont get pointer to input tensor, model["
              << model << "]"
              << "\n";
    return;
  }
  auto input_pointer = (float*)input_tensor->data.data;
  int h = input_tensor->dims->data[1];
  int w = input_tensor->dims->data[2];
  switch (input_type) {
    case INPUT_TYPE::MNIST:
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          input_pointer[i * w + j] = ((float)input.at<uchar>(i, j) / 255.0);
        }
      }
      break;
    case INPUT_TYPE::IMAGENET224:
      memcpy(input_pointer, input.data,
             input.total() * input.elemSize());
      // for (int i = 0; i < 224; ++i) {
      //   for (int j = 0; j < 224; ++j) {
      //     input_pointer[i * 224 + j * 3] =
      //         ((float)input.at<cv::Vec3b>(i, j)[0]);
      //     input_pointer[i * 224 + j * 3 + 1] =
      //         ((float)input.at<cv::Vec3b>(i, j)[1]);
      //     input_pointer[i * 224 + j * 3 + 2] =
      //         ((float)input.at<cv::Vec3b>(i, j)[2]);
      //   }
      // }
      break;
    case INPUT_TYPE::IMAGENET300:
      for (int i = 0; i < 300; ++i) {
        for (int j = 0; j < 300; ++j) {
          input_pointer[i * 300 + j * 3] =
              ((float)input.at<cv::Vec3b>(i, j)[0]);
          input_pointer[i * 300 + j * 3 + 1] =
              ((float)input.at<cv::Vec3b>(i, j)[1]);
          input_pointer[i * 300 + j * 3 + 2] =
              ((float)input.at<cv::Vec3b>(i, j)[2]);
        }
      }
      break;
    case INPUT_TYPE::IMAGENET416:
      memcpy(input_pointer, input.data,
             input.total() * input.elemSize());
      // for (int i = 0; i < 416; ++i) {
      //   for (int j = 0; j < 416; ++j) {
      //     input_pointer[i * 416 + j * 3] =
      //         ((float)input.at<cv::Vec3b>(i, j)[0]);
      //     input_pointer[i * 416 + j * 3 + 1] =
      //         ((float)input.at<cv::Vec3b>(i, j)[1]);
      //     input_pointer[i * 416 + j * 3 + 2] =
      //         ((float)input.at<cv::Vec3b>(i, j)[2]);
      //   }
      // }
      break;
    default:
      break;
  }
  // PrintTensorSerial(*input_tensor);
  if(false){
    auto q_input_pointer = (float*)quant_input_tensor->data.data;
    h = quant_input_tensor->dims->data[1];
    w = quant_input_tensor->dims->data[2];
    switch (input_type) {
      case INPUT_TYPE::MNIST:
      for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
          q_input_pointer[i * w + j] = ((float)input.at<uchar>(i+(28-h), j) / 255.0);
        }
      }
        break;
      case INPUT_TYPE::IMAGENET224:
        memcpy(q_input_pointer, input.data,
              input.total() * input.elemSize());
        break;
      case INPUT_TYPE::IMAGENET300:
        for (int i = 0; i < 300; ++i) {
          for (int j = 0; j < 300; ++j) {
            q_input_pointer[i * 300 + j * 3] =
                ((float)input.at<cv::Vec3b>(i, j)[0]);
            q_input_pointer[i * 300 + j * 3 + 1] =
                ((float)input.at<cv::Vec3b>(i, j)[1]);
            q_input_pointer[i * 300 + j * 3 + 2] =
                ((float)input.at<cv::Vec3b>(i, j)[2]);
          }
        }
        break;
    case INPUT_TYPE::IMAGENET416:
      memcpy(q_input_pointer, input.data,
             input.total() * input.elemSize());
      break;
      default:
        break;
    }
  }
  // PrintTensorSerial(*quant_input_tensor);
}

void TfLiteRuntime::FeedInputToModel(const char* model,
                                     cv::Mat& input,
                                     INPUT_TYPE input_type) {
  TfLiteTensor* input_tensor = nullptr;
  input_tensor = interpreter->input_tensor_of_model(0);

  if (input_tensor == nullptr) {
    std::cout << "TfLiteRuntime : cannont get pointer to input tensor, model["
              << model << "]"
              << "\n";
    return;
  }
  auto input_pointer = (float*)input_tensor->data.data;
  switch (input_type) {
    case INPUT_TYPE::MNIST:
      for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
          input_pointer[i * 28 + j] = ((float)input.at<uchar>(i, j) / 255.0);
        }
      }
      break;
    case INPUT_TYPE::IMAGENET224:
      memcpy(input_pointer, input.data,
             input.total() * input.elemSize());
      // for (int i=0; i < 224; ++i){
      //   for(int j=0; j < 224; ++j){
      //     input_pointer[i*224 + j*3] = (float)input[0].at<cv::Vec3b>(i, j)[0]
      //     /255.0; input_pointer[i*224 + j*3 + 1] =
      //     (float)input[0].at<cv::Vec3b>(i, j)[1] /255.0; input_pointer[i*224
      //     + j*3 + 2] = (float)input[0].at<cv::Vec3b>(i, j)[2] /255.0;
      //   }
      // }
      break;
    case INPUT_TYPE::IMAGENET300:
      for (int i = 0; i < 300; ++i) {
        for (int j = 0; j < 300; ++j) {
          input_pointer[i * 300 + j * 3] =
              ((float)input.at<cv::Vec3b>(i, j)[0]);
          input_pointer[i * 300 + j * 3 + 1] =
              ((float)input.at<cv::Vec3b>(i, j)[1]);
          input_pointer[i * 300 + j * 3 + 2] =
              ((float)input.at<cv::Vec3b>(i, j)[2]);
        }
      }
      break;
    default:
      break;
  }
  // PrintTensor(*input_tensor);
}

void TfLiteRuntime::WakeScheduler() {
  interpreter->WakeScheduler();
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

void TfLiteRuntime::JoinScheduler() { interpreter->JoinScheduler(); }

TfLiteStatus TfLiteRuntime::DebugCoInvoke(){
  c_thread = std::thread(&TfLiteRuntime::DebugSyncInvoke, this, 
                          PrecisionType::MINIMAL_PRECISION);
  DebugSyncInvoke(PrecisionType::MAX_PRECISION);
  c_thread.join();
}

void TfLiteRuntime::DebugSyncInvoke(PrecisionType type){
  // For prototye, invoke first layer only with HW-partitioning and merge them.
  Subgraph* subgraph;
  int subgraph_idx = 0;
  double response_time = 0; 
  std::vector<double> latency;
  struct timespec begin, end;
  while(true){
    if(type == PrecisionType::MINIMAL_PRECISION){
      if(quantized_interpreter->subgraphs_size() < 1){
        // std::cout << "No invokable subgraph for cpu" << "\n";
        break;
      }
      // sync with gpu here (notified by gpu)
      std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
      invoke_sync_cv.wait(lock_invoke, [&]{ return invoke_cpu; });
      invoke_cpu = false;
      subgraph = quantized_interpreter->subgraph(subgraph_idx);
      if(main_execution_graph != nullptr)
        CopyIntermediateDataIfNeeded(subgraph, main_execution_graph);
      // std::cout << "[Minimal precision] Invoke subgraph " << subgraph->GetGraphid() << "\n";
      clock_gettime(CLOCK_MONOTONIC, &begin);
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking CPU subgraph " << subgraph->GetGraphid() << "\n";
        return;
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      response_time =  (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      latency.push_back(response_time);
      // sync with gpu here (wake gpu)
      std::unique_lock<std::mutex> lock_data(data_sync_mtx);
      is_execution_done = true;
      co_execution_graph = subgraph;
      data_sync_cv.notify_one();
      if(subgraph->GetNextSubgraph() != nullptr)
        subgraph_idx++;
      else{
        WriteVectorLog(latency, 1);
        // std::cout << "Minimal precision graph invoke done" << "\n";
        break;
      }
    }else if(type == PrecisionType::MAX_PRECISION){
      subgraph = interpreter->subgraph(subgraph_idx);
      if(subgraph->GetResourceType() == CO_GPU){
        // wake cpu thread here
        std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
        invoke_cpu = true;
        main_execution_graph = subgraph;
        invoke_sync_cv.notify_one();
      }else{ // if not co-execution, it needs additional imtermediate data copy.
        if(subgraph->GetPrevSubgraph() != nullptr &&
            subgraph->GetPrevSubgraph()->GetResourceType() != ResourceType::CO_GPU){
          CopyIntermediateDataIfNeeded(subgraph);
        }
      }
      // std::cout << "[Max precision] Invoke subgraph " << subgraph->GetGraphid() << "\n";
      clock_gettime(CLOCK_MONOTONIC, &begin);
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking subgraph id " << subgraph->GetGraphid() << "\n";
        return;
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      response_time =  (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      latency.push_back(response_time);
      if(subgraph->GetResourceType() == ResourceType::CO_GPU){
        // sync with cpu here
        std::unique_lock<std::mutex> lock_data(data_sync_mtx);
        data_sync_cv.wait(lock_data, [&]{ return is_execution_done; });
        is_execution_done = false;
        // data merge here
        if(co_execution_graph != nullptr){
          MergeCoExecutionData(co_execution_graph, subgraph);
          co_execution_graph = nullptr;
          // int input_tensor = subgraph->GetNextSubgraph()->GetInputTensorIndex();
          // PrintTensorSerial(*(subgraph->GetNextSubgraph()->tensor(input_tensor)));
        }
      }
      if(subgraph->GetNextSubgraph() != nullptr){
        subgraph_idx++;
      }
      else{
        WriteVectorLog(latency, 0);
        // std::cout << "Max precision graph invoke done" << "\n";
        // PrintyoloOutput(*(subgraph->tensor(109)));
        // PrintTensorSerial(*(subgraph->tensor(subgraph->GetOutputTensorIndex())));
        global_output_tensor = subgraph->tensor(subgraph->GetOutputTensorIndex());
        break;
      }
    }
  }
  return;
}

TfLiteStatus TfLiteRuntime::DebugInvoke() {
  Subgraph* subgraph;
  if(quantized_interpreter != nullptr){
    int subgraph_idx = 0;
    while(subgraph_idx < quantized_interpreter->subgraphs_size()){ // subgraph iteration
      subgraph = quantized_interpreter->subgraph(subgraph_idx);
      if(subgraph->GetPrevSubgraph() != nullptr){
        CopyIntermediateDataIfNeeded(subgraph);
      }
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking subgraph " << subgraph->GetGraphid() << "\n";
        return kTfLiteError;
      }
      subgraph_idx++;
    }
  }else{
    int subgraph_idx = 0;
    while(subgraph_idx < interpreter->subgraphs_size()){ // subgraph iteration
      subgraph = interpreter->subgraph(subgraph_idx);
      if(subgraph->GetPrevSubgraph() != nullptr){
        CopyIntermediateDataIfNeeded(subgraph);
      }
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking subgraph " << subgraph->GetGraphid() << "\n";
        return kTfLiteError;
      }
      subgraph_idx++;
    }
  }
  return kTfLiteOk;
};

TfLiteStatus TfLiteRuntime::Invoke(){
  TfLiteStatus state;
  if(co_execution){
    state = InvokeCoExecution();
  }else{
    state = InvokeSingleExecution();
  }
  return state;
}

TfLiteStatus TfLiteRuntime::InvokeCoExecution(){
  if(state != RuntimeState::INVOKE_){
    std::cout << "ERROR cannot invoke runtime [" << runtime_id << "]\n";
    std::cout << "State is not INVOKE. cur state is " << state << "\n";
    return kTfLiteError;
  }
  Subgraph* subgraph;
  int subgraph_idx = 0;
  while(subgraph_idx < interpreter->subgraphs_size()){ // subgraph iteration
    subgraph = interpreter->subgraph(subgraph_idx);
    tf_packet tx_packet;
    memset(&tx_packet, 0, sizeof(tf_packet));
    tx_packet.runtime_id = runtime_id;
    tx_packet.runtime_current_state = state;

    if(subgraph != nullptr){
      if(subgraph->GetResourceType() == ResourceType::CPU)
        tx_packet.cur_graph_resource = 0;
      else if(subgraph->GetResourceType() == ResourceType::GPU)
        tx_packet.cur_graph_resource = 1;
      else // Subject to change. (impl CPUGPU Co-execution)
        tx_packet.cur_graph_resource = 0;
    }
    
    if(SendPacketToScheduler(tx_packet) != kTfLiteOk){ // Request invoke permission to scheduler
      return kTfLiteError;
    }
    tf_packet rx_packet;
    if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
      return kTfLiteError;
    }
    switch (rx_packet.runtime_next_state)
    {
    case RuntimeState::INVOKE_ :{
      // Invoke next subgraph in subgraph order.
      if(subgraph->GetPrevSubgraph() != nullptr){
        CopyIntermediateDataIfNeeded(subgraph);
      }
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking subgraph " << subgraph->GetGraphid() << "\n";
        return kTfLiteError;
      }
      if(subgraph->GetNextSubgraph() == nullptr){
        PrintOutput(subgraph);
        if(!output_correct){
          std::cout << "OUTPUT WRONG!" << "\n";
          exit(-1);
          output_correct = false;
        }
      }
      subgraph_idx++;
      break;
    }
    case RuntimeState::BLOCKED_ : {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      break;
    }
    case RuntimeState::NEED_PROFILE :{
      // Need profile.
      if(RegisterModeltoScheduler() != kTfLiteOk){
        std::cout << "RegisterModeltoScheduler ERROR" << "\n";
        return kTfLiteError;
      }
      break;
    }
    case RuntimeState::SUBGRAPH_CREATE :{
      // Need subgraph partitioning.
      // this will delete the existing subgraphs and partition from original model.
      if(PartitionSubgraphs() != kTfLiteOk){
        std::cout << "PartitionSubgraphs ERROR" << "\n";
        return kTfLiteError;
      }
      subgraph_idx = 0;
      break;
    }
    default:
      break;
    }
  } // end of subgraph interation
  return kTfLiteOk;
}

// working function
TfLiteStatus TfLiteRuntime::InvokeSingleExecution() {
  if(state != RuntimeState::INVOKE_){
    std::cout << "ERROR cannot invoke runtime [" << runtime_id << "]\n";
    std::cout << "State is not INVOKE. cur state is " << state << "\n";
    return kTfLiteError;
  }
  Subgraph* subgraph;
  int subgraph_idx = 0;
  while(subgraph_idx < interpreter->subgraphs_size()){ // subgraph iteration
    subgraph = interpreter->subgraph(subgraph_idx);
    tf_packet tx_packet;
    memset(&tx_packet, 0, sizeof(tf_packet));
    tx_packet.runtime_id = runtime_id;
    tx_packet.runtime_current_state = state;

    if(subgraph != nullptr){
      if(subgraph->GetResourceType() == ResourceType::CPU)
        tx_packet.cur_graph_resource = 0;
      else if(subgraph->GetResourceType() == ResourceType::GPU)
        tx_packet.cur_graph_resource = 1;
      else // Subject to change. (impl CPUGPU Co-execution)
        tx_packet.cur_graph_resource = 0;
    }

    if(SendPacketToScheduler(tx_packet) != kTfLiteOk){ // Request invoke permission to scheduler
      return kTfLiteError;
    }
    tf_packet rx_packet;
    if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
      return kTfLiteError;
    }
    switch (rx_packet.runtime_next_state)
    {
    case RuntimeState::INVOKE_ :{
      // Invoke next subgraph in subgraph order.
      if(subgraph->GetPrevSubgraph() != nullptr){
        CopyIntermediateDataIfNeeded(subgraph);
      }
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking subgraph " << subgraph->GetGraphid() << "\n";
        return kTfLiteError;
      }
      if(subgraph->GetNextSubgraph() == nullptr){
        PrintOutput(subgraph);
        if(!output_correct){
          std::cout << "OUTPUT WRONG!" << "\n";
          exit(-1);
          output_correct = false;
        }
      }
      subgraph_idx++;
      break;
    }
    case RuntimeState::BLOCKED_ : {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      break;
    }
    case RuntimeState::NEED_PROFILE :{
      // Need profile.
      if(RegisterModeltoScheduler() != kTfLiteOk){
        std::cout << "RegisterModeltoScheduler ERROR" << "\n";
        return kTfLiteError;
      }
      break;
    }
    case RuntimeState::SUBGRAPH_CREATE :{
      // Need subgraph partitioning.
      // this will delete the existing subgraphs and partition from original model.
      if(PartitionSubgraphs() != kTfLiteOk){
        std::cout << "PartitionSubgraphs ERROR" << "\n";
        return kTfLiteError;
      }
      subgraph_idx = 0;
      break;
    }
    default:
      break;
    }
  } // end of subgraph interation
  return kTfLiteOk;
}


// Description below is conceptual. 
// Get dest subgraphs('D')'s next subgraph. (subgraph 'Dn')
// Compare input & output dims between 'Dn' and 'S'(source subgraph, probabily cpu side)
// Merge 'S' and 'D's output tensor to 'Dn's input tensor.
// NEED TO FIX FOR MULTIPLE OUTPUT TENSORS!!!!
// NEED TO FIX FOR MULTIPLE OUTPUT TENSORS!!!!
// NEED TO FIX FOR MULTIPLE OUTPUT TENSORS!!!!
void TfLiteRuntime::MergeCoExecutionData(Subgraph* min_precision_subgraph
                                      , Subgraph* max_precision_subgraph){
  Subgraph* dest_subgraph;
  dest_subgraph = max_precision_subgraph->GetNextSubgraph();
  if(dest_subgraph == nullptr){
    std::cout << "MergeCoExecutionData ERROR" << "\n";
    std::cout << "dest_subgraph nullptr." << "\n";
    return;
  }
  int dest_tensor_idx = dest_subgraph->GetInputTensorIndex();
  int min_precision_tensor_idx = min_precision_subgraph->GetOutputTensorIndex();
  int max_precision_tensor_idx = max_precision_subgraph->GetOutputTensorIndex();
  TfLiteTensor* dest_tensor = dest_subgraph->tensor(dest_tensor_idx);
  TfLiteTensor* min_precision_tensor = min_precision_subgraph->tensor(min_precision_tensor_idx);
  TfLiteTensor* max_precision_tensor = max_precision_subgraph->tensor(max_precision_tensor_idx);
  if(dest_tensor == nullptr || min_precision_tensor == nullptr ||
      max_precision_tensor == nullptr){
    std::cout << "MergeCoExecutionData ERROR" << "\n";
    std::cout << "Tensor NULLPTR" << "\n";
    return;
  }

  if(dest_tensor->dims->size < 4 || min_precision_tensor->dims->size < 4 ||
      max_precision_tensor->dims->size < 4){
    std::cout << "MergeCoExecutionData ERROR" << "\n";
    std::cout << "Tensor rank must 4" << "\n";
    return;
  }

  // check dims 
  if(dest_tensor->dims->data[1] == min_precision_tensor->dims->data[1]){
    // Merge CW-partitioned data 
    int dest_ch, min_tensor_ch, max_tensor_ch;
    dest_ch = dest_tensor->dims->data[3];
    min_tensor_ch = min_precision_tensor->dims->data[3];
    max_tensor_ch = max_precision_tensor->dims->data[3];
    if((min_tensor_ch + max_tensor_ch) != dest_ch){
      std::cout << "Tensor dim [OCH] min_prec + max_prec != dest ERROR" << "\n";
      return;
    }
    auto data_min = (float*)min_precision_tensor->data.data;
    auto data_max = (float*)max_precision_tensor->data.data;
    auto data_dest = (float*)dest_tensor->data.data;
    int tensor_data_size = 1;
    for(int i=0; i<dest_tensor->dims->size; ++i){
      tensor_data_size *= dest_tensor->dims->data[i];
    }
    int tensor_data_per_ch = tensor_data_size / dest_ch;
    for(int i=0; i<tensor_data_per_ch; ++i){ //copy maximum precision side data
      memcpy(data_dest + (min_tensor_ch * i), data_max + (max_tensor_ch * i),
              max_tensor_ch * sizeof(float));
    }
    for(int i=0; i<tensor_data_per_ch; ++i){ //copy minimum precision side data
      memcpy(data_dest + (max_tensor_ch * (i+1)), data_min + (min_tensor_ch * i),
              min_tensor_ch * sizeof(float));
    }
  }else{ // Merge HW-partitioned data
    int dest_ch = dest_tensor->dims->data[3];
    int dest_ht = dest_tensor->dims->data[1];
    int min_tensor_ht = min_precision_tensor->dims->data[1];
    int max_tensor_ht = max_precision_tensor->dims->data[1];
    int tensor_dest_data_size = 1;
    int min_precision_data_size = 1;
    int max_precision_data_size = 1;
    for(int i=0; i<dest_tensor->dims->size; ++i){
      tensor_dest_data_size *= dest_tensor->dims->data[i];
    }
    for(int i=0; i<max_precision_tensor->dims->size; ++i){
      max_precision_data_size *= max_precision_tensor->dims->data[i];
    }
    for(int i=0; i<min_precision_tensor->dims->size; ++i){
      min_precision_data_size *= min_precision_tensor->dims->data[i];
    }
    // Need to drop padding data before merge.
    // Drop minimum precision data because it is dequantized and might drop accuracy.
    if((min_tensor_ht + max_tensor_ht) != dest_ht){
      auto data_min = (float*)min_precision_tensor->data.data;
      auto data_max = (float*)max_precision_tensor->data.data;
      auto data_dest = (float*)dest_tensor->data.data;
      int drop_height = (dest_ht - (min_tensor_ht + max_tensor_ht)) * (-1/2);
      if(drop_height < 0){
        std::cout << "Wrong drop in HW merging ERROR " << "\n";
        return;
      }
      memcpy(data_dest, data_max, sizeof(float)*max_precision_data_size);
      memcpy(data_dest+max_precision_data_size, data_min, 
              sizeof(float)*(tensor_dest_data_size - min_precision_data_size));
    }else{  // No need to drop.
      auto data_min = (float*)min_precision_tensor->data.data;
      auto data_max = (float*)max_precision_tensor->data.data;
      auto data_dest = (float*)dest_tensor->data.data;

      memcpy(data_dest, data_max, sizeof(float)*max_precision_data_size);
      memcpy(data_dest+max_precision_data_size, data_min, 
              sizeof(float)*min_precision_data_size);
    }
  }
  // PrintTensorSerial(*source_tensor_cpu);
  // PrintTensorSerial(*source_tensor_gpu);
  // PrintTensorSerial(*dest_tensor);
  return; 
}

void TfLiteRuntime::CopyIntermediateDataIfNeeded(Subgraph* subgraph) {
  // use source_graph_id, dest_graph_id
  auto connect = [&](int source_subgraph, int dest_subgraph) {
    Subgraph* source_graph = interpreter->subgraph_id(source_subgraph);
    Subgraph* dest_graph = interpreter->subgraph_id(dest_subgraph);
    std::vector<int> dest_tensor_indices; 
    TfLiteIntArray* source_tensor_idx = source_graph->GetOutputTensorIndices();
    TfLiteIntArray* input_tensor_indices = dest_graph->GetInputTensorIndices();
    for(int i=0; i<input_tensor_indices->size; ++i){
      for(int j=0; j<source_tensor_idx->size; ++j){
        if(source_tensor_idx->data[j] == input_tensor_indices->data[i])
          dest_tensor_indices.push_back(input_tensor_indices->data[i]);
      }
    }
    if(dest_tensor_indices.empty()){
      std::cout << "Output tensor of subgraph [" << source_subgraph << "] cannot"
                << " found a matching input tensor in subgraph ["
                << dest_subgraph << "]\n";
      return kTfLiteError;
    }
    for(int i=0; i<dest_tensor_indices.size(); ++i){
      TfLiteTensor* source_tensor = source_graph->tensor(dest_tensor_indices[i]);
      TfLiteTensor* dest_tensor = dest_graph->tensor(dest_tensor_indices[i]);
      size_t source_byte_size = source_tensor->bytes;
      size_t dest_byte_size = dest_tensor->bytes;
      if (source_byte_size != dest_byte_size) {
        std::cout << "Source tensor[" << dest_tensor_indices[i] << "] size "
                  << static_cast<int>(source_byte_size) << " and Dest tensor["
                  << dest_tensor_indices[i] << "] size "
                  << static_cast<int>(dest_byte_size) << " missmatch!"
                  << "\n";
        return kTfLiteError;
      }
      // PrintTensorSerial(*dest_tensor);
      
      if(source_tensor->type == kTfLiteFloat32 && dest_tensor->type == kTfLiteFloat32){
        // auto data_source = (float*)source_tensor->data.data;
        // auto data_dest = (float*)dest_tensor->data.data;
        // memcpy(data_dest, data_source, source_byte_size);
        dest_tensor->data.data = source_tensor->data.data;
      }else if(source_tensor->type == kTfLiteInt8 && dest_tensor->type == kTfLiteInt8){
        // auto data_source = (int8_t*)source_tensor->data.data;
        // auto data_dest = (int8_t*)dest_tensor->data.data;
        // memcpy(data_dest, data_source, source_byte_size);
        dest_tensor->data.data = source_tensor->data.data;
      }
      std::cout << "Copied intermediate data" << "\n";
    }
    return kTfLiteOk;
  };
  Subgraph* prev_graph = subgraph->GetPrevSubgraph();
  if (prev_graph != nullptr) {  // Need to copy output from previous graph.
    int source_graph_id = prev_graph->GetGraphid();
    int dest_graph_id = subgraph->GetGraphid();
    if (connect(source_graph_id, dest_graph_id) != kTfLiteOk) {
      std::cout << "Subgraph intermediate data copy failed"
                << "\n";
      return;
    }
  } else {  // if nulltpr returned
    return;
  }
  return;
}

// SOURCE TENSOR AND DEST TENSOR SHOULD BE IN SAME PRECISION...
void TfLiteRuntime::CopyIntermediateDataIfNeeded(Subgraph* co_subgraph, Subgraph* subgraph) {
  // use source_graph_id, dest_graph_id
  auto connect = [&](Subgraph* source_subgraph, Subgraph* dest_subgraph) {
    std::vector<int> dest_tensor_indices; 
    std::vector<int> source_tensor_idx = source_subgraph->outputs();
    std::vector<int> input_tensor_indices = dest_subgraph->inputs();
    for(int i=0; i<input_tensor_indices.size(); ++i){
      for(int j=0; j<source_tensor_idx.size(); ++j){
        if(source_tensor_idx[j] == input_tensor_indices[i])
          dest_tensor_indices.push_back(input_tensor_indices[i]);
      }
    }
    if(dest_tensor_indices.empty()){
      std::cout << "Output tensor of subgraph [" << source_subgraph->GetGraphid() << "] cannot"
                << " found a matching input tensor in subgraph ["
                << dest_subgraph->GetGraphid() << "]\n";
      return kTfLiteError;
    }
    for(int i=0; i<dest_tensor_indices.size(); ++i){
      // std::cout << "dest_tensor_indices : " << dest_tensor_indices[i] << "\n";
      TfLiteTensor* source_tensor = source_subgraph->tensor(dest_tensor_indices[i]);
      TfLiteTensor* dest_tensor = dest_subgraph->tensor(dest_tensor_indices[i]);
      size_t source_byte_size = source_tensor->bytes;
      size_t dest_byte_size = dest_tensor->bytes;
      //int offset = (source_byte_size - dest_byte_size) / sizeof(source_tensor->type);
      // PrintTensorSerial(*dest_tensor);
      auto data_source = (float*)source_tensor->data.data;
      auto data_dest = (float*)dest_tensor->data.data;
      // std::cout << "source : " <<  source_byte_size << " dest " << dest_byte_size << "\n";
      dest_tensor->data.data = source_tensor->data.data;
      // memcpy(data_dest, data_source, dest_byte_size);
      std::cout << "Copied intermediate data" << "\n";
    }
    return kTfLiteOk;
  };
  Subgraph* prev_graph = subgraph->GetPrevSubgraph();
  if (prev_graph != nullptr) {  // Need to copy output from previous graph.
    if (connect(prev_graph, co_subgraph) != kTfLiteOk) {
      std::cout << "Subgraph intermediate data copy failed"
                << "\n";
      return;
    }
  } else {  // if nulltpr returned
    return;
  }
  return;
}

void TfLiteRuntime::PrintOutput(Subgraph* subgraph){
  int output_tensor_idx = subgraph->outputs()[0];
  TfLiteTensor* output_tensor = subgraph->tensor(output_tensor_idx);
  if(output_tensor != nullptr){
    PrintTensor(*output_tensor, true);
  }else{
    std::cout << "Worker : output tensor print ERROR" << "\n";
  }
  return;  
}

void TfLiteRuntime::PrintTensor(TfLiteTensor& tensor, bool is_output){
  // std::cout << "[Print Tensor]" << "\n";
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
  std::cout << " Nunber of ch : " << tensor_data_ch_size << "\n";
  // std::cout << " Tensor DATA " << "\n";
  if(tensor.type == TfLiteType::kTfLiteFloat32){
    auto data_st = (float*)tensor.data.data;
    for(int i=0; i<tensor_data_ch_size; i++){
      if(!is_output)
        std::cout << "CH [" << i << "] \n";
      for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
        float data = *(data_st+(i+j*tensor_data_ch_size));
        if(is_output){
          switch (interpreter->GetInputType())
          {
          case INPUT_TYPE::MNIST :
            if(data > 0.8){ // threshhold
              std::cout << "CH [" << i << "] ";
              printf("%s%0.6f%s \n", C_GREN, data, C_NRML);
              output_correct = true;
            }
            break;
          
          case INPUT_TYPE::IMAGENET224 :
            if(data > 8.0){ // threshhold
              std::cout << "CH [" << i << "] ";
              printf("%s%0.6f%s \n", C_GREN, data, C_NRML);
              output_correct = true;
            }
            break;

          case INPUT_TYPE::IMAGENET300 :
            if(data > 8.0){ // threshhold
              std::cout << "CH [" << i << "] ";
              printf("%s%0.6f%s \n", C_GREN, data, C_NRML);
              output_correct = true;
            }
            break;
          default:
            break;
          }
        }else{
          if (data == 0) {
            printf("%0.6f ", data);
          }
          else if (data != 0) {
            printf("%s%0.6f%s ", C_GREN, data, C_NRML);
          }
        }
        if (j % tensor_axis == tensor_axis-1) {
          printf("\n");
        }
      }
 //     std::cout << "\n";
    }
  }
}

void TfLiteRuntime::PrintTensorSerial(TfLiteTensor& tensor){
  std::cout << "[Print Tensor]" << "\n";
  int tensor_channel_idx = tensor.dims->size-1;
  int tensor_data_ch_size = tensor.dims->data[tensor_channel_idx];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< tensor.dims->size; i++){
    if(i == 2){
      tensor_axis = tensor.dims->data[i];
    }
    tensor_data_size *= tensor.dims->data[i]; 
  }
  std::cout << " Number of data : " << tensor_data_size << "\n";
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

void TfLiteRuntime::PrintyoloOutput(TfLiteTensor& tensor){
  std::cout << "[Print Yolo output Tensor]" << "\n";
  int tensor_channel_idx = tensor.dims->size-1;
  int tensor_data_ch_size = tensor.dims->data[tensor_channel_idx];
  int tensor_data_size = 1;
  int tensor_axis;
  int ch = (tensor.dims->data[1] * tensor.dims->data[2]);
  int data_size = tensor.dims->data[3];

  std::cout << " Number of data : " << tensor_data_size << "\n";
  std::cout << " Tensor DATA " << "\n";
  if(tensor.type == TfLiteType::kTfLiteFloat32){
    std::cout << "[FLOAT32 TENSOR]" << "\n";
    auto data_st = (float*)tensor.data.data;
    for(int i=0; i<ch; i++){
      std::cout << "CH [" << i << "] \n";
      for(int j=0; j<data_size; j++){
        float data = *(data_st+ j + (data_size * i));
        if (data == 0) {
          printf("%0.6f", data);
        }
        else if (data != 0) {
            printf("%s%0.6f%s", C_GREN, data, C_NRML);
        }
        printf(",");
      }
      std::cout << "\n";
    }
  }
}

std::vector<std::vector<float>*>* TfLiteRuntime::GetFloatOutputInVector(){
  std::vector<std::vector<float>*>* output = new std::vector<std::vector<float>*>;
  if(global_output_tensor == nullptr){
    std::cout << "No output tensor to parse " << "\n";
    return output;
  }
  TfLiteTensor* tensor = global_output_tensor;
  if(tensor->dims->size == 2){
    int tensor_channel_idx = tensor->dims->size-1;
    int tensor_data_ch_size = tensor->dims->data[tensor_channel_idx];
    int tensor_data_size = 1;
    int tensor_axis;
    int ch = tensor->dims->data[1];
    auto data_st = (float*)tensor->data.data;
    output->push_back(new std::vector<float>());
    for(int j=0; j<ch; j++){
      float data = *(data_st + j);
      output->at(0)->push_back(data);
    }
  }else if(tensor->dims->size == 4){
    int tensor_channel_idx = tensor->dims->size-1;
    int tensor_data_ch_size = tensor->dims->data[tensor_channel_idx];
    int tensor_data_size = 1;
    int tensor_axis;
    int ch = (tensor->dims->data[1] * tensor->dims->data[2]);
    int data_size = tensor->dims->data[3];
    auto data_st = (float*)tensor->data.data;
    for(int i=0; i<ch; i++){
      output->push_back(new std::vector<float>());
      for(int j=0; j<data_size; j++){
        float data = *(data_st+ j + (data_size * i));
        output->at(i)->push_back(data);
      }
    }
  }
  return output;
}

}  // namespace tflite