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
  std::cout << "Runtime :Send packet to scheduler" << "\n";
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
  PrintInterpreterStateV3(interpreter);
  std::cout << "============================" << "\n";
  std::cout << "Minimal precision interpreter" << "\n";
  PrintInterpreterStateV3(quantized_interpreter);
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
  interpreter->PrintSubgraphInfo();
  quantized_interpreter->PrintSubgraphInfo();
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::PartitionSubgraphs(){
  std::vector<std::vector<int>> raw_plan;
  for(int i=0; i<TF_P_PLAN_LENGTH; ++i){
    while(partitioning_plan[i][TF_P_IDX_START] != TF_P_END_MASTER){
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
    std::cout << "Runtime : CopyRawPartitioningPlan" << "\n";
    raw_plan.clear();
  } 
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
  PrintInterpreterStateV3(interpreter);
  std::cout << "Successfully partitioned subgraph" << "\n";
  std::cout << "Ready to invoke" << "\n";
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::PartitionCoSubgraphs(){
  std::vector<std::vector<int>> raw_plan;
  int inner_plan_idx = 0;

  for(int i=0; i<TF_P_PLAN_LENGTH; ++i){
    raw_plan.push_back(std::vector<int>());
    inner_plan_idx = raw_plan.size() - 1;
    if(partitioning_plan[i][TF_P_IDX_START] == TF_P_END_PLAN){
      raw_plan[inner_plan_idx].push_back(TF_P_END_PLAN);
      interpreter_builder->CopyRawPartitioningPlan(raw_plan);
      quantized_builder->CopyRawPartitioningPlan(raw_plan);
      raw_plan.clear();
      inner_plan_idx = 0;
      raw_plan.push_back(std::vector<int>());
      i++;
    }
    if(partitioning_plan[i][TF_P_IDX_START] == TF_P_END_MASTER)
      break;
    for(int j=0; j<TF_P_PLAN_SIZE; ++j){ // third idx means processor.
      raw_plan[inner_plan_idx].push_back(partitioning_plan[i][j]);
    }
  }

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
  
  // Fill the subgraph id section in tf_packet so that the scheduler can
  // see the total graph of view.
  std::vector<int> full_prec_subgraphs, min_prec_subgraphs;
  interpreter->GetTotalSubgraphID(full_prec_subgraphs);
  quantized_interpreter->GetTotalSubgraphID(min_prec_subgraphs);
  
  std::cout << "full_prec_subgraphs.size() : " << full_prec_subgraphs.size() <<"\n";
  for(int i=0; i<full_prec_subgraphs.size(); ++i){
    std::cout << "id : " << full_prec_subgraphs[i] << " ";
    tx_packet.subgraph_ids[0][i] = full_prec_subgraphs[i];
  }
  tx_packet.subgraph_ids[0][full_prec_subgraphs.size()] = -1;
  std::cout << "\n";
  std::cout << "min_prec_subgraphs.size() : " << min_prec_subgraphs.size() <<"\n";
  for(int i=0; i<min_prec_subgraphs.size(); ++i){
    std::cout << "id : " << min_prec_subgraphs[i] << " ";
    tx_packet.subgraph_ids[1][i] = min_prec_subgraphs[i];
  }
  tx_packet.subgraph_ids[1][min_prec_subgraphs.size()] = -1;
  std::cout << "\n";

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
  std::cout << "MAX precicion interpreter state" << "\n";
  // PrintInterpreterStateV3(interpreter);
  std::cout << "=====================" << "\n";
  std::cout << "MIN precicion interpreter state" << "\n";
  // PrintInterpreterStateV3(quantized_interpreter);
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
                                     cv::Mat& input, cv::Mat& input_quant,
                                     INPUT_TYPE input_type) {
  // for (int i = 0; i < 224; ++i) {
  //   for (int j = 0; j < 224; ++j) {
  //     printf("%d ", ((uint8_t)input.at<uchar>(i,j)));
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
  if(input_tensor->type == kTfLiteFloat32){
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
              h * w * input.elemSize());
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
        memcpy(input_pointer, input.data,
              h * w * input.elemSize());
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
      case INPUT_TYPE::LANENET144800:
        memcpy(input_pointer, input.data,
              h * w * input.elemSize());      
      default:
        break;
      }
  }else if(input_tensor->type == kTfLiteUInt8){
    auto input_pointer = (uint8_t*)input_tensor->data.data;
    int h = input_tensor->dims->data[1];
    int w = input_tensor->dims->data[2];
    switch (input_type) {
      case INPUT_TYPE::MNIST:
        for (int i = 0; i < h; ++i) {
          for (int j = 0; j < w; ++j) {
            input_pointer[i * w + j] = ((float)input_quant.at<uchar>(i, j) / 255.0);
          }
        }
        break;
      case INPUT_TYPE::IMAGENET224:
        memcpy(input_pointer, input_quant.data,
              h * w * input_quant.elemSize());
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
        memcpy(input_pointer, input_quant.data,
              h * w * input_quant.elemSize());
        break;
      case INPUT_TYPE::IMAGENET416:
        memcpy(input_pointer, input_quant.data,
              input_quant.total() * input_quant.elemSize());
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
      case INPUT_TYPE::LANENET144800:
        memcpy(input_pointer, input_quant.data,
              h * w * input_quant.elemSize()); 
      default:
        break;
    }
  }
  // PrintTensorSerial(*input_tensor);
  if(use_two_interpreter){
    int input_tensor_idx = quantized_interpreter->subgraph_id(1)->GetInputTensorIndex();
    // std::cout << "min_prec input_tensor_idx : " << input_tensor_idx << "\n";
    quant_input_tensor = quantized_interpreter->subgraph_id(1)->tensor(input_tensor_idx);
    if(quant_input_tensor != nullptr){
      if(quant_input_tensor->type == kTfLiteFloat32){
        auto input_pointer = (float*)quant_input_tensor->data.data;
        int h = quant_input_tensor->dims->data[1];
        int w = quant_input_tensor->dims->data[2];
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
                  h * w * input.elemSize());
            break;
          case INPUT_TYPE::IMAGENET300:
            memcpy(input_pointer, input.data,
                  h * w * input.elemSize());
            break;
          case INPUT_TYPE::IMAGENET416:
            memcpy(input_pointer, input.data,
                  input.total() * input.elemSize());
          case INPUT_TYPE::LANENET144800:
            memcpy(input_pointer, input.data,
                  input.total() * input.elemSize());
            break;
          default:
            break;
          }
      }else if(quant_input_tensor->type == kTfLiteUInt8){
        auto input_pointer = (uint8_t*)quant_input_tensor->data.data;
        int h = quant_input_tensor->dims->data[1];
        int w = quant_input_tensor->dims->data[2];
        switch (input_type) {
          case INPUT_TYPE::MNIST:
            for (int i = 0; i < h; ++i) {
              for (int j = 0; j < w; ++j) {
                input_pointer[i * w + j] = ((float)input_quant.at<uchar>(i, j) / 255.0);
              }
            }
            break;
          case INPUT_TYPE::IMAGENET224:
            memcpy(input_pointer, input_quant.data,
                  h * w * input_quant.elemSize());
            break;
          case INPUT_TYPE::IMAGENET300:
            memcpy(input_pointer, input_quant.data,
                  h * w * input_quant.elemSize());
            break;
          case INPUT_TYPE::IMAGENET416:
            memcpy(input_pointer, input_quant.data,
                  input_quant.total() * input_quant.elemSize());
            break;
          case INPUT_TYPE::LANENET144800:
            memcpy(input_pointer, input_quant.data,
                  input_quant.total() * input_quant.elemSize());
          default:
            break;
        }
      }
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

////////////////////////////////////////////////////////////////////////////////
// Working function
// TODO : 1. Get subgraph id to invoke from Graphselector.
//        2. Get system monitoring info from scheduler at each invoke.
////////////////////////////////////////////////////////////////////////////////
// void TfLiteRuntime::DebugSyncInvoke(PrecisionType type){
//   // For prototye, invoke first layer only with HW-partitioning and merge them.
//   Subgraph* subgraph; 
//   int subgraph_idx = 0; 
//   double response_time = 0; 
//   std::vector<double> latency; 
//   struct timespec begin, end; 
//   while(true){
//     if(type == PrecisionType::MINIMAL_PRECISION){
//       if(quantized_interpreter->subgraphs_size() < 1){
//         // std::cout << "No invokable subgraph for cpu" << "\n";
//         break;
//       }
//       // sync with gpu here (notified by gpu)
//       std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
//       invoke_sync_cv.wait(lock_invoke, [&]{ return invoke_cpu; });
//       invoke_cpu = false;
//       subgraph = quantized_interpreter->subgraph(subgraph_idx);
//       if(main_execution_graph != nullptr)
//         CopyIntermediateDataIfNeeded(subgraph, main_execution_graph);
//       // std::cout << "[Minimal precision] Invoke subgraph " << subgraph->GetGraphid() << "\n";
//       clock_gettime(CLOCK_MONOTONIC, &begin);
//       if(subgraph->Invoke() != kTfLiteOk){
//         std::cout << "ERROR on invoking CPU subgraph " << subgraph->GetGraphid() << "\n";
//         return;
//       }
//       clock_gettime(CLOCK_MONOTONIC, &end);
//       response_time =  (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
//       latency.push_back(response_time);
//       // sync with gpu here (wake gpu)
//       std::unique_lock<std::mutex> lock_data(data_sync_mtx);
//       is_execution_done = true;
//       co_execution_graph = subgraph;
//       data_sync_cv.notify_one();
//       if(subgraph->GetNextSubgraph() != nullptr)
//         subgraph_idx++;
//       else{
//         WriteVectorLog(latency, 1);
//         // std::cout << "Minimal precision graph invoke done" << "\n";
//         break;
//       }
//     }else if(type == PrecisionType::MAX_PRECISION){
//       subgraph = interpreter->subgraph(subgraph_idx);
//       if(subgraph->GetResourceType() == CO_GPU){
//         // wake cpu thread here
//         std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
//         invoke_cpu = true;
//         if(subgraph->GetPrevSubgraph() != nullptr)
//           main_execution_graph = subgraph;
//         invoke_sync_cv.notify_one();
//       }else{ // if not co-execution, it needs additional imtermediate data copy.
//         if(subgraph->GetPrevSubgraph() != nullptr &&
//             subgraph->GetPrevSubgraph()->GetResourceType() != ResourceType::CO_GPU){
//           CopyIntermediateDataIfNeeded(subgraph);
//         }
//       }
//       // std::cout << "[Max precision] Invoke subgraph " << subgraph->GetGraphid() << "\n";
//       clock_gettime(CLOCK_MONOTONIC, &begin);
//       if(subgraph->Invoke() != kTfLiteOk){
//         std::cout << "ERROR on invoking subgraph id " << subgraph->GetGraphid() << "\n";
//         return;
//       }
//       clock_gettime(CLOCK_MONOTONIC, &end);
//       response_time =  (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
//       latency.push_back(response_time);
//       if(subgraph->GetResourceType() == ResourceType::CO_GPU){
//         // sync with cpu here
//         std::unique_lock<std::mutex> lock_data(data_sync_mtx);
//         data_sync_cv.wait(lock_data, [&]{ return is_execution_done; });
//         is_execution_done = false;
//         // data merge here
//         if(co_execution_graph != nullptr){
//           MergeCoExecutionData(co_execution_graph, subgraph);
//           co_execution_graph = nullptr;
//           // int input_tensor = subgraph->GetNextSubgraph()->GetInputTensorIndex();
//           // PrintTensorSerial(*(subgraph->GetNextSubgraph()->tensor(input_tensor)));
//         }
//       }
//       if(subgraph->GetNextSubgraph() != nullptr){
//         subgraph_idx++;
//       }
//       else{
//         main_execution_graph = nullptr;
//         WriteVectorLog(latency, 0);
//         // std::cout << "Max precision graph invoke done" << "\n";
//         // PrintyoloOutput(*(subgraph->tensor(109)));
//         global_output_tensor = subgraph->tensor(subgraph->GetFirstOutputTensorIndex());
//         break;
//       }
//     }
//   }
//   return;
// }
//////////////////////////////////////////////////////////////////////

// Before modification (original one)
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
        if(subgraph->GetPrevSubgraph() != nullptr)
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
        main_execution_graph = nullptr;
        WriteVectorLog(latency, 0);
        // std::cout << "Max precision graph invoke done" << "\n";
        // PrintyoloOutput(*(subgraph->tensor(109)));
        global_output_tensor = subgraph->tensor(subgraph->GetFirstOutputTensorIndex());
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
  // std::cout << "Merge" << "\n";
  Subgraph* dest_subgraph;
  dest_subgraph = max_precision_subgraph->GetNextSubgraph();
  if(dest_subgraph == nullptr){
    std::cout << "MergeCoExecutionData ERROR" << "\n";
    std::cout << "dest_subgraph nullptr." << "\n";
    return;
  }
  std::vector<int> dest_tensor_indicies = 
                              dest_subgraph->inputs();

  int dest_tensor_idx = dest_subgraph->GetInputTensorIndex();
  int min_precision_tensor_idx = min_precision_subgraph->GetFirstOutputTensorIndex(); // for uint model
  int max_precision_tensor_idx = max_precision_subgraph->GetFirstOutputTensorIndex();
  int dequant_reference_tensor_idx = 0;
  
  // std::cout << "min id : " << min_precision_subgraph->GetGraphid() <<
  //               " max id : " << max_precision_subgraph->GetGraphid() << "\n";
  // std::cout << "Merge two tensors, " << min_precision_tensor_idx << " "
  //             << max_precision_tensor_idx << " to " << dest_tensor_idx << "\n";
  TfLiteTensor* dest_tensor = dest_subgraph->tensor(dest_tensor_idx);
  TfLiteTensor* min_precision_tensor = 
                  min_precision_subgraph->tensor(min_precision_tensor_idx);
  TfLiteTensor* max_precision_tensor = 
                  max_precision_subgraph->tensor(max_precision_tensor_idx);
             
  int quantization_param_tensor_idx =  
                  min_precision_subgraph->GetFirstInputTensorIndex();
  TfLiteTensor* dequant_reference_tensor = nullptr;
  if(min_precision_subgraph->tensor(quantization_param_tensor_idx)->quantization.type
    == kTfLiteAffineQuantization){
    dequant_reference_tensor = min_precision_subgraph->tensor(quantization_param_tensor_idx);
  }else{
    quantization_param_tensor_idx= max_precision_subgraph->GetFirstInputTensorIndex();
    dequant_reference_tensor = max_precision_subgraph->tensor(quantization_param_tensor_idx); 
  }
            
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
  // This flag means that the tensor is de-quantized temporary for merge.
  // Restore to the original buffer after merge.
  float* dequantized_buffer = nullptr;

  if(min_precision_tensor->type == kTfLiteUInt8 || 
      min_precision_tensor->type == kTfLiteInt8 ){
    dequantized_buffer = (float*)DequantizeGivenTensorWithReference(
                                  min_precision_tensor, 
                                  dequant_reference_tensor);
    if(dequantized_buffer == nullptr){
      std::cout << "DeQuantizeGivenTensor returned nullptr ERROR" << "\n";
      return;
    }
  }
  // minimum precision side data buffer.
  float* data_min; 
  if(dequantized_buffer != nullptr){
    data_min = dequantized_buffer;
  }else{
    data_min = (float*)min_precision_tensor->data.data;
  }
  // max precision side data buffer.
  float* data_max = (float*)max_precision_tensor->data.data;
  
  // destination buffer.
  float* data_dest = (float*)dest_tensor->data.data;

  // check dims 
  if(dest_tensor->dims->data[3] == 
      min_precision_tensor->dims->data[3] + max_precision_tensor->dims->data[3]){
    // Merge CW-partitioned data 
    int dest_ch, min_tensor_ch, max_tensor_ch;
    dest_ch = dest_tensor->dims->data[3];
    min_tensor_ch = min_precision_tensor->dims->data[3];
    max_tensor_ch = max_precision_tensor->dims->data[3];
    // std::cout << "min_tensor_ch " << min_tensor_ch << " max_tensor_ch " << max_tensor_ch << "\n";
    if((min_tensor_ch + max_tensor_ch) != dest_ch){
      std::cout << "Tensor dim [OCH] min_prec + max_prec != dest ERROR" << "\n";
      return;
    }
    int tensor_data_size = 1;
    for(int i=0; i<dest_tensor->dims->size; ++i){
      tensor_data_size *= dest_tensor->dims->data[i];
    }
    // WRONG MERGE
    // WRONG MERGE
    // WRONG MERGE
    // Note : max pricision side is front channel.
    int tensor_data_per_ch = tensor_data_size / dest_ch;
    for(int i=0; i<tensor_data_per_ch; ++i){ //copy minimum precision side data
      memcpy(data_dest + (dest_ch * i), data_max + (max_tensor_ch * i),
              max_tensor_ch * sizeof(float));
    }
    
    for(int i=0; i<tensor_data_per_ch; ++i){ //copy minimum precision side data
      memcpy(data_dest + max_tensor_ch + (dest_ch * i), data_min + (min_tensor_ch * i),
              min_tensor_ch * sizeof(float));
    }
    
  }else{ // Merge HW-partitioned data
    int dest_ht = dest_tensor->dims->data[1];
    int min_tensor_ht = min_precision_tensor->dims->data[1];
    int max_tensor_ht = max_precision_tensor->dims->data[1];
    int tensor_dest_data_size = 1;
    int min_precision_data_size = 1;
    int max_precision_data_size = 1;

    // Calculate data size to copy.
    for(int i=0; i<dest_tensor->dims->size; ++i){
      tensor_dest_data_size *= dest_tensor->dims->data[i];
    }
    for(int i=0; i<max_precision_tensor->dims->size; ++i){
      max_precision_data_size *= max_precision_tensor->dims->data[i];
    }
    for(int i=0; i<min_precision_tensor->dims->size; ++i){
      min_precision_data_size *= min_precision_tensor->dims->data[i];
    }
    // Need to drop padding data before merge if it's not fit with destination tensor.
    // Drop minimum precision data because it is dequantized and might drop accuracy.
    if((min_tensor_ht + max_tensor_ht) != dest_ht){
      float drop_height = (dest_ht - (min_tensor_ht + max_tensor_ht))/(-2);
      if(drop_height < 0){
        std::cout << "Wrong drop in HW merging ERROR on graph" << "\n";
        std::cout << "min sub: " << min_precision_subgraph->GetGraphid() <<
              " h: " << min_tensor_ht << 
              " max sub: " << max_precision_subgraph->GetGraphid() << 
              " h: " << max_tensor_ht << " dest: " << dest_ht << "\n";
        return;
      }
      memcpy(data_dest, data_max, sizeof(float)*max_precision_data_size);
      memcpy(data_dest+max_precision_data_size, data_min, 
              sizeof(float)*(tensor_dest_data_size - min_precision_data_size));
    }else{  // No need to drop (fits with destination tensor).
      memcpy(data_dest, data_max, sizeof(float)*max_precision_data_size);
      memcpy(data_dest+max_precision_data_size, data_min, 
              sizeof(float)*min_precision_data_size);
    }
  }
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
      // std::cout << "Copied intermediate data" << "\n";
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

// SOURCE TENSOR AND DEST TENSOR MUST BE IN SAME PRECISION...
// TODO : Quantization, height partitioning aware data copy
// IMPORTANT : There are some cases that input, output tensor indices are not same.
//             So, be aware to use this function in those cases.
void TfLiteRuntime::CopyIntermediateDataIfNeeded(Subgraph* min_precision_subgraph_
                                              , Subgraph* max_precision_subgraph_) {
  auto connect = [&](Subgraph* source_subgraph, Subgraph* dest_subgraph) {
    int source_tensor_idx = source_subgraph->inputs()[0];
    int input_tensor_idx = dest_subgraph->GetFirstInputTensorIndex();
    TfLiteTensor* source_tensor = source_subgraph->tensor(source_tensor_idx);
    TfLiteTensor* dest_tensor = dest_subgraph->tensor(input_tensor_idx);
    // std::cout << "source_tensor : " << source_tensor_idx << "\n";
    // std::cout << "dest_tensor : " << input_tensor_idx << "\n";
    void* data_source = nullptr;
    int source_data_size = 1;
    int dest_data_size = 1;
    for(int i=0; i<source_tensor->dims->size; ++i){
      source_data_size *= source_tensor->dims->data[i]; 
    }
    for(int i=0; i<dest_tensor->dims->size; ++i){
      dest_data_size *= dest_tensor->dims->data[i]; 
    }
    // Match tensor precision (quantize)
    if(source_tensor->type == kTfLiteFloat32 &&
          dest_tensor->type == kTfLiteUInt8){
      // std::cout << "quant int" << "\n";
      auto data_dest = (uint8_t*)dest_tensor->data.data;
      TfLiteAffineQuantization* new_quantization_params = new TfLiteAffineQuantization;
      data_source = QuantizeGivenTensorandReturnBuffer(source_tensor,
                                                       new_quantization_params);
      int offset = source_data_size - dest_data_size;
      memcpy(data_dest, data_source + offset, dest_data_size*(sizeof(uint8_t)));
      free(data_source);           
      // std::cout << "Copied intermediate data from main graph" << "\n";
      dest_tensor->quantization.params = reinterpret_cast<void*>(new_quantization_params);
      // printf("%p \n", new_quantization_params);
      // std::cout << "Asdfa" << "\n";
      dest_tensor->quantization.type = kTfLiteAffineQuantization;
      // printf("set scale %.6f, zp %d \n",
      //   new_quantization_params->scale->data[0],
      //   new_quantization_params->zero_point->data[0]);
    }else{ 
      // std::cout << "no quant int" << "\n";
      // Maybe consider memory footprint.
      auto data_dest = (float*)dest_tensor->data.data;
      auto data_source = (float*)source_tensor->data.data;
      int offset = source_data_size - dest_data_size;
      memcpy(data_dest, data_source + offset, dest_data_size*(sizeof(float)));
      // std::cout << "Copied intermediate data from main graph" << "\n";
    }  
    return kTfLiteOk;
  };
  // std::cout << "Asdf" << "\n";
  if (max_precision_subgraph_ != nullptr) {  // Need to copy output from previous graph.
    if (connect(max_precision_subgraph_, min_precision_subgraph_) != kTfLiteOk) {
      std::cout << "Subgraph intermediate data copy failed"
                << "\n";
      return;
    }
    // std::cout << "Asdf" << "\n";
  } else {  // if nulltpr returned
    return;
  }
  return;
}

// param : n_batch must be 1.
void TfLiteRuntime::QuantizeFloats(const float* float_data_ptr, int n_batch,
                              int n_data, int8_t* quantized_data_ptr,
                              float* scaling_factors, int32_t* zero_points,
                              bool do_asymmetric) {
  for (int b = 0; b < n_batch; ++b) {
    const int offset = b * n_data;
    if (do_asymmetric) {
      std::cout << "Asymmetric Quantization Not Implemented \n";
    } 
    else {
      float unused_min, unused_max;
      QuantizeSymFloats(
          float_data_ptr + offset, n_data, quantized_data_ptr + offset,
          &unused_min, &unused_max, scaling_factors, zero_points);
    }
  }
}

void TfLiteRuntime::QuantizeSymFloats(const float* values, const int size,
                                int8_t* quantized_values, float* min_value,
              float* max_value, float* scaling_factor, int32_t* zero_points){
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;
  const int zero_point = 
    std::round(((*max_value * 0) - (*min_value * 255))/(*max_value - *min_value));
  QuantizeSymFloatsMain(values, size, quantized_values, *min_value,
                            *max_value, scaling_factor, zero_points);
}

// Minsung
// Fixed for uint8 quantization.
void TfLiteRuntime::QuantizeSymFloatsMain(const float* values, const int size,
                                    int8_t* quantized_values, float min_value,
                float max_value, float* scaling_factor, int32_t * zero_points){
  const int32_t kScale = 255;
  const float range = max_value - min_value;
  if (range == 0) { //means given array is zero
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  //zero_points =  
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(TfLiteRound(values[i] * *scaling_factor) + *zero_points);
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = static_cast<int8_t>(
        std::min(kScale, std::max(0, quantized_value)));
  }
}

TfLiteAffineQuantization* TfLiteRuntime::CalcQuantizationParamsFromTensor(
                                                    TfLiteTensor* tensor){
  TfLiteAffineQuantization* new_params = new TfLiteAffineQuantization;
  const int32_t kScale = 255; // case of uint8
  int tensor_data_size = 1;
  for(int i=0; i<tensor->dims->size; ++i){
    tensor_data_size *= tensor->dims->data[i]; 
  }
  auto values = (float*)tensor->data.data;
  auto minmax = std::minmax_element(values, values + tensor_data_size);
  float min_value = *minmax.first;
  float max_value = *minmax.second;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  const float scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  const int zero_point = std::round(((max_value * 0) - (min_value * 255))/(max_value - min_value));
  new_params->quantized_dimension = 1;
  TfLiteFloatArray* new_scale_ary = TfLiteFloatArrayCreate(1);
  TfLiteIntArray* new_zero_point_ary = TfLiteIntArrayCreate(1);
  new_scale_ary->data[0] = scaling_factor;
  new_zero_point_ary->data[0] = zero_point;
  new_params->scale = new_scale_ary;
  new_params->zero_point = new_zero_point_ary;
  return new_params;
}

TfLiteStatus TfLiteRuntime::QuantizeGivenTensor(TfLiteTensor* tensor){
  TfLiteTensor* working_tensor = tensor;
  working_tensor->allocation_type = kTfLiteDynamic;
  int tensor_data_dims_size = working_tensor->dims->size-1; 
  int tensor_data_ch_size = working_tensor->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i<working_tensor->dims->size; ++i){
    tensor_data_size *= working_tensor->dims->data[i]; 
  }
  int8_t* quantized_values = (int8_t*)malloc(tensor_data_size);
  auto data_st_origin_float = (float*)working_tensor->data.data;
  float* scaling_factors = new float;
  int32_t* zero_points = new int32_t;
  QuantizeFloats(data_st_origin_float, 1, tensor_data_size, quantized_values,
                          scaling_factors, zero_points, false);
  working_tensor->type = TfLiteType::kTfLiteInt8;
  working_tensor->data.data = quantized_values;
  working_tensor->bytes = tensor_data_size;
  // TODO : Change TfLiteQuantizationParams to TfLiteAffineQuantization later.
  TfLiteQuantizationParams* quant_params = new TfLiteQuantizationParams;
  quant_params->scale = *scaling_factors;
  quant_params->zero_point = *zero_points;
  working_tensor->params.scale = *scaling_factors;
  working_tensor->params.zero_point = *zero_points;
  working_tensor->quantization.params = &quant_params;
  working_tensor->quantization.type = TfLiteQuantizationType::kTfLiteAffineQuantization;
  //PrintTensor(*working_tensor, UnitType::CPU0);
  return kTfLiteOk;
}

void* TfLiteRuntime::QuantizeGivenTensorandReturnBuffer(TfLiteTensor* tensor,
                                      TfLiteAffineQuantization* quant_params){
  TfLiteTensor* working_tensor = tensor;
  working_tensor->allocation_type = kTfLiteDynamic;
  int tensor_data_dims_size = working_tensor->dims->size-1; 
  int tensor_data_ch_size = working_tensor->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i<working_tensor->dims->size; ++i){
    tensor_data_size *= working_tensor->dims->data[i]; 
  }
  int8_t* quantized_values = (int8_t*)malloc(tensor_data_size);
  auto data_st_origin_float = (float*)working_tensor->data.data;
  float scaling_factor = 0;
  int32_t zero_point = 0;
  QuantizeFloats(data_st_origin_float, 1, tensor_data_size, quantized_values,
                          &scaling_factor, &zero_point, false);

  quant_params->scale = TfLiteFloatArrayCreate(1);
  quant_params->zero_point = TfLiteIntArrayCreate(1);
  quant_params->scale->data[0] = scaling_factor;
  quant_params->zero_point->data[0] = zero_point;
  quant_params->quantized_dimension = 1;
  // printf("set scale %.6f, zp %d \n",
  //   quant_params->scale->data[0], quant_params->zero_point->data[0]);
  // printf("%p \n", quant_params);
  return quantized_values;
}

void* TfLiteRuntime::DequantizeGivenTensor(TfLiteTensor* tensor){
  TfLiteTensor* working_tensor = tensor;
  if(working_tensor->quantization.type != kTfLiteAffineQuantization &&\
      working_tensor->type != kTfLiteInt8){
    std::cout << "Dequantization Tensor Type Error \n";
    return nullptr;
  }
  //if(working_tensor->allocation_type != kTfLiteDynamic || 
  //  working_tensor->quantization.type != kTfLiteAffineQuantization){
  //  return kTfLiteError;
  //}
  int tensor_data_dims_size = working_tensor->dims->size-1; 
  int tensor_data_ch_size = working_tensor->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i<working_tensor->dims->size; i++){
    tensor_data_size *= working_tensor->dims->data[i]; 
  }
  auto data_st_origin = (uint8_t*)tensor->data.data;
  auto dequantized_values = (float*)malloc(tensor_data_size * sizeof(float));
  int quantized_dimension = 
        ((TfLiteAffineQuantization *)(working_tensor->quantization.params))->quantized_dimension;
  TfLiteFloatArray* scaling_factor = 
        ((TfLiteAffineQuantization *)(working_tensor->quantization.params))->scale;
  TfLiteIntArray* zero_point = 
        ((TfLiteAffineQuantization *)(working_tensor->quantization.params))->zero_point;
  std::vector<float> scaling_factors;
  std::vector<int> zero_points;
  for(int i=0; i<scaling_factor->size; ++i){
    scaling_factors.push_back(scaling_factor->data[i]);
  }
  for(int i=0; i<zero_point->size; ++i){
    zero_points.push_back(zero_point->data[i]);
  }
  // printf("quantized_dimension : %d \n", quantized_dimension);
  // printf("scaling factor size: %d \n", scaling_factors.size());
  // printf("zero point size: %d \n", zero_points.size());
  // std::cout << "tensor data byte : " << working_tensor->bytes << "\n";
  // std::cout << "tensor data size : " << tensor_data_size << "\n";
  for(int i=0; i<tensor_data_size; ++i){
    float temp = (static_cast<int>(data_st_origin[i]) - zero_points[0]) * scaling_factors[0];
    dequantized_values[i] = temp;
  }
  // std::cout << "Dequnatize Done\n";
  // and return the new buffer to restore
  return dequantized_values;
}

// !!!! AFTER DEQUANTIZE, RESTORE THE ORIGINAL BUFFER
void* TfLiteRuntime::DequantizeGivenTensorWithReference(
                      TfLiteTensor* tensor, TfLiteTensor* ref_tensor){
  TfLiteTensor* working_tensor = tensor;
  if(working_tensor->quantization.type != kTfLiteAffineQuantization &&\
      working_tensor->type != kTfLiteInt8){
    std::cout << "Dequantization Tensor Type ERROR" << "\n";
    return nullptr;
  }
  if(ref_tensor == nullptr){
    std::cout << "Got reference tensor nullptr ERROR" << "\n";
    return nullptr;
  }
  int tensor_data_dims_size = working_tensor->dims->size-1; 
  int tensor_data_ch_size = working_tensor->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i<working_tensor->dims->size; i++){
    tensor_data_size *= working_tensor->dims->data[i]; 
  }
  auto data_st_origin = (uint8_t*)tensor->data.data;
  auto dequantized_values = (float*)malloc(tensor_data_size * sizeof(float));
  if(ref_tensor->quantization.params != nullptr){
    TfLiteAffineQuantization* params =
      (TfLiteAffineQuantization*)ref_tensor->quantization.params;
    float scale = params->scale->data[0];
    int zero_point = params->zero_point->data[0];
    if(zero_point == 128)
      zero_point = 0;
    for(int i=0; i<tensor_data_size; ++i){
      float temp = (static_cast<int>(data_st_origin[i]) - zero_point) * scale;
      dequantized_values[i] = temp;
    }
  }else{
    TfLiteAffineQuantization* params = CalcQuantizationParamsFromTensor(ref_tensor);
    int quantized_dimension = params->quantized_dimension;
    TfLiteFloatArray* scaling_factor = params->scale;
    TfLiteIntArray* zero_point = params->zero_point;
    std::vector<float> scaling_factors;
    std::vector<int> zero_points;
    for(int i=0; i<scaling_factor->size; ++i){
      scaling_factors.push_back(scaling_factor->data[i]);
    }
    for(int i=0; i<zero_point->size; ++i){
      zero_points.push_back(zero_point->data[i]);
    }
    for(int i=0; i<tensor_data_size; ++i){
      float temp = (static_cast<int>(data_st_origin[i]) - zero_points[0]) * scaling_factors[0];
      dequantized_values[i] = temp;
    }
  }
  // and return the new buffer
  return dequantized_values;
}


// This mechanism is not thread safe. 
void TfLiteRuntime::RestoreOriginalBuffer(TfLiteTensor* tensor, void* buffer){
  if(buffer == nullptr)
    return;
  int tensor_data_size = 1;
  for(int i=0; i<tensor->dims->size; i++){
    tensor_data_size *= tensor->dims->data[i]; 
  }
  tensor->type = TfLiteType::kTfLiteUInt8;
  free(tensor->data.data);
  tensor->data.data = buffer;
  tensor->bytes = tensor_data_size * sizeof(uint8_t);
  std::cout << "restored original buffer" << "\n";
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
  }else if(tensor.type == TfLiteType::kTfLiteUInt8){
    std::cout << "[UINT8 TENSOR]" << "\n";
    auto data_st = (uint8_t*)tensor.data.data;
    for(int i=0; i<tensor_data_ch_size; i++){
      std::cout << "CH [" << i << "] \n";
      for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
        uint8_t data = *(data_st+(i+j*tensor_data_ch_size));
        if (data == 0) {
          printf("%d ", data);
        }
        else if (data != 0) {
            printf("%s%d%s ", C_GREN, data, C_NRML);
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

std::vector<std::vector<uint8_t>*>* TfLiteRuntime::GetUintOutputInVector(){
  std::vector<std::vector<uint8_t>*>* output = new std::vector<std::vector<uint8_t>*>;
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
    auto data_st = (uint8_t*)tensor->data.data;
    output->push_back(new std::vector<uint8_t>());
    for(int j=0; j<ch; j++){
      uint8_t data = *(data_st + j);
      output->at(0)->push_back(data);
    }
  }else if(tensor->dims->size == 4){
    int tensor_channel_idx = tensor->dims->size-1;
    int tensor_data_ch_size = tensor->dims->data[tensor_channel_idx];
    int tensor_data_size = 1;
    int tensor_axis;
    int ch = (tensor->dims->data[1] * tensor->dims->data[2]);
    int data_size = tensor->dims->data[3];
    auto data_st = (uint8_t*)tensor->data.data;
    for(int i=0; i<ch; i++){
      output->push_back(new std::vector<uint8_t>());
      for(int j=0; j<data_size; j++){
        uint8_t data = *(data_st+ j + (data_size * i));
        output->at(i)->push_back(data);
      }
    }
  }
  return output;
}

}  // namespace tflite