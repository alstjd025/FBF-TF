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
                                                    const char* model) {
  interpreter = new tflite::Interpreter;
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
  PrintInterpreterStateV2(interpreter);
  if(PartitionSubgraphs() != kTfLiteOk){
    std::cout << "Model partitioning ERROR" << "\n";
    exit(-1);
  }
    
};

TfLiteRuntime::~TfLiteRuntime() {
  std::cout << "TfLiteRuntime destructor called"
            << "\n";
};

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
  if(rx_p.runtime_next_state != state){
    state = static_cast<RuntimeState>(rx_p.runtime_next_state);
    std::cout << "Runtime " << runtime_id << " state changed to " << state << "\n";
    return kTfLiteOk;
  }else{
    std::cout << "Runtime " << runtime_id << " state change failed" << "\n";
    return kTfLiteError;
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
      **model_, *resolver, interpreter, model, 0, true);

  // Now creates an invokable origin subgraph from new model.
  if (interpreter_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk) {
    std::cout << "CreateSubgraphFromFlatBuffer returned Error"
              << "\n";
    exit(-1);
  }
  interpreter->PrintSubgraphInfo();
  PrintInterpreterStateV2(interpreter);
  // scheduler->RegisterInterpreterBuilder(new_builder);

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
  // Some profiling code here. later. //
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
  memcpy(partitioning_plan, rx_packet.partitioning_plan, sizeof(int)*1000*3);

  if(ChangeStatewithPacket(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  std::cout << "Successfully registered model to scheduler" << "\n";
  return kTfLiteOk;
}

TfLiteStatus TfLiteRuntime::PartitionSubgraphs(){
  std::vector<std::vector<int>> raw_plan;
  for(int i=0; i<1000; ++i){
    raw_plan.push_back(std::vector<int>());
    if(partitioning_plan[i][0] == -1){
      raw_plan[i].push_back(-1);
      break;
    }
    for(int j=0; j<2; ++j){ // third idx means processor.
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
  tx_packet.runtime_id = runtime_id;
  tx_packet.cur_subgraph = state;
  if(SendPacketToScheduler(tx_packet) != kTfLiteOk){
    return kTfLiteError;
  }

  tf_packet rx_packet;
  if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  if(ChangeStatewithPacket(rx_packet) != kTfLiteOk){
    return kTfLiteError;
  }
  PrintInterpreterStateV2(interpreter);
  std::cout << "Successfully partitioned subgraph" << "\n";
  return kTfLiteOk;
}

void TfLiteRuntime::FeedInputToInterpreter(std::vector<cv::Mat>& mnist,
                                           std::vector<cv::Mat>& imagenet) {
  interpreter->mnist_input = mnist;
  interpreter->imagenet_input = imagenet;
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

void TfLiteRuntime::WakeScheduler() {
  interpreter->WakeScheduler();
  std::this_thread::sleep_for(std::chrono::seconds(3));
}

void TfLiteRuntime::JoinScheduler() { interpreter->JoinScheduler(); }

TfLiteStatus TfLiteRuntime::DebugInvoke() {
  if (interpreter->DebugInvoke() != kTfLiteOk) {
    return kTfLiteError;
  }
  return kTfLiteOk;
};

// working function
TfLiteStatus TfLiteRuntime::Invoke() {
  if(state != RuntimeState::INVOKE_){
    std::cout << "ERROR cannot invoke runtime [" << runtime_id << "]\n";
    std::cout << "State is not INVOKE. cur state is " << state << "\n";
    return kTfLiteError;
  }
  Subgraph* subgraph;
  for(int i=0; i<interpreter->subgraphs_size(); ++i){
    subgraph = interpreter->subgraph(i);
    if(subgraph->GetPrevSubgraph() != nullptr){
      CopyIntermediateDataIfNeeded(subgraph);
    }
    if(subgraph->Invoke() != kTfLiteOk){
      
    }
  }
  // int graphs_to_invoke = interpreter->Get
}

void TfLiteRuntime::CopyIntermediateDataIfNeeded(Subgraph* subgraph) {
  // use source_graph_id, dest_graph_id
  auto connect = [&](int source_subgraph, int dest_subgraph) {
    Subgraph* source_graph = interpreter->subgraph_id(source_subgraph);
    Subgraph* dest_graph = interpreter->subgraph_id(dest_subgraph);
    int source_tensor_idx = source_graph->outputs()[0];
    int dest_tensor_idx = dest_graph->GetInputTensorIndex();
    TfLiteTensor* source_tensor = source_graph->tensor(source_tensor_idx);
    TfLiteTensor* dest_tensor = dest_graph->tensor(dest_tensor_idx);
    size_t source_byte_size = source_tensor->bytes;
    size_t dest_byte_size = dest_tensor->bytes;
    // source_graph->PrintTensor(*source_tensor, UnitType::GPU0);
    if (source_byte_size != dest_byte_size) {
      std::cout << "Source tensor[" << source_tensor_idx << "] size "
                << static_cast<int>(source_byte_size) << " and Dest tensor["
                << dest_tensor_idx << "] size "
                << static_cast<int>(dest_byte_size) << " missmatch!"
                << "\n";
      return kTfLiteError;
    }
    auto data_source = (float*)source_tensor->data.data;
    auto data_dest = (float*)dest_tensor->data.data;
    memcpy(data_dest, data_source, source_byte_size);
    // dest_graph->PrintTensor(*dest_tensor, UnitType::GPU0);
    if (dest_tensor->data.raw == nullptr) {
      std::cout << "dest data nullptr!"
                << "\n";
    }
    #ifdef latency_debug
        std::cout << "Tensor connection done"
                  << "\n";
    #endif
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
  // std::cout << " Nunber of data : " << tensor_data_size << "\n";
  // std::cout << " Tensor DATA " << "\n";
  if(tensor.type == TfLiteType::kTfLiteFloat32){
    auto data_st = (float*)tensor.data.data;
    for(int i=0; i<tensor_data_ch_size; i++){
      if(!is_output)
        std::cout << "CH [" << i << "] \n";
      for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
        float data = *(data_st+(i+j*tensor_data_ch_size));
        if(is_output){
          if(data > 0.8){ // threshhold
            std::cout << "CH [" << i << "] ";
            printf("%s%0.6f%s \n", C_GREN, data, C_NRML);
            //output_correct = true;
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

}  // namespace tflite