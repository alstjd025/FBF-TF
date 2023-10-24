#include "tensorflow/lite/lite_runtime.h"
#include "tensorflow/lite/lite_scheduler.h"
// #define YOLO
// #define mobilenet
// #define debug_print

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
  sub_interpreter = nullptr;
  sub_builder = nullptr;
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
  // interpreter->RegisterDelegate(MyDelegate);
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
  sub_interpreter = new tflite::Interpreter(true);
  sub_builder = nullptr;
  interpreter->SetInputType(type);
  sub_interpreter->SetInputType(type);
  SetInputType(type);
  state = RuntimeState::INITIALIZE;
  uds_runtime_filename = uds_runtime;
  uds_scheduler_filename = uds_scheduler;
  // sj
  // need to edit for delegation
  // use Class Delegation
  TfLiteDelegate* gpu_delegate = NULL;
  TfLiteDelegate* xnn_delegate = NULL;
  int32_t num_threads;

  const TfLiteGpuDelegateOptionsV2 options = {
      .is_precision_loss_allowed = 0,
      .inference_preference =
          TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
      .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION,
      .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .experimental_flags = 1,
      .max_delegated_partitions = 1000,
  };
  gpu_delegate = TfLiteGpuDelegateV2Create(&options);

  num_threads = 6;
	TfLiteXNNPackDelegateOptions xnnpack_options =
		TfLiteXNNPackDelegateOptionsDefault();
	xnnpack_options.num_threads = num_threads;
  xnn_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);


  interpreter->RegisterDelegate(gpu_delegate, xnn_delegate);
  sub_interpreter->RegisterDelegate(gpu_delegate, xnn_delegate);
  if(InitializeUDS() != kTfLiteOk){
    std::cout << "UDS socker init ERROR" << "\n";
    exit(-1);
  }
  if(AddModelToRuntime(f_model, i_model) != kTfLiteOk){
    std::cout << "Model registration to runtime ERROR" << "\n";
  }
  if(RegisterModeltoScheduler() != kTfLiteOk){
    std::cout << "Model registration to scheduler ERROR" << "\n";
  }
  if(PartitionCoSubgraphs() != kTfLiteOk){
    std::cout << "Model partitioning ERROR" << "\n";
  }
};

TfLiteRuntime::~TfLiteRuntime() {
  std::cout << "TfLiteRuntime destructor called"
            << "\n";
};

void TfLiteRuntime::SetTestSequenceName(std::string name){
  sequence_name = name;
  std::cout << "TEST " << sequence_name << " \n";
}

void TfLiteRuntime::SetLogPath(std::string path){
  log_path = path;
  std::cout << "Log path " << log_path << " \n";
}

void TfLiteRuntime::WriteInitStateLog(){
  std::string buf;
  PrintInterpreterStateSimple(interpreter, sub_interpreter, buf);
  m_interpreter_lat_log << buf;
  m_interpreter_t_stamp_log << buf;
  s_interpreter_lat_log << buf;
  s_interpreter_t_stamp_log << buf;
}

void TfLiteRuntime::InitLogFile(){
  std::string lat_file_name = "_latency.txt";
  std::string ts_file_name = "_timestamps.txt";

  m_interpreter_lat_log.open(log_path + sequence_name + lat_file_name);
  s_interpreter_lat_log.open(log_path + sequence_name + "_s" + lat_file_name);
  
  m_interpreter_t_stamp_log.open(log_path + sequence_name + ts_file_name);
  s_interpreter_t_stamp_log.open(log_path + sequence_name + "_s" + ts_file_name);
  return;
}

void TfLiteRuntime::WriteVectorLog(std::vector<double>& log, int log_id){
  switch (log_id)
  {
  case 0:
    if(m_interpreter_lat_log.is_open()){
      for(int i=0; i<log.size(); ++i){
        m_interpreter_lat_log << log[i] << " ";
      }
      m_interpreter_lat_log << "\n";
    }
    break;
  case 1:
    if(s_interpreter_lat_log.is_open()){
      for(int i=0; i<log.size(); ++i){
        s_interpreter_lat_log << log[i] << " ";
      }
      s_interpreter_lat_log << "\n";
    }  
    break;
  case 2:
    if(m_interpreter_t_stamp_log.is_open()){
      m_interpreter_t_stamp_log.precision(14);
      for(int i=0; i<log.size(); ++i){
        m_interpreter_t_stamp_log << log[i] << " ";
      }
      m_interpreter_t_stamp_log << "\n";
    }
    break;
  case 3:
    if(s_interpreter_t_stamp_log.is_open()){
      s_interpreter_t_stamp_log.precision(14);
      for(int i=0; i<log.size(); ++i){
        s_interpreter_t_stamp_log << log[i] << " ";
      }
      s_interpreter_t_stamp_log << "\n";
    } 
    break;
  default:
    std::cout << "Wrong logging id" << "\n";
    break;
  }
}

void TfLiteRuntime::WriteVectorLog(std::vector<double>& log,
                                   std::vector<string>& label, int log_id){
  switch (log_id)
  {
  case 0:
    if(m_interpreter_lat_log.is_open()){
      for(int i=0; i<log.size(); ++i){
        m_interpreter_lat_log << log[i] << " ";
      }
      m_interpreter_lat_log << "\n";
    }
    break;
  case 1:
    if(s_interpreter_lat_log.is_open()){
      for(int i=0; i<log.size(); ++i){
        s_interpreter_lat_log << log[i] << " ";
      }
      s_interpreter_lat_log << "\n";
    }  
    break;
  case 2:
    if(m_interpreter_t_stamp_log.is_open()){
      m_interpreter_t_stamp_log.precision(14);
      for(int i=0; i<log.size(); ++i){
        m_interpreter_t_stamp_log << label[i] << " " << log[i] << " ";
      }
      m_interpreter_t_stamp_log << "\n";
    }
    break;
  case 3:
    if(s_interpreter_t_stamp_log.is_open()){
      for(int i=0; i<log.size(); ++i){
        s_interpreter_t_stamp_log << label[i] << " " << log[i] << " ";
      }
      s_interpreter_t_stamp_log << "\n";
    } 
    break;
  default:
    std::cout << "Wrong logging id" << "\n";
    break;
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

void TfLiteRuntime::ShutdownScheduler(){
  tf_packet tx_packet;
  memset(&tx_packet, 0, sizeof(tf_packet));
  tx_packet.runtime_current_state = RuntimeState::TERMINATE;
  tx_packet.runtime_id = runtime_id;
  if(SendPacketToScheduler(tx_packet) != kTfLiteOk){
    std::cout << "Sechduler Shutdown Error" << "\n";
  }
  return;
}

TfLiteStatus TfLiteRuntime::SendPacketToScheduler(tf_packet& tx_p){
  // std::cout << "Runtime :Send packet to scheduler" << "\n";
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
  sub_builder = new tflite::InterpreterBuilder(
      **int_model, *int_resolver, sub_interpreter, i_model, 0, true);

  // Now creates an invokable (float)origin subgraph from new model.
  if (interpreter_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk) {
    std::cout << "CreateSubgraphFromFlatBuffer returned Error"
              << "\n";
    exit(-1);
  }

  // Now creates an invokable (int)origin subgraph from new model.
  if (sub_builder->CreateSubgraphFromFlatBuffer() != kTfLiteOk) {
    std::cout << "CreateSubgraphFromFlatBuffer returned Error"
              << "\n";
    exit(-1);
  }
  #ifdef debug_print
    std::cout << "============================" << "\n";
    std::cout << "Full precision interpreter" << "\n";
    PrintInterpreterStateV3(interpreter);
    std::cout << "============================" << "\n";
    std::cout << "Minimal precision interpreter" << "\n";
    PrintInterpreterStateV3(sub_interpreter);
    interpreter->PrintSubgraphInfo();
  #endif
  
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
  // interpreter->PrintSubgraphInfo();
  // sub_interpreter->PrintSubgraphInfo();
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
      sub_builder->CopyRawPartitioningPlan(raw_plan);
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
  Subgraph* origin_quantized_subgraph = sub_interpreter->returnProfiledOriginalSubgraph(0);
  if(origin_quantized_subgraph == nullptr){
    std::cout << "Model id " << interpreter_builder->GetModelid() << " no subgraph. \n"; 
    return kTfLiteError;
  }
  if(sub_builder->CreateSubgraphsFromProfiling(origin_quantized_subgraph)
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
  sub_interpreter->GetTotalSubgraphID(min_prec_subgraphs);
  
  for(int i=0; i<full_prec_subgraphs.size(); ++i){
    tx_packet.subgraph_ids[0][i] = full_prec_subgraphs[i];
  }
  tx_packet.subgraph_ids[0][full_prec_subgraphs.size()] = -1;
  for(int i=0; i<min_prec_subgraphs.size(); ++i){  
    tx_packet.subgraph_ids[1][i] = min_prec_subgraphs[i];
  }
  tx_packet.subgraph_ids[1][min_prec_subgraphs.size()] = -1;
  
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
  PrintInterpreterStateV3(interpreter);
  PrintInterpreterStateDimandSize(interpreter);
  std::cout << "=====================" << "\n";
  std::cout << "MIN precicion interpreter state" << "\n";
  PrintInterpreterStateV3(sub_interpreter);
  PrintInterpreterStateDimandSize(sub_interpreter);
  std::cout << "Successfully partitioned subgraph" << "\n";
  std::cout << "Ready to invoke" << "\n";
  return kTfLiteOk;
} 

TfLiteStatus TfLiteRuntime::PrepareCoExecution(){
  if(sub_interpreter == nullptr){
    std::cout << "PrepareCoExecution ERROR" << "\n";
    std::cout << "minimal precision interpreter nullptr" << "\n";
    return kTfLiteError;
  }
  int co_subgraph_idx = 0;
  for(int subgraph_idx=0; subgraph_idx < interpreter->subgraphs_size();
        subgraph_idx++){
    Subgraph* subgraph = interpreter->subgraph(subgraph_idx);
    if(subgraph->GetResourceType() == ResourceType::CO_GPU){
      if(sub_interpreter->subgraphs_size() < 1){
        std::cout << "PrepareCoExecution ERROR" << "\n";
        std::cout << "minimal precision interpreter has no subgraph" << "\n";
        return kTfLiteError;
      }
      Subgraph* co_subgraph = sub_interpreter->subgraph(co_subgraph_idx);
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

INPUT_TYPE TfLiteRuntime::GetInputTypeFromString(string input_type){
  if(strcmp(input_type.c_str(), "IMAGENET224") == 0){
    return INPUT_TYPE::IMAGENET224;
  }else if(strcmp(input_type.c_str(), "IMAGENET300") == 0){
    return INPUT_TYPE::IMAGENET300;
  }else if(strcmp(input_type.c_str(), "COCO416") == 0){
    return INPUT_TYPE::COCO416;
  }else if(strcmp(input_type.c_str(), "MNIST") == 0){
    return INPUT_TYPE::MNIST;
  }
  else{
    return INPUT_TYPE::USER;
  }
}

void TfLiteRuntime::SetInputType(INPUT_TYPE input_type_){
  input_type = input_type_;
}

// TODO : Impl global input tensor sharing
// MUST_REFACTOR (628b2) : MUST refactor for sub-interpreter input.
void TfLiteRuntime::CopyInputToInterpreter(const char* model,
                                     cv::Mat& input, cv::Mat& input_quant) {
  bool use_two_interpreter = false;
  ResourceType primary_resource = interpreter->primary_subgraph().GetResourceType();
  if(primary_resource != ResourceType::GPU && primary_resource != ResourceType::CPU
     && primary_resource != ResourceType::CPU_XNN){
    if(sub_interpreter != nullptr){
      use_two_interpreter = true;
    }
  }

  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* input_tensor_sub = nullptr;
  input_tensor = interpreter->input_tensor_of_model(0);
  if(use_two_interpreter){
    input_tensor_sub = sub_interpreter->input_tensor_of_model(0);
  }
  if(input_tensor == nullptr) {
    std::cout << "TfLiteRuntime : cannot get pointer to input tensor, model["
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
            printf("%.02f ", ((float)input.at<uchar>(i, j) / 255.0));
          }
          printf("\n");
        }
        if(use_two_interpreter){
          auto input_pointer = (float*)input_tensor_sub->data.data;
          int h = input_tensor_sub->dims->data[1];
          int w = input_tensor_sub->dims->data[2];
          for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){ 
              input_pointer[i * w + j] = ((float)input.at<uchar>(i + (w - h), j) / 255.0);
              printf("%.02f ", ((float)input.at<uchar>(i + (w - h), j) / 255.0));
            }
            printf("\n");
          }
        }       
        break;
      case INPUT_TYPE::IMAGENET224:
        for (int i=0; i<h; i++){ // row
          for (int j=0; j<w; j++){ // col
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            *(input_pointer + i * w*3 + j * 3) = ((float)pixel[0])/255.0;
            *(input_pointer + i * w*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
            *(input_pointer + i * w*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
          }
        }
        if(use_two_interpreter){
          auto input_pointer_sub = (float*)input_tensor_sub->data.data;
          int h = input_tensor_sub->dims->data[1];
          int w = input_tensor_sub->dims->data[2];
          for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){   
              cv::Vec3b pixel = input.at<cv::Vec3b>(i + (w - h), j);
              *(input_pointer_sub + i * w*3 + j * 3) = ((float)pixel[0])/255.0;
              *(input_pointer_sub + i * w*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
              *(input_pointer_sub + i * w*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
            }
          }
        }
        break;
      case INPUT_TYPE::IMAGENET300:
        for (int i=0; i<h; i++){ // row
          for (int j=0; j<w; j++){ // col
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            *(input_pointer + i * w*3 + j * 3) = ((float)pixel[0])/255.0;
            *(input_pointer + i * w*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
            *(input_pointer + i * w*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
          }
        }
        if(use_two_interpreter){
          auto input_pointer_sub = (float*)input_tensor_sub->data.data;
          int h = input_tensor_sub->dims->data[1];
          int w = input_tensor_sub->dims->data[2];
          for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){   
              cv::Vec3b pixel = input.at<cv::Vec3b>(i + (w - h), j);
              *(input_pointer_sub + i * w*3 + j * 3) = ((float)pixel[0])/255.0;
              *(input_pointer_sub + i * w*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
              *(input_pointer_sub + i * w*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
            }
          }
        }
        break;
      case INPUT_TYPE::COCO416:
        for (int i=0; i<h; i++){ // row
          for (int j=0; j<w; j++){ // col
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            *(input_pointer + i * w*3 + j * 3) = ((float)pixel[0])/255.0;
            *(input_pointer + i * w*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
            *(input_pointer + i * w*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
          }
        }
        if(use_two_interpreter){
          auto input_pointer_sub = (float*)input_tensor_sub->data.data;
          int h = input_tensor_sub->dims->data[1];
          int w = input_tensor_sub->dims->data[2];
          for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){   
              cv::Vec3b pixel = input.at<cv::Vec3b>(i + (w - h), j);
              *(input_pointer_sub + i * w*3 + j * 3) = ((float)pixel[0])/255.0;
              *(input_pointer_sub + i * w*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
              *(input_pointer_sub + i * w*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
            }
          }
        }
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
      case INPUT_TYPE::IMAGENET300:
        memcpy(input_pointer, input_quant.data,
              h * w * input_quant.elemSize());
        break;
      case INPUT_TYPE::COCO416:
         ////////////////////////////////////////////////////////////////////
        // HOON : correct method to push image to input_tensor in YOLOv4-tiny
        for (int i=0; i<h; i++){
          for (int j=0; j<w; j++){   
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            *(input_pointer + i * 416*3 + j * 3) = ((float)pixel[0])/255.0;
            *(input_pointer + i * 416*3 + j * 3 + 1) = ((float)pixel[1])/255.0;
            *(input_pointer + i * 416*3 + j * 3 + 2) = ((float)pixel[2])/255.0;
          }
        }
        ////////////////////////////////////////////////////////////////////
        break;
      case INPUT_TYPE::LANENET144800:
        memcpy(input_pointer, input_quant.data,
              h * w * input_quant.elemSize()); 
      default:
        break;
    }
  }
}

TfLiteStatus TfLiteRuntime::Invoke(){
  TfLiteStatus return_state_sub = TfLiteStatus::kTfLiteOk;
  TfLiteStatus return_state_main = TfLiteStatus::kTfLiteOk;
  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC, &begin);
  timestamp_main_interpreter.push_back(begin.tv_sec + (begin.tv_nsec / 1000000000.0));
  timestamp_label_main_interpreter.push_back("Start");
  c_thread = std::thread(&TfLiteRuntime::DoInvoke, this, 
                          InterpreterType::SUB_INTERPRETER, std::ref(return_state_sub));
  DoInvoke(InterpreterType::MAIN_INTERPRETER, return_state_main);
  if(return_state_main != kTfLiteOk || return_state_sub != kTfLiteOk){
    std::cout << "Invoke error in interpreter" << "\n";
    c_thread.join();
    return kTfLiteError;
  }
  c_thread.join();
  clock_gettime(CLOCK_MONOTONIC, &end);
  timestamp_main_interpreter.push_back(end.tv_sec + (end.tv_nsec / 1000000000.0));
  timestamp_label_main_interpreter.push_back("End");
  double response_time = (end.tv_sec - begin.tv_sec) +
                         ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
  
  latency_main_interpreter.push_back(response_time);
  WriteVectorLog(latency_main_interpreter, 0);
  WriteVectorLog(timestamp_main_interpreter, timestamp_label_main_interpreter, 2);
  WriteVectorLog(latency_sub_interpreter, 1);
  WriteVectorLog(timestamp_sub_interpreter, timestamp_label_sub_interpreter, 3);
  latency_main_interpreter.clear();
  timestamp_main_interpreter.clear();
  timestamp_label_main_interpreter.clear();
  latency_sub_interpreter.clear();
  timestamp_sub_interpreter.clear();
  timestamp_label_sub_interpreter.clear();
  return kTfLiteOk;
}

////////////////////////////////////////////////////////////////////////////////
// Working function
// TODO : 1. Get subgraph id to invoke from Graphselector.
//        2. Get system monitoring info from scheduler at each invoke.
////////////////////////////////////////////////////////////////////////////////
void TfLiteRuntime::DoInvoke(InterpreterType type, TfLiteStatus& return_state){
  // For prototye, invoke first layer only with HW-partitioning and merge them.
  Subgraph* subgraph;
  int subgraph_idx = 0;
  double response_time = 0; 
  struct timespec begin, end;
  int subgraph_id = -1;
  int prev_subgraph_id = -1;
  int prev_co_subgraph_id = -1;
  while(true){
    if(type == InterpreterType::SUB_INTERPRETER){
      if(sub_interpreter->subgraphs_size() < 1){
        // std::cout << "No invokable subgraph for cpu" << "\n";
        break;
      }
      // sync with gpu here (notified by gpu)
      std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
      invoke_sync_cv.wait(lock_invoke, [&]{ return invoke_cpu; });
      invoke_cpu = false;
      if(co_subgraph_id == -1){
        #ifdef debug_print
          std::cout << "Sub Interpreter invoke done" << "\n";
        #endif
        return_state = kTfLiteOk;
        return;
      }
      #ifdef debug_print
        std::cout << "[Sub Interpreter] get subgraph " << co_subgraph_id << "\n";
      #endif
      subgraph = sub_interpreter->subgraph_id(co_subgraph_id);
      // TfLiteTensor* input_t = subgraph->tensor(0);
      // PrintTensorSerial(*input_t);
      if(main_execution_graph != nullptr){
        #ifdef debug_print
          std::cout << "sub CopyIntermediateDataIfNeeded" << "\n";
        #endif
        clock_gettime(CLOCK_MONOTONIC, &begin);
        if(CopyIntermediateDataIfNeeded(subgraph, main_execution_graph, merge_tensor)
           != kTfLiteOk){
          std::cout << "sub CopyIntermediateDataIfNeeded Failed" << "\n";
          return_state = kTfLiteError;
          return;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        timestamp_sub_interpreter.push_back(begin.tv_sec + (begin.tv_nsec / 1000000000.0));
        timestamp_label_sub_interpreter.push_back("CPs");
        timestamp_sub_interpreter.push_back(end.tv_sec + (end.tv_nsec / 1000000000.0));
        timestamp_label_sub_interpreter.push_back("CPe");
      }
      #ifdef debug_print
        std::cout << "[Sub Interpreter] Invoke subgraph " << co_subgraph_id << "\n";
      #endif
      clock_gettime(CLOCK_MONOTONIC, &begin);
      // if(co_subgraph_id == 1){
      //   TfLiteTensor* input_t = subgraph->tensor(0);
      //   PrintTensorSerial(*input_t);
      // }
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking CPU subgraph " << subgraph->GetGraphid() << "\n";
        return_state = kTfLiteError;
        return;
      }
      // if(co_subgraph_id == 1){
      //   TfLiteTensor* input_t = subgraph->tensor(70);
      //   PrintTensorSerial(*input_t);
      // }
      clock_gettime(CLOCK_MONOTONIC, &end);
      response_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      timestamp_sub_interpreter.push_back(begin.tv_sec + (begin.tv_nsec / 1000000000.0));
      timestamp_label_sub_interpreter.push_back("IVs");
      timestamp_sub_interpreter.push_back(end.tv_sec + (end.tv_nsec / 1000000000.0));
      timestamp_label_sub_interpreter.push_back("IVe");
      latency_sub_interpreter.push_back(response_time);
      // sync with gpu here (wake gpu)
      std::unique_lock<std::mutex> lock_data(data_sync_mtx);
      is_execution_done = true;
      co_execution_graph = subgraph;
      data_sync_cv.notify_one();
      
    }else if(type == InterpreterType::MAIN_INTERPRETER){
      // TODO (d9a62) : Make this part to an individual function.
      tf_packet tx_packet;
      memset(&tx_packet, 0, sizeof(tf_packet));
      tx_packet.runtime_id = runtime_id;
      tx_packet.runtime_current_state = state;
      tx_packet.cur_subgraph = subgraph_id;
      merge_tensor = nullptr;
      if(SendPacketToScheduler(tx_packet) != kTfLiteOk){ // Request invoke permission to scheduler
        return_state = kTfLiteError;
        break;
      }
      tf_packet rx_packet;
      if(ReceivePacketFromScheduler(rx_packet) != kTfLiteOk){
        return_state = kTfLiteError;
        break;
      }
      if(rx_packet.subgraph_ids[0][0] == -1){
        #ifdef debug_print
          std::cout << "Main Interpreter graph invoke done" << "\n";
        #endif
        subgraph_id = -1;
        prev_subgraph_id = -1;
        prev_co_subgraph_id = -1;
        co_subgraph_id = -1;
        main_execution_graph = nullptr;
        std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
        invoke_cpu = true;
        invoke_sync_cv.notify_one();
        #ifdef mobilenet
          global_output_tensor = subgraph->tensor(303);
        #endif
        // PrintTensor(*(subgraph->tensor(18)), true);
        ////////////////////////////////////////////////////////////////////
        //HOON
        #ifdef YOLO
        YOLO_Parser yolo_parser;
        printf("\033[0;33mStart YOLO parsing\033[0m\n");
        std::vector<int> real_bbox_index_vector;
        real_bbox_index_vector.clear();
        YOLO_Parser::real_bbox_cls_index_vector.clear();
        YOLO_Parser::real_bbox_cls_vector.clear();
        YOLO_Parser::real_bbox_loc_vector.clear();
        YOLO_Parser::result_boxes.clear();
        TfLiteTensor* cls_tensor = subgraph->tensor(212); // 1 2535 80
        TfLiteTensor* loc_tensor = subgraph->tensor(233); // 1 2535 4
        yolo_parser.make_real_bbox_cls_vector(cls_tensor, real_bbox_index_vector,
                                               YOLO_Parser::real_bbox_cls_vector);
        YOLO_Parser::real_bbox_cls_index_vector = \
                    yolo_parser.get_cls_index(YOLO_Parser::real_bbox_cls_vector); 
        yolo_parser.make_real_bbox_loc_vector(loc_tensor, real_bbox_index_vector, 
                                              YOLO_Parser::real_bbox_loc_vector);
        float iou_threshold = 0.5;
        yolo_parser.PerformNMSUsingResults(real_bbox_index_vector, YOLO_Parser::real_bbox_cls_vector, 
              YOLO_Parser::real_bbox_loc_vector, iou_threshold, YOLO_Parser::real_bbox_cls_index_vector);
        printf("\033[0;33mEND YOLO parsing\033[0m\n");
        for(int i=0; i<YOLO_Parser::result_boxes.size(); ++i){
          std::cout << "BOX i " << i << "\n";
          std::cout << "LABEL : " << YOLO_Parser::result_boxes[i].class_id + 1
                    << ":" << yolo_parser.yolo_labels[YOLO_Parser::result_boxes[i].class_id] << "\n";
          std::cout << "SCORE : " << YOLO_Parser::result_boxes[i].score << "\n";
        }
        #endif
        ////////////////////////////////////////////////////////////////////
        return_state = kTfLiteOk;
        return;
        // break;
      }
      subgraph_id = rx_packet.subgraph_ids[0][0];
      if(rx_packet.subgraph_ids[1][0] != -1)
        co_subgraph_id = rx_packet.subgraph_ids[1][0];
      bool merged = false;
      // Check if co execution. If so, give co-execution graph to sub-interpreter and notify.
      subgraph = interpreter->subgraph_id(subgraph_id);
      #ifdef debug_print
        std::cout << "[Main interpreter] get subgraph " << subgraph_id << "\n";
      #endif
      // if previous subgraph was co-execution, merge co-exectuion data here.
      if(prev_subgraph_id != subgraph_id && prev_subgraph_id != -1 &&
          interpreter->subgraph_id(prev_subgraph_id)->GetResourceType() == CO_GPU){
        #ifdef debug_print
          std::cout << "Merge " << prev_subgraph_id << " " << prev_co_subgraph_id << 
                    " " << subgraph_id << "\n";
        #endif
        merged = true;
        // Need extra buffer tensor if previous subgraph was co-execution and
        // current subgraph is co-execution.
        if(subgraph->GetResourceType() == CO_GPU){ 
          merge_tensor = new TfLiteMergeTensor;
          merge_tensor->tensor = new TfLiteTensor;
          merged = false;
        }
        clock_gettime(CLOCK_MONOTONIC, &begin);
        if(MergeCoExecutionData(prev_co_subgraph_id, prev_subgraph_id, subgraph_id, merge_tensor)
           != kTfLiteOk){
          std::cout << "MergeCoExecutionData Error" << "\n";
          return_state = kTfLiteError;
          break;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        timestamp_label_main_interpreter.push_back("MGs");
        timestamp_main_interpreter.push_back(begin.tv_sec + (begin.tv_nsec / 1000000000.0));
        timestamp_label_main_interpreter.push_back("MGe");
        timestamp_main_interpreter.push_back(end.tv_sec + (end.tv_nsec / 1000000000.0));
      }
      
      if(subgraph->GetResourceType() == CO_GPU){
        // wake cpu thread here
        if(prev_subgraph_id != -1){
          // Consider intermediate tensor is located in prev-previous subgraph?
          // (99462)
          main_execution_graph = interpreter->subgraph_id(prev_subgraph_id);
        }
        std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
        invoke_cpu = true;
        invoke_sync_cv.notify_one();
      }
      // if not co-execution, it needs additional imtermediate data copy.
      if(prev_subgraph_id != -1 && !merged){
        #ifdef debug_print
          std::cout << "Main CopyIntermediateDataIfNeeded" << "\n";
        #endif
        clock_gettime(CLOCK_MONOTONIC, &begin);
        if(CopyIntermediateDataIfNeeded(subgraph, prev_subgraph_id, merge_tensor)
          != kTfLiteOk){
          std::cout << "Main CopyIntermediateDataIfNeeded Failed" << "\n";
          return_state = kTfLiteError;
          break;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        timestamp_label_main_interpreter.push_back("CPs");
        double response_time = (end.tv_sec - begin.tv_sec) +
                        ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
        printf("CPS %.6f", response_time);
        timestamp_main_interpreter.push_back(begin.tv_sec + (begin.tv_nsec / 1000000000.0));
        timestamp_label_main_interpreter.push_back("CPe");
        timestamp_main_interpreter.push_back(end.tv_sec + (end.tv_nsec / 1000000000.0));
        #ifdef debug_print
          std::cout << "Main CopyIntermediateDataIfNeeded Done" << "\n";
        #endif
      }
      
      #ifdef debug_print
      std::cout << "[Main interpreter] Invoke subgraph " << subgraph->GetGraphid() << "\n";
      #endif
      FeedDummyInputToTensor(subgraph->tensor(58));
      clock_gettime(CLOCK_MONOTONIC, &begin);
      if(subgraph->Invoke() != kTfLiteOk){
        std::cout << "ERROR on invoking subgraph id " << subgraph->GetGraphid() << "\n";
        return_state = kTfLiteError;
        break;
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      response_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0); 
      timestamp_label_main_interpreter.push_back("IVs");
      timestamp_main_interpreter.push_back(begin.tv_sec + (begin.tv_nsec / 1000000000.0));
      timestamp_label_main_interpreter.push_back("IVe");
      timestamp_main_interpreter.push_back(end.tv_sec + (end.tv_nsec / 1000000000.0));
      latency_main_interpreter.push_back(response_time);
      if(subgraph->GetResourceType() == ResourceType::CO_GPU){
        // sync with cpu here
        std::unique_lock<std::mutex> lock_data(data_sync_mtx);
        data_sync_cv.wait(lock_data, [&]{ return is_execution_done; });
        is_execution_done = false;
        if(co_execution_graph != nullptr){
          prev_co_subgraph_id = co_subgraph_id;
          co_execution_graph = nullptr;
          // clean merge tensor here (used in previous subgraph.)
          // std::cout << "merge_tensor is used " << merge_tensor->is_used << "\n";
          if(merge_tensor != nullptr && merge_tensor->is_used){
            free(merge_tensor->tensor->data.data); 
            free(merge_tensor->tensor->dims);
            free(merge_tensor->tensor);
            free(merge_tensor);
            merge_tensor = nullptr;
          }
        }
      }
      prev_subgraph_id = subgraph_id;
    }
  }
  if(return_state != kTfLiteOk){
    if(type == InterpreterType::MAIN_INTERPRETER){
      co_subgraph_id = -1;
      main_execution_graph = nullptr;
      std::unique_lock<std::mutex> lock_invoke(invoke_sync_mtx);
      invoke_cpu = true;
      invoke_sync_cv.notify_one();
    }
    return;
  }
  return_state = kTfLiteOk;
  return;
}

////////////////////////////////////////////////////////////////////
// HOON
std::vector<YOLO_Parser::BoundingBox> YOLO_Parser::result_boxes;
std::vector<std::vector<float>> YOLO_Parser::real_bbox_cls_vector; 
std::vector<int> YOLO_Parser::real_bbox_cls_index_vector;
std::vector<std::vector<int>> YOLO_Parser::real_bbox_loc_vector;

std::vector<int> YOLO_Parser::get_cls_index(std::vector<std::vector<float>>& real_bbox_cls_vector){
  float max=0;
  int max_index = -1;
  int index = 0;
  for (auto i : real_bbox_cls_vector) { 
    index = 0;
		for (auto j : i) { 
      if (j > max){
        max = j;
        max_index = index;
      }
      index+=1;
		}
    real_bbox_cls_index_vector.push_back(max_index);
    max = 0;
    max_index = -1;
	}
  return real_bbox_cls_index_vector;
}

void YOLO_Parser::make_real_bbox_cls_vector(TfLiteTensor* cls_tensor, 
 std::vector<int>& real_bbox_index_vector, std::vector<std::vector<float>>& real_bbox_cls_vector){
  TfLiteTensor* output_tensor = cls_tensor;  
  const float* output_data_2 = (float*)output_tensor->data.data;
  const int num_boxes_2 = output_tensor->dims->data[1]; 
  std::vector<float> classifications;
  float cls_thresh = 0.05; // Hyperparam
  for (int i = 0; i < num_boxes_2; ++i) {
    for (int j = 0; j < 80; ++j) {
        classifications.push_back(output_data_2[i*80 + j]);  
       }
  }
  int conf_count = 0;
  int real_bbox_index = -1;
  std::vector<float> raw_vector;
  for (int i = 0; i < num_boxes_2; ++i) {
    int box_per_conf_count = 0;
    for (int j = 0; j < 80; ++j) {
      raw_vector.push_back(classifications[i * 80 + j]); 
    }
    // SOFTMAX(raw_vector); // Not use Softmax currently
    for (int k = 0; k < 80; ++k) {
      if (raw_vector[k] > cls_thresh){
        box_per_conf_count +=1;
      }
    }
    if(box_per_conf_count >0){
      conf_count +=1;
      real_bbox_index_vector.push_back(i); 
      real_bbox_cls_vector.push_back(raw_vector);
    }
    raw_vector.clear();
  }
  classifications.clear();
  printf("\033[0;32mBefore NMS : \033[0m");
  std::cout << " Number of bounding boxes before NMS : " << real_bbox_index_vector.size() << std::endl;
}

void YOLO_Parser::make_real_bbox_loc_vector(TfLiteTensor* loc_tensor,std::vector<int>& real_bbox_index_vector,
                                            std::vector<std::vector<int>>& real_bbox_loc_vector){
  TfLiteTensor* output_tensor = loc_tensor;
  auto input_pointer = (float *)output_tensor->data.data;
  const float* output_data = (float*)output_tensor->data.data; 
  const int num_boxes = output_tensor->dims->data[1]; 
  const int num_columns = output_tensor->dims->data[2]; 
  std::vector<float> boxes;
  for (int i = 0; i < 2535; ++i) {
       for (int j = 0; j < 4; ++j) {
          boxes.push_back(output_data[i * 4 + j]);  
       }
  }
  int bbox_count = 0;
  int image_size = 416; 
  for (int i = 0; i < 2535; ++i) {
      std::vector<int>tmp;
      for(int j=0 ; j < real_bbox_index_vector.size(); j++){
          if(i == real_bbox_index_vector[j]) {
            float first = boxes[i * 4];      
            float second = boxes[i * 4 + 1]; 
            float third = boxes[i * 4 + 2]; 
            float fourth = boxes[i* 4 + 3];   
            int left = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
            (image_size), first - third/2)));
            int top = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
            (image_size), second - fourth/2)));
            int right = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
            (image_size), first + third/2)));
            int bottom = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
            (image_size), second + fourth/2)));
            tmp.push_back(left);
            tmp.push_back(top);
            tmp.push_back(right);
            tmp.push_back(bottom);
            real_bbox_loc_vector.push_back(tmp);
            break;
          }
      }
      tmp.clear();
  }
}
////////////////////////////////////////////////////////////////////
// Description below is conceptual. 
// Get dest subgraphs('D')'s next subgraph. (subgraph 'Dn')
// Compare input & output dims between 'Dn' and 'S'(source subgraph, probabily cpu side)
// Merge 'S' and 'D's output tensor to 'Dn's input tensor.
// NEED TO FIX FOR MULTIPLE OUTPUT TENSORS.
TfLiteStatus TfLiteRuntime::MergeCoExecutionData(int prev_sub_subgraph
                                        , int prev_main_subgraph
                                        , int dest_subgraph_
                                        , TfLiteMergeTensor* buffer_tensor){
  Subgraph* min_precision_subgraph = sub_interpreter->subgraph_id(prev_sub_subgraph);
  Subgraph* max_precision_subgraph = interpreter->subgraph_id(prev_main_subgraph);
  Subgraph* dest_subgraph = interpreter->subgraph_id(dest_subgraph_);
  PartitioningType partitioned_type = PartitioningType::NO_PARTITIONING;
  if(dest_subgraph == nullptr){
    std::cout << "MergeCoExecutionData ERROR" << "\n";
    std::cout << "dest_subgraph nullptr." << "\n";
    return kTfLiteError;
  }
  std::vector<int> dest_tensor_indicies = 
                              dest_subgraph->inputs();

  int dest_tensor_idx = dest_subgraph->GetInputTensorIndex();
  int min_precision_tensor_idx = min_precision_subgraph->GetFirstOutputTensorIndex(); // for uint model
  int max_precision_tensor_idx = max_precision_subgraph->GetFirstOutputTensorIndex();
  int dequant_reference_tensor_idx = 0;
  partitioned_type = max_precision_subgraph->GetPartitioningType();
  #ifdef debug_print
    std::cout << "Merge two tensors, " << min_precision_tensor_idx << " "
              << max_precision_tensor_idx << " to " << dest_tensor_idx << "\n";
  #endif 
  TfLiteTensor* min_precision_tensor = 
                  min_precision_subgraph->tensor(min_precision_tensor_idx);
  TfLiteTensor* max_precision_tensor = 
                  max_precision_subgraph->tensor(max_precision_tensor_idx);
  TfLiteTensor* dest_tensor = nullptr;
  if(buffer_tensor != nullptr){
    dest_tensor = buffer_tensor->tensor;
    buffer_tensor->is_used = true;
    buffer_tensor->tensor_idx = dest_tensor_idx;
    buffer_tensor->partition_type = partitioned_type;
    // initialize merge tensor
    int new_size = 1;
    std::vector<int> new_dim;
    dest_tensor->dims = TfLiteIntArrayCreate(4);
    dest_tensor->dims->data[0] = max_precision_tensor->dims->data[0];
    dest_tensor->dims->data[2] = max_precision_tensor->dims->data[2];
    if(partitioned_type == PartitioningType::CHANNEL_PARTITIONING){
      dest_tensor->dims->data[1] = max_precision_tensor->dims->data[1];
      dest_tensor->dims->data[3] = max_precision_tensor->dims->data[3] + \
                                   min_precision_tensor->dims->data[3];
    }else{ // HW partitioned
      dest_tensor->dims->data[3] = max_precision_tensor->dims->data[3];
      // Assume that H,W always same in the origin shape of tensor.
      dest_tensor->dims->data[1] = max_precision_tensor->dims->data[2];
    }
    for(int i=0; i<4; ++i){
      new_size *= dest_tensor->dims->data[i];
    }
    dest_tensor->type = kTfLiteFloat32;
    dest_tensor->data.data = new float[new_size];
    dest_tensor->bytes = static_cast<size_t>(new_size * sizeof(float));
  }else{
    dest_tensor = dest_subgraph->tensor(dest_tensor_idx);
  }
             
  int quantization_param_tensor_idx =  
                  min_precision_subgraph->GetFirstInputTensorIndex();
  TfLiteTensor* dequant_reference_tensor = nullptr;
  if(min_precision_subgraph->tensor(quantization_param_tensor_idx)->quantization.type
    == kTfLiteAffineQuantization){
    dequant_reference_tensor = min_precision_subgraph->tensor(quantization_param_tensor_idx);
  }else{
    quantization_param_tensor_idx = max_precision_subgraph->GetFirstInputTensorIndex();
    dequant_reference_tensor = max_precision_subgraph->tensor(quantization_param_tensor_idx); 
  }
  
  if(dest_tensor == nullptr || min_precision_tensor == nullptr ||
      max_precision_tensor == nullptr){
    std::cout << "MergeCoExecutionData ERROR" << "\n";
    std::cout << "Tensor NULLPTR" << "\n";
    return kTfLiteError;
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
      return kTfLiteError;
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
  if(partitioned_type == PartitioningType::CHANNEL_PARTITIONING){
    // Merge CW-partitioned data 
    int dest_ch, min_tensor_ch, max_tensor_ch;
    dest_ch = dest_tensor->dims->data[3];
    min_tensor_ch = min_precision_tensor->dims->data[3];
    max_tensor_ch = max_precision_tensor->dims->data[3];
    // std::cout << "min_tensor_ch " << min_tensor_ch << " max_tensor_ch " << max_tensor_ch << "\n";
    if((min_tensor_ch + max_tensor_ch) != dest_ch){
      std::cout << "Tensor dim [OCH] min_prec + max_prec != dest ERROR" << "\n";
      return kTfLiteError;
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
    
  }else if(partitioned_type == PartitioningType::HEIGHT_PARTITIONING){ // Merge HW-partitioned data
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
      float drop_height = ((min_tensor_ht + max_tensor_ht) - dest_ht);
      int dropped_max_data_size = (max_precision_data_size + min_precision_data_size)
                                                             - tensor_dest_data_size;
      max_precision_data_size -= dropped_max_data_size;
      if(drop_height < 0){
        std::cout << "Wrong drop in HW merging ERROR on graph" << "\n";
        std::cout << "min sub: " << min_precision_subgraph->GetGraphid() <<
              " h: " << min_tensor_ht << 
              " max sub: " << max_precision_subgraph->GetGraphid() << 
              " h: " << max_tensor_ht << " dest: " << dest_ht << "\n";
        return kTfLiteError;
      }
      // bug code
      memcpy(data_dest, data_max, sizeof(float)*max_precision_data_size);
      memcpy(data_dest+max_precision_data_size, data_min, 
              sizeof(float)*(tensor_dest_data_size - max_precision_data_size));
    }else{  // No need to drop (fits with destination tensor).
      memcpy(data_dest, data_max, sizeof(float)*max_precision_data_size);
      memcpy(data_dest+max_precision_data_size, data_min, 
              sizeof(float)*min_precision_data_size);
    }
  }
  return kTfLiteOk; 
}

// For main subgraph
TfLiteStatus TfLiteRuntime::CopyIntermediateDataIfNeeded(Subgraph* subgraph,
                                                         int prev_subgraph_id,
                                                         TfLiteMergeTensor* merge_tensor) {
  // use source_graph_id, dest_graph_id
  auto connect = [&](int source_subgraph, int dest_subgraph) {
    Subgraph* source_graph = interpreter->subgraph_id(source_subgraph);
    Subgraph* dest_graph = interpreter->subgraph_id(dest_subgraph);
    std::vector<int> dest_tensor_indices; 
    /* This method deprecated since (628b2)
      In case of XNN delegated subgraph, GetOutputTensorIndices() and 
      GetInputTensorIndices() method is unsafe because XNN graph has multiple XNN delegated
      nodes and they have multiple inputs and outputs which are trivial for copy.

      TfLiteIntArray* source_tensor_idx = source_graph->GetOutputTensorIndices();
      TfLiteIntArray* source_tensor_idx_ = source_graph->GetInputTensorIndices();
      TfLiteIntArray* input_tensor_indices = dest_graph->GetInputTensorIndices();
    */
    std::vector<int> source_tensor_idx;
    std::vector<int> source_tensor_idx_;
    if(merge_tensor == nullptr){
      source_tensor_idx = source_graph->inputs();
      source_tensor_idx_ = source_graph->outputs();
    }else{
      source_tensor_idx.push_back(merge_tensor->tensor_idx);
    }
    std::vector<int>::iterator source_itr = source_tensor_idx.end();
    source_tensor_idx.insert(source_itr, source_tensor_idx_.begin(), source_tensor_idx_.end()); 
    std::vector<int> dest_tensor_idx = dest_graph->inputs();
    for(int i=0; i<dest_tensor_idx.size(); ++i){
      for(int j=0; j<source_tensor_idx.size(); ++j){
        #ifdef debug_print
          std::cout << "source : " << source_tensor_idx[j] << " " << " dest "
                    << dest_tensor_idx[i] << "\n";
        #endif
        if(source_tensor_idx[j] == dest_tensor_idx[i])
          dest_tensor_indices.push_back(dest_tensor_idx[i]);
      }
    }
    if(dest_tensor_indices.empty()){
      std::cout << "Output tensor of subgraph [" << source_subgraph << "] cannot"
                << " found a matching input tensor in subgraph ["
                << dest_subgraph << "]\n";
      return kTfLiteError;
    }
    for(int i=0; i<dest_tensor_indices.size(); ++i){
      #ifdef debug_print
        std::cout << "Copy tensor " << dest_tensor_indices[i] << "\n";
      #endif
      TfLiteTensor* source_tensor = nullptr;
      if(merge_tensor != nullptr){
        source_tensor = merge_tensor->tensor;
      }else{
        source_tensor = source_graph->tensor(dest_tensor_indices[i]);
      }
      TfLiteTensor* dest_tensor = dest_graph->tensor(dest_tensor_indices[i]);
      size_t source_byte_size = source_tensor->bytes;
      size_t dest_byte_size = dest_tensor->bytes;
      // if (source_byte_size != dest_byte_size) {
      //   std::cout << "Source tensor[" << dest_tensor_indices[i] << "] size " 
      //             << static_cast<int>(source_byte_size) << " and Dest tensor["
      //             << dest_tensor_indices[i] << "] size "
      //             << static_cast<int>(dest_byte_size) << " missmatch!" 
      //             << "\n";
      //   return kTfLiteError;
      // }
      // PrintTensorSerial(*dest_tensor);
      
      if(source_tensor->type == kTfLiteFloat32 && dest_tensor->type == kTfLiteFloat32){
        auto data_source = (float*)source_tensor->data.data;
        auto data_dest = (float*)dest_tensor->data.data;
        memcpy(data_dest, data_source, dest_byte_size);
        // dest_tensor->data.data = source_tensor->data.data;
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
  int source_graph_id = prev_subgraph_id;
  int dest_graph_id = subgraph->GetGraphid();
  #ifdef debug_print
    std::cout << "[Main] Copy sub " << source_graph_id << " to sub " << dest_graph_id << " \n";
  #endif debug_print
  if (connect(source_graph_id, dest_graph_id) != kTfLiteOk) {
    std::cout << "Subgraph intermediate data copy failed"
              << "\n";
    return kTfLiteError;
  }
  // std::cout << "copy done"
  return kTfLiteOk;
}

// SOURCE TENSOR AND DEST TENSOR MUST BE IN SAME PRECISION...
// TODO : Quantization, height partitioning aware data copy
// IMPORTANT : There are some cases that input, output tensor indices are not same.
//             Be carefull to use this function in those cases.
TfLiteStatus TfLiteRuntime::CopyIntermediateDataIfNeeded(Subgraph* sub_subgraph
                                              , Subgraph* main_subgraph
                                              , TfLiteMergeTensor* merge_tensor) {
  auto connect = [&](Subgraph* source_subgraph, Subgraph* dest_subgraph) {
    std::vector<int> dest_tensor_indices; 
    std::vector<int> source_tensor_idx;
    std::vector<int> source_tensor_idx_;
    if(merge_tensor == nullptr){
      source_tensor_idx = source_subgraph->inputs();
      source_tensor_idx_ = source_subgraph->outputs();
    }else{
      source_tensor_idx.push_back(merge_tensor->tensor_idx);
    }
    std::vector<int>::iterator source_itr = source_tensor_idx.end();
    source_tensor_idx.insert(source_itr, source_tensor_idx_.begin(), source_tensor_idx_.end()); 
    std::vector<int> dest_tensor_idx = dest_subgraph->inputs();
    for(int i=0; i<dest_tensor_idx.size(); ++i){
      for(int j=0; j<source_tensor_idx.size(); ++j){
        #ifdef debug_print
          std::cout << "source : " << source_tensor_idx[j] << " " << " dest "
                    << dest_tensor_idx[i] << "\n";
        #endif
        if(source_tensor_idx[j] == dest_tensor_idx[i]){
          dest_tensor_indices.push_back(dest_tensor_idx[i]);
        }
      }
    }
    if(dest_tensor_indices.empty()){
      std::cout << "Output tensor of subgraph [" << source_subgraph << "] cannot"
                << " found a matching input tensor in subgraph ["
                << dest_subgraph << "]\n";
      return kTfLiteError;
    }
    
    
    // int source_tensor_idx = source_subgraph->inputs()[0];
    // int input_tensor_idx = dest_subgraph->GetFirstInputTensorIndex();
    for(int copy_tensor_idx=0; copy_tensor_idx<dest_tensor_indices.size(); ++copy_tensor_idx){
      #ifdef debug_print
        std::cout << "Merge tensor " << dest_tensor_indices[copy_tensor_idx] << "\n";
      #endif
      TfLiteTensor* source_tensor = nullptr;
      if(merge_tensor != nullptr){
        source_tensor = merge_tensor->tensor;
      }else{
        source_tensor = source_subgraph->tensor(dest_tensor_indices[copy_tensor_idx]);
      }
      TfLiteTensor* dest_tensor = dest_subgraph->tensor(dest_tensor_indices[copy_tensor_idx]);
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
        // Maybe consider memory footprint.
        auto data_dest = (float*)dest_tensor->data.data;
        auto data_source = (float*)source_tensor->data.data;
        #ifdef debug_print
          std::cout << "source: " << source_data_size << " "
                    << "dest: " << dest_data_size <<  "\n";
        #endif
        int offset = source_data_size - dest_data_size;
        memcpy(data_dest, data_source + offset, dest_data_size*(sizeof(float)));
        #ifdef debug_print
          std::cout << "Copied intermediate data from main graph" << "\n";
        #endif
      }
    }
    return kTfLiteOk;
  };
  
  // std::cout << "copy from subgraph " << max_precision_subgraph_->GetGraphid() << "\n";
  // std::cout << "copy to subgraph " << min_precision_subgraph_->GetGraphid() << "\n";
  if (main_subgraph != nullptr) {  // Need to copy output from previous graph.
    if (connect(main_subgraph, sub_subgraph) != kTfLiteOk) {
      std::cout << "Subgraph intermediate data copy failed"
                << "\n";
      return kTfLiteError;
    }
  } else {  // if nulltpr returned
    return kTfLiteOk;
  }
  return kTfLiteOk;
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
            if(data > 0.1){ // threshhold
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
      // std::cout << "data : " << data << "\n";
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

void TfLiteRuntime::FeedDummyInputToTensor(TfLiteTensor* tensor){
  if(tensor == nullptr){
    std::cout << "FeedDummyInputToTensor Error" << "\n";
    return;
  }
  int data_size = 1;
  int dim_size = tensor->dims->size;
  for(int i=0; i<dim_size; ++i){
    data_size *= tensor->dims->data[i];
  }
  std::mt19937_64 engine((unsigned int)time(NULL)); 
  std::uniform_real_distribution<double> distribution(0, 1.0);       // 생성 범위
  auto generator = std::bind(distribution, engine);
  float* buffer = new float[data_size];
  for(int i=0; i<data_size; ++i)
    buffer[i] = float(generator());
  memcpy((float*)tensor->data.data, buffer, data_size * sizeof(float));
  // printf("\n");
}

}  // namespace tflite