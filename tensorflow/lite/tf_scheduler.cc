#include "tensorflow/lite/tf_scheduler.h"

namespace tflite{

TfScheduler::TfScheduler() {};

TfScheduler::TfScheduler(const char* uds_file_name) {
  // delete if sock file already exists.
  if(access(uds_file_name, F_OK) == 0)
    unlink(uds_file_name);
  
  scheduler_fd = socket(PF_FILE, SOCK_DGRAM, 0);
  if(scheduler_fd == -1){
    std::cout << "Socker create ERROR" << "\n";
    exit(-1);
  }
  addr_size = sizeof(runtime_addr);

  memset(&scheduler_addr, 0, sizeof(scheduler_addr));
  scheduler_addr.sun_family = AF_UNIX;
  strcpy(scheduler_addr.sun_path, uds_file_name);

  if(bind(scheduler_fd, (struct sockaddr*)&scheduler_addr, sizeof(scheduler_addr))
      == -1){
    std::cout << "Socket bind ERROR" << "\n";
    exit(-1);
  }
  std::cout << "Scheduler initializaing done" << "\n";
};

void TfScheduler::Work(){
  while(1){
    tf_packet recv_packet;
    memset(&recv_packet, 0, sizeof(tf_packet));
    if(recvfrom(scheduler_fd, &recv_packet, sizeof(tf_packet), 0,
                (struct sockaddr*)&runtime_addr, (socklen_t*)&addr_size) == -1){
      std::cout << "Receive failed" << "\n";
      return;
    }
    std::cout << "Recieved packet from runtime " << recv_packet.runtime_id << "\n";
    switch (recv_packet.runtime_current_state)
    {
    case RuntimeState::INITIALIZE :{
      for(auto runtime : runtimes){
        if(runtime->id == recv_packet.runtime_id){
          std::cout << "Runtime " << runtime->id << " already registered." << "\n"; 
          break;
        }
      }
      // initializing new_runtime
      runtime_* new_runtime = new runtime_;
      new_runtime->id = runtimes_created;
      runtimes_created++;
      
      new_runtime->addr.sun_family = runtime_addr.sun_family;
      strcpy(new_runtime->addr.sun_path, runtime_addr.sun_path);
      
      tf_packet new_packet;
      memset(&new_packet, 0, sizeof(tf_packet));
      new_packet.runtime_id = new_runtime->id;
      new_packet.runtime_next_state = RuntimeState::NEED_PROFILE;

      if(sendto(scheduler_fd, (void*)&new_packet, sizeof(tf_packet), 0,
               (struct sockaddr*)&runtime_addr, sizeof(runtime_addr)) == -1){
        std::cout << "Sending packet to " << new_runtime->id << " Failed" << "\n";
        std::cout << "sock : " << runtime_addr.sun_path  << " " << runtime_addr.sun_family << "\n";
        printf("errno : %d \n", errno);
        return;
      }

      runtimes.push_back(new_runtime);
      std::cout << "Registered new runtime " << new_runtime->id << " \n";
      continue;
    }
    case RuntimeState::NEED_PROFILE :
      /* code */
      break;
    case RuntimeState::SUBGRAPH_CREATE :
      /* code */
      break;
    case RuntimeState::INVOKE_ :
      /* code */
      break;
    
    default:
      break;
    }
  }
}

TfScheduler::~TfScheduler() {};

}

