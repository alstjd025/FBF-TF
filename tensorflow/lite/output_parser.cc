#include "tensorflow/lite/output_parser.h"


OutputParser::OutputParser(){ 
};

OutputParser::OutputParser(int input_w, int input_h, int anchor_size,
                            int num_class, float conf_threshold){
  input_w_ = input_w;
  input_h_ = input_h;
  anchor_size_ = anchor_size;
  num_class_ = num_class;
  conf_threshhold_ = conf_threshold;
};

bool OutputParser::setOutputTensors(TfLiteTensor* output_tensor){
  // if(output_tensors.size() < 1){
  //   std::cout << "OutputParser: output tensor size < 1 ERROR" << "\n";
  //   return false;
  // }
  output_tensor_ = output_tensor;
};

bool OutputParser::ParseOutput(){
  // output tensor is temporarly 948[1, 10647, 80] now.
  TfLiteTensor* output_tensor = output_tensor_;
  
  // get output tensor dimensions
  int num_dims = output_tensor->dims->size;
  if (num_dims < 3) {
    std::cerr << "Invalid number of dimensions in output tensor!" << std::endl;
    return false;
  }
  int batch_size = 1; // batch size is fixed
  int grid_size = output_tensor->dims->data[0];
  int num_anchors = output_tensor->dims->data[1];
  int num_classes = output_tensor->dims->data[2];
  std::cout << "Num classes : " << num_classes << "\n";
  std::cout << "num anchors : " << num_anchors << "\n";
  if(num_classes != num_class_){
    std::cerr << "Invalid number of classes in output tensor" << "\n";
    return false;
  }

  // check if number of anchors is valid
  // if (num_anchors != ANCHORS_SIZE / 2) {
  //   std::cerr << "Invalid number of anchors in output tensor!" << std::endl;
  //   return detected_objects;
  // }

  // get output tensor data
  auto data = (float*)output_tensor->data.data;
  std::cout << output_tensor->dims->size << "\n";
  for(int i=0; i<num_classes; ++i){
    for(int j=0; j<num_anchors; ++j){
      float conf = *(data+(i * num_classes + j));
      if(conf > 0.00001){
        printf("%0.5f \n", conf);
        DetectedObject obj;
        obj.class_id = i;
        obj.confidence = conf;
        obj.box_idx = j;
        output_objects.push_back(obj);
      }
    }
  }
};

void OutputParser::PrintOutput(){
  std::cout << "Detected " << output_objects.size() << " objects \n";
  for(size_t i=0; i<output_objects.size(); ++i){
    std::cout << "Class [" << output_objects[i].class_id <<"] \n";
    std::cout << "Prediction score [";
    printf("%0.6f] \n", output_objects[i].confidence);
    std::cout << "Box idx [" << output_objects[i].box_idx << "] \n";
    std::cout << "\n";
  }
};


