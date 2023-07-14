#Run YOLOv4 on CPU [TODO]
#bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/yolov4-tiny-416.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt

#---------------------------------------------------------------------------------------------------------------

#Run SSD on CPU
#bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt

# (Hexagon)
#bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt  --delegate='hexagon' 

# (XNNPACK) [SUCCESS]
#bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt  --delegate='xnnpack'

# (NNAPi)
#bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt  --delegate='nnapi'

# (GPU) [FAILED] [opt --copt -DCL_DELEGATE_NO_GL]
#external/opencl_headers/CL/cl_version.h:34:104: note: #pragma message: cl_version.h: CL_TARGET_OPENCL_VERSION is not defined. Defaulting to 220 (OpenCL 2.2)
#bazel run -c opt --copt -DCL_DELEGATE_NO_GL  --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt  --delegate='gpu'

# (GPU) [FAILED] [opt --copt -DCL_DELEGATE_ALLOW_GL]  [linking error]
#bazel run -c opt  --copt -DCL_DELEGATE_ALLOW_GL   -- //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt  --delegate='gpu'

# (GPU) [experiment] [TODO] [Not use toolkit's compile-level delegation]
bazel run -c opt  -- //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/ssd_mobile.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/mscoco_complete_label_map.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt  --delegate='gpu'
