bazel run -c opt   --   //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval   --model_file=/home/nvidia/FBF-TF-hoon/models/yolov4-tiny-416.tflite    --ground_truth_images_path=/home/nvidia/coco_raw/output/folder/images   --ground_truth_proto=/home/nvidia/coco_raw/output/folder/ground_truth.pb   --model_output_labels=/home/nvidia/coco_raw/labelmap.txt --output_file_path=/home/nvidia/coco_raw/output/result/coco_output.txt


