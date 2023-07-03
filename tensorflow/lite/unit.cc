#include "unit.h"
#include "algorithm"
#define GPUONLY
// #define CPUONLY
//#define MULTITHREAD
//#define quantize
//#define MONITORING
//#define mnist
//#define catdog
//#define imagenet

#define yolo  // Y/N

#ifdef yolo
#define SEQ 100   // debugging
#define OUT_SEQ 1
#endif

#ifndef yolo
#define SEQ 1 //1000   ---> 4개부터 20000, 5개 15000, 6개 10000... 각 케이스마다 하루 정도 걸림.
#define OUT_SEQ 1
#endif

std::mutex mtx_lock;

namespace tflite
{

// UnitCPU
UnitCPU::UnitCPU() : name("NONE"), interpreterCPU(nullptr){}

UnitCPU::UnitCPU(UnitType eType_, std::unique_ptr<tflite::Interpreter>* interpreter) 
            : eType(eType_), interpreterCPU(std::move(interpreter)) {}

#ifdef MULTITHREAD
TfLiteStatus UnitCPU::Invoke(UnitType eType, std::mutex& mtx_lock,
                            std::mutex& mtx_lock_,
                            std::mutex& mtx_lock_timing,
                            std::mutex& mtx_lock_debug,
                            std::condition_variable& Ucontroller,
                            std::condition_variable& Outcontroller,
                            std::queue<SharedContext*>* qSharedData,
                            int* C_Counter, int* G_Counter) { 
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            //std::cout << "CPU " << *C_Counter << "\n";
            #ifdef catdog
            for (int i=0; i < 300; ++i) {
                for(int j=0; j < 300; j++){
                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*300 + j*3] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[0])/255.0;    

                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 1] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[1])/255.0;

                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 2] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[2])/255.0;
                }
            }
            #endif

            #ifdef mnist
            for (int i=0; i<Image_x; i++){
                for (int j=0; j<Image_y; j++){
                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                     ((float)input[k].at<uchar>(i, j)/255.0);          
                }
            } 
            #endif 
            // Run inference
            if(interpreterCPU->get()->Invoke(eType, mtx_lock, mtx_lock_, mtx_lock_debug,
                                                Ucontroller, qSharedData) 
                                            != kTfLiteOk){
                return kTfLiteError;
            }
            mtx_lock_timing.lock();
            *C_Counter += 1;
            if(*G_Counter >= * C_Counter){
                std::unique_lock<std::mutex>lock (mtx_lock);
                Outcontroller.notify_one();
            }
            mtx_lock_timing.unlock();
        }
    }
    std::cout << "\n" << "CPU All Jobs Done" << "\n";
    return kTfLiteOk;
}
#endif

#ifndef MULTITHREAD
TfLiteStatus UnitCPU::Invoke(UnitType eType, std::mutex& mtx_lock,
                            std::mutex& mtx_lock_,
                            std::mutex& mtx_lock_timing,
                            std::mutex& mtx_lock_debug,
                            std::condition_variable& Ucontroller,
                            std::condition_variable& Outcontroller,
                            std::queue<SharedContext*>* qSharedData,
                            int* C_Counter, int* G_Counter ) { 
    double time = 0;
    struct timespec begin, end;
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            std::cout << "CPU " << *C_Counter << "\n";
            #ifdef yolo
            for (int i=0; i<416; i++){
                for (int j=0; j<416; j++){
                        interpreterCPU->get()->typed_input_tensor<float>(0)[i*416 + j*3] = \
                        ((float)input[0].at<cv::Vec3b>(i, j)[0])/255.0;
                        // printf("%0.6f ",(float)input[0].at<cv::Vec3b>(i, j)[0]/255.0);     
                        interpreterCPU->get()->typed_input_tensor<float>(0)[i*416 + j*3+1] = \
                        ((float)input[0].at<cv::Vec3b>(i, j)[1])/255.0;
                        interpreterCPU->get()->typed_input_tensor<float>(0)[i*416 + j*3+2] = \
                        ((float)input[0].at<cv::Vec3b>(i, j)[2])/255.0;
                        // printf("%0.6f\n",(float)input[0].at<cv::Vec3b>(i, j)[0]);
                        // printf("%0.6f\n",(float)input[0].at<cv::Vec3b>(i, j)[1]);
                        // printf("%0.6f\n",(float)input[0].at<cv::Vec3b>(i, j)[2]);
                }
                // printf("\n");
            } 

            // auto input_pointer = interpreterCPU->get()->typed_input_tensor<float>(0);
            // memcpy(input_pointer, input[0].data, input[0].total() * input[0].elemSize());
           
            #endif
            #ifdef catdog
            for (int i=0; i < 300; ++i) {
                for(int j=0; j< 300; j++){
                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*300 + j*3] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[0])/255.0;    

                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 1] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[1])/255.0;

                    interpreterCPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 2] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[2])/255.0;
                }
            }
            #endif
            #ifdef mnist
                #ifdef quantize
                for (int i=0; i<Image_x; i++){
                    for (int j=0; j<Image_y; j++){
                        interpreterCPU->get()->typed_input_tensor<int8_t>(0)[i*28 + j] = 100;
                        //static_cast<int8_t>((float)input[k].at<uchar>(i, j)/255.0);          
                    }
                } 
                #endif
                
                #ifndef quantize
                for (int i=0; i<Image_x; i++){
                    for (int j=0; j<Image_y; j++){
                        interpreterCPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                        ((float)input[k].at<uchar>(i, j)/255.0);          
                    }
                } 
                #endif
            #endif
            // Run inference
            clock_gettime(CLOCK_MONOTONIC, &begin);
            if(interpreterCPU->get()->Invoke(eType, mtx_lock, mtx_lock_, mtx_lock_debug,
                                                Ucontroller, qSharedData) 
                                            != kTfLiteOk){
                return kTfLiteError;
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            *C_Counter += 1;
            double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
            //printf("time : %.6fs \n", temp_time);
            time += temp_time;
            #ifdef yolo
                // interpreterCPU->get()->PrintOutputTensor(eType);            
                // interpreterCPU->get()->PrintInputTensor(eType);
            #endif
            #ifdef MONITORING
            // for (int i =0; i<1001; i++){
            //     float value = interpreterCPU->get()->typed_output_tensor<float>(1)[i];
            //     if(value > 0.5)
            //         printf("label : %d, pre : %0.5f \n", i, value);
            // }
            // PrintInterpreterState(interpreterCPU->get());
            std::cout << "\n";
            #endif
            if(!(*C_Counter % 1000))
                std::cout << "Progress " << int(*C_Counter/1000) << "% \n";
        }
    }
    time = time / (SEQ * OUT_SEQ);
    printf("Average elepsed time : %.6fs \n", time);
    std::cout << "\n" << "CPU All Jobs Done" << "\n";
    return kTfLiteOk;
}
#endif

Interpreter* UnitCPU::GetInterpreter(){return interpreterCPU->get();}

void UnitCPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}

UnitType UnitCPU::GetUnitType(){
    return eType;
}

// UnitGPU
UnitGPU::UnitGPU() : name("NONE"), interpreterGPU(nullptr){}

UnitGPU::UnitGPU(UnitType eType_, std::unique_ptr<tflite::Interpreter>* interpreter) 
            : eType(eType_), interpreterGPU(std::move(interpreter)) {}

#ifdef MULTITHREAD
TfLiteStatus UnitGPU::Invoke(UnitType eType, std::mutex& mtx_lock, 
                            std::mutex& mtx_lock_,
                            std::mutex& mtx_lock_timing,
                            std::mutex& mtx_lock_debug,
                            std::condition_variable& Ucontroller,
                            std::condition_variable& Outcontroller,
                            std::queue<SharedContext*>* qSharedData,
                            int* C_Counter, int* G_Counter) {
    std::cout << "Starting GPU Job" << "\n";
    struct timespec begin, end;
    double time = 0;
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            mtx_lock_timing.lock();
            //std::cout << "GPU " << *G_Counter << "\n";
            #ifdef catdog
            for (int i=0; i < 300; ++i) {
                for(int j=0; j< 300; j++){
                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*300 + j*3] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[0])/255.0;    

                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 1] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[1])/255.0;

                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 2] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[2])/255.0;
                }
            }
            #endif

            #ifdef mnist
            for (int i=0; i<Image_x; i++){
                for (int j=0; j<Image_y; j++){
                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                     ((float)input[k].at<uchar>(i, j)/255.0);          
                }
            } 
            #endif
            // Run inference
            clock_gettime(CLOCK_MONOTONIC, &begin);
            if(interpreterGPU->get()->Invoke(eType, mtx_lock, mtx_lock_, mtx_lock_debug,
                                                Ucontroller, qSharedData) 
                                            != kTfLiteOk){
                return kTfLiteError;
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            *G_Counter += 1;
            double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
            time += temp_time;
            if(*G_Counter > *C_Counter){
                std::unique_lock<std::mutex>lock (mtx_lock);
                mtx_lock_timing.unlock();
                Outcontroller.wait(lock);
            }else{
                mtx_lock_timing.unlock();
            }
            //printf("time : %.6fs \n", temp_time);
            #ifdef MONITORING
            for (int i =0; i<10; i++){
                printf("%0.5f", interpreterGPU->get()->typed_output_tensor<float>(0)[i] );
                std:: cout << " ";
            }
            std::cout << "\n";
            #endif
            //std::cout << *G_Counter << "\n";
            if(!(*G_Counter % 1000))
                std::cout << "Progress " << int(*G_Counter/1000) << "% \n";
        }
    }
    //std::cout << time << "\n";
    time = time / (SEQ * OUT_SEQ);
    printf("Average elepsed time : %.6fs \n", time);
    std::cout << "\n";
    std::cout << "GPU All Jobs done" << "\n";
    return kTfLiteOk;
}
#endif



std::vector<double> b_delegation_optimizer; // HOON : vector for delegation optimizing test
extern bool print_flag; // slave 




#ifndef MULTITHREAD
TfLiteStatus UnitGPU::Invoke(UnitType eType, std::mutex& mtx_lock, 
                            std::mutex& mtx_lock_,
                            std::mutex& mtx_lock_timing,
                            std::mutex& mtx_lock_debug,
                            std::condition_variable& Ucontroller,
                            std::condition_variable& Outcontroller,
                            std::queue<SharedContext*>* qSharedData,
                            int* C_Counter, int* G_Counter) {
    std::cout << "Starting GPU Job" << "\n";
    double time = 0;
    struct timespec begin, end;
    double sum_average = 0;
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            //std::cout << "GPU " << *G_Counter << "\n";

            #ifdef yolo  // same code as catdog //HOON
            for (int i=0; i<416; i++){
                for (int j=0; j<416; j++){
                        interpreterGPU->get()->typed_input_tensor<float>(0)[i*416 + j*3] = \
                        ((float)input[0].at<cv::Vec3b>(i, j)[0])/255.0;
                        // printf("%0.6f ",(float)input[0].at<cv::Vec3b>(i, j)[0]/255.0);     
                        interpreterGPU->get()->typed_input_tensor<float>(0)[i*416 + j*3+1] = \
                        ((float)input[0].at<cv::Vec3b>(i, j)[1])/255.0;
                        interpreterGPU->get()->typed_input_tensor<float>(0)[i*416 + j*3+2] = \
                        ((float)input[0].at<cv::Vec3b>(i, j)[2])/255.0;
                        // printf("%0.6f\n",(float)input[0].at<cv::Vec3b>(i, j)[0]);
                        // printf("%0.6f\n",(float)input[0].at<cv::Vec3b>(i, j)[1]);
                        // printf("%0.6f\n",(float)input[0].at<cv::Vec3b>(i, j)[2]);
                }
                // printf("\n");
            } 
            // auto *input_pointer = interpreterGPU->get()->typed_input_tensor<float>(0);
            // memcpy(input_pointer, input[0].data, input[0].total() * input[0].elemSize());
            #endif

            #ifdef catdog
            for (int i=0; i < 300; ++i) {
                for(int j=0; j < 300; j++){
                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*300 + j*3] = \   //image size : 300*300
                     ((float)input[0].at<cv::Vec3b>(i, j)[0])/255.0;    

                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 1] = \  // cause of RGB
                     ((float)input[0].at<cv::Vec3b>(i, j)[1])/255.0;

                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*300 + j*3 + 2] = \
                     ((float)input[0].at<cv::Vec3b>(i, j)[2])/255.0;
                }
            }
            #endif

            #ifdef mnist
            for (int i=0; i<Image_x; i++){
                for (int j=0; j<Image_y; j++){
                    interpreterGPU->get()->typed_input_tensor<float>(0)[i*28 + j] = \
                     ((float)input[k].at<uchar>(i, j)/255.0);          // HOON input[0]
                }
            } 
            #endif

            // Run inference
            clock_gettime(CLOCK_MONOTONIC, &begin);
            // HOON : add extra parameter to test delegation optimizing? TODO
            if(interpreterGPU->get()->Invoke(eType, mtx_lock, mtx_lock_, mtx_lock_debug,
                                                Ucontroller, qSharedData) 
                                            != kTfLiteOk){
                return kTfLiteError;
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            //printf("Begin Timestamp %.6f \n", (begin.tv_sec + (begin.tv_nsec) / 1000000000.0));
            *G_Counter += 1;
            double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
            time += temp_time;
            //printf("time : %.6fs \n", temp_time);
            #ifdef MONITORING
            #ifdef yolo
                for (int i =0; i<1000; i++){
                    float value = interpreterGPU->get()->typed_output_tensor<float>(0)[i];
                    printf("label : %d, pre : %0.5f \n", i, value);
                    //if(value > 0.5)
                    //    printf("label : %d, pre : %0.5f \n", i, value);
                }
                //interpreterGPU->get()->PrintOutputTensor(eType);
            #endif
            #ifdef mnist
                // for (int i =0; i<10; i++){
                //     float value = interpreterGPU->get()->typed_output_tensor_final<float>(0)[i];
                //     printf("label : %d, pre : %0.5f \n", i, value);
                //     if(value > 0.5)
                //         printf("label : %d, pre : %0.5f \n", i, value);
                // }
                interpreterGPU->get()->PrintOutputTensor(eType);
            #endif
            //PrintInterpreterState(interpreterGPU->get());
            std::cout << "\n";
            #endif
            //std::cout << *G_Counter << "\n";

            // <<<<<<<<<<<<<<<<<<<<<<<<<< HOON : TODO >>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // make runtime-softmax function 

            // <<<<<<<<<<<<<<<<<<<<<<<<<< HOON : TODO >>>>>>>>>>>>>>>>>>>>>>>>>>>>
            float max  = 0;
            for (int i =0; i<10; i++){
                float value = interpreterGPU->get()->typed_output_tensor_final<float>(0)[i];
                if (value > max){
                    max = value;
                }
                //printf("label : %d, pre : %0.5f \n", i, value);  ///HOON
            }
            sum_average += max;
            if(!(*G_Counter % 10000))
                // std::cout << "Progress " << int(*G_Counter/10000) << "% \n";
                std::cout << "Progressing... \n";
        }
    }
    std::cout << "All invoke time : " << time << "s" << "\n"; // HOON : This is all-invoke time , not ACCURACY ,,,,, 
    time = time / (SEQ * OUT_SEQ);
    sum_average = sum_average / (SEQ * OUT_SEQ);
    printf("Average invoke time : \033[0;31m%0.6f\033[0m ms \n", time*1000);
    //printf("Average accuracy : %.6f % \n", sum_average*100);
    std::cout << "\n";
    std::cout << "GPU All Jobs done" << "\n";
    b_delegation_optimizer.push_back(time*1000);
    // std::cout << "delegation_optimizer vector size : " << b_delegation_optimizer.size() << std::endl;
    //
    if(print_flag) PrintTest(b_delegation_optimizer);
    //interpreterGPU memory delete
    return kTfLiteOk;
}
#endif

// bool print_flag;

void UnitGPU::PrintTest(std::vector<double> b_delegation_optimizer){
    std::cout << "\033[0;31mLatency for each of cases in delegation optimizing\033[0m : " <<std::endl;
    double min = *min_element(b_delegation_optimizer.begin(), b_delegation_optimizer.end());
    for (int i=0;i< b_delegation_optimizer.size(); i++){
        if(b_delegation_optimizer.at(i) < min + 0.3)   // 0.3 is bias
        {
            printf("\033[0;31m%d case's latency is %0.6fms\033[0m\n",i,b_delegation_optimizer.at(i));
            // std::cout << i << " \033[0;31mcase's latency is :\033[0m " << b_delegation_optimizer.at(i) << "ms" << std::endl;   
        }
        else{
            std::cout << i << " case's latency is : " << b_delegation_optimizer.at(i) << "ms" << std::endl;
        }
    }    // int min_index = min_element(b_delegation_optimizer.begin(), b_delegation_optimizer.end()) - b_delegation_optimizer.begin();

}

Interpreter* UnitGPU::GetInterpreter(){return interpreterGPU->get();}

void UnitGPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}

UnitType UnitGPU::GetUnitType(){
    return eType;
}

} // End of namespace tflite
