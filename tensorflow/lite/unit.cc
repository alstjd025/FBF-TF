#include "unit.h"
#define SEQ 1
#define OUT_SEQ 1
//#define MULTITHREAD
#define CPUONLY
#define quantize
//#define MONITORING
#define mnist

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
            if(interpreterCPU->get()->Invoke(eType, mtx_lock, mtx_lock_, Ucontroller, qSharedData) 
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
                            std::condition_variable& Ucontroller,
                            std::condition_variable& Outcontroller,
                            std::queue<SharedContext*>* qSharedData,
                            int* C_Counter, int* G_Counter) { 
    double time = 0;
    struct timespec begin, end;
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            std::cout << "CPU " << *C_Counter << "\n";
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
                
                #endif
            #endif
            // Run inference
            clock_gettime(CLOCK_MONOTONIC, &begin);
            if(interpreterCPU->get()->Invoke(eType, mtx_lock, mtx_lock_, Ucontroller, qSharedData) 
                                            != kTfLiteOk){
                return kTfLiteError;
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            *C_Counter += 1;
            double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
            printf("time : %.6fs \n", temp_time);
            time += temp_time;
            #ifdef MONITORING
            for (int i =0; i<10; i++){
                printf("%0.5f", interpreterCPU->get()->typed_output_tensor<float>(0)[i] );
                std:: cout << " ";
            }
            PrintInterpreterState(interpreterCPU->get());
            std::cout << "\n";
            #endif
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
            if(interpreterGPU->get()->Invoke(eType, mtx_lock, mtx_lock_, Ucontroller, qSharedData) 
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
            printf("time : %.6fs \n", temp_time);
            #ifdef MONITORING
            for (int i =0; i<1; i++){
                printf("%0.5f", interpreterGPU->get()->typed_output_tensor<float>(0)[i] );
                std:: cout << " ";
            }
            std::cout << "\n";
            #endif
            //std::cout << *G_Counter << "\n";
            if(!(*G_Counter % 100))
                std::cout << "Progress " << int(*G_Counter/100) << "% \n";
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

#ifndef MULTITHREAD
TfLiteStatus UnitGPU::Invoke(UnitType eType, std::mutex& mtx_lock, 
                            std::mutex& mtx_lock_,
                            std::mutex& mtx_lock_timing,
                            std::condition_variable& Ucontroller,
                            std::condition_variable& Outcontroller,
                            std::queue<SharedContext*>* qSharedData,
                            int* C_Counter, int* G_Counter) {
    std::cout << "Starting GPU Job" << "\n";
    double time = 0;
    struct timespec begin, end;
    for(int o_loop=0; o_loop<OUT_SEQ; o_loop++){
        for(int k=0; k<SEQ; k++){
            //std::cout << "GPU " << *G_Counter << "\n";
            #ifdef catdog
            for (int i=0; i < 300; ++i) {
                for(int j=0; j < 300; j++){
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
            if(interpreterGPU->get()->Invoke(eType, mtx_lock, mtx_lock_, Ucontroller, qSharedData) 
                                            != kTfLiteOk){
                return kTfLiteError;
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            *G_Counter += 1;
            double temp_time = (end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
            time += temp_time;
            printf("time : %.6fs \n", temp_time);
            #ifdef MONITORING
            for (int i =0; i<1; i++){
                printf("%0.5f", interpreterGPU->get()->typed_output_tensor<float>(0)[i] );
                std:: cout << " ";
            }
            //PrintInterpreterState(interpreterGPU->get());
            std::cout << "\n";
            #endif
            //std::cout << *G_Counter << "\n";
            if(!(*G_Counter % 100))
                std::cout << "Progress " << int(*G_Counter/100) << "% \n";
        }
    }
    std::cout << time << "\n";
    time = time / (SEQ * OUT_SEQ);
    printf("Average elepsed time : %.6fs \n", time);
    std::cout << "\n";
    std::cout << "GPU All Jobs done" << "\n";
    return kTfLiteOk;
}
#endif

Interpreter* UnitGPU::GetInterpreter(){return interpreterGPU->get();}

void UnitGPU::SetInput(std::vector<cv::Mat> input_){
    input = input_;
}

UnitType UnitGPU::GetUnitType(){
    return eType;
}

} // End of namespace tflite
