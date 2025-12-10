#include <iostream>
#include <stdlib.h>
#include <string>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/mat.hpp>
#include <arm_neon.h>
#include <papi.h>

using namespace std;
using namespace cv;

#define NUM_THREADS 4

struct args{
    Mat *source;
    Mat *gray;
    Mat *dest;
    int startIndex;
    int endIndex;
    uint8_t *status;
};

void* graySobel(void*);
void handle_error(int);
pthread_barrier_t barrierA, barrierB, barrierC;

void handle_error (int retval)
{
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}

int main(int argc, char **argv){
    //ensure number of arguments are correct
    if (argc != 2){
        cout << "Error: Invalid input" << endl;
        return 0;
    }


    // PAPI counter INIT
    int retval, EventSet = PAPI_NULL;
    long long start_cycles, end_cycles, start_usec, end_usec;

    retval =  PAPI_library_init(PAPI_VER_CURRENT);   
    if (retval != PAPI_VER_CURRENT)
	handle_error(retval);
			
    string videoPath = argv[1];
    VideoCapture cap(videoPath);

    if (!cap.isOpened()){ 
        perror("Error: Could not open video file.");
        return 0;
    }
    pthread_t threads[NUM_THREADS];
    struct args thr_args[NUM_THREADS];
    Mat src, dest, gray;
    uint8_t Mat_init = 0, pthread_init = 0, status = 0;
    pthread_barrier_init(&barrierA, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&barrierB, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&barrierC, NULL, NUM_THREADS);


    long long frameCount = 0;
    long long start_us   = PAPI_get_real_usec();

    /* PAPI COUNTER START */
    /* Gets the starting time in clock cycles */
    start_cycles = PAPI_get_virt_cyc();
			
    /* Gets the starting time in microseconds */
    start_usec = PAPI_get_real_usec();

    /*Create an EventSet */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK)
	handle_error(retval);


    while (cap.read(src)){

        frameCount++;


        //flip(src, src, -1);
        if (!Mat_init){
            dest.create(src.rows, src.cols, CV_8UC1);
            gray.create(src.rows, src.cols, CV_8UC1);
            Mat_init = 1;
        }
        if (!pthread_init){
            // create pthread + arguments
            for (int i = 0; i < NUM_THREADS; i++){
                thr_args[i].source = &src;
                thr_args[i].dest = &dest;
                thr_args[i].gray = &gray;
                thr_args[i].startIndex = src.rows*i/NUM_THREADS;
                thr_args[i].endIndex = src.rows*(i+1)/NUM_THREADS;
                thr_args[i].status = &status;
                pthread_create(&threads[i], NULL, graySobel, (void *) &thr_args[i]);
            }
            pthread_init = 1;
        }
        // ready
        status = 1;
        pthread_barrier_wait(&barrierB);
        pthread_barrier_wait(&barrierA);
        namedWindow("sImage", WINDOW_NORMAL);
        imshow("sImage", dest);
        waitKey(1);
    }
    status = 0;
    pthread_barrier_wait(&barrierB);

    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }


    /* Gets the ending time in clock cycles */
    end_cycles = PAPI_get_virt_cyc();
			
    /* Gets the ending time in microseconds */
    end_usec = PAPI_get_real_usec();

    // Print Ending times
    printf("Virtual clock cycles: %lld\n", end_cycles - start_cycles);
    printf("Real clock time in microseconds: %lld\n", end_usec - start_usec);

    /* Gets the ending time in microseconds */
    long long end_us = PAPI_get_real_usec();

    long long total_us = end_us - start_us;

    if (frameCount > 0) {
        double total_s  = total_us / 1e6;
        double total_ms = total_us / 1000.0;
        double fps      = frameCount / total_s;

        cout << "Frames processed: " << frameCount << endl;
        cout << "Average FPS: " << fps << endl;
    } else {
        cout << "No frames processed (check video path)." << endl;
    }


    /* Executes if all low-level PAPI
    function calls returned PAPI_OK */
    printf("\033[0;32m\n\nPASSED\n\033[0m");



    return 0;
}

void* graySobel(void *arg){
    while(true){
        pthread_barrier_wait(&barrierB);
        struct args *arguments = static_cast<struct args*>(arg);
        if (!(*(arguments->status))){
            pthread_exit(0);
        }

        vector<Mat> channels;
        split(*(arguments->source), channels);
        uint8_t *rowB, *rowG, *rowR;
        uint8x8_t bByte, rByte, gByte, grayByte;
        uint16x8_t bU16, rU16, gU16, bU16_scaled, rU16_scaled, gU16_scaled;

        // grayscale
        for (int r = arguments->startIndex; r < arguments->endIndex; r++){ 
            rowB = channels[0].ptr<uint8_t>(r);
            rowG = channels[1].ptr<uint8_t>(r);
            rowR = channels[2].ptr<uint8_t>(r);
            uchar *gCurr = arguments->gray->ptr<uchar>(r);
            for (int c = 0; c < arguments->source->cols; c+=8){
                if (arguments->source->cols - c < 8){
                    c = arguments->source->cols - 8;
                }
                bByte = vld1_u8(rowB + c);
                rByte = vld1_u8(rowR + c);
                gByte = vld1_u8(rowG + c);

                // change size to uint16x8_t
                bU16 = vmovl_u8(bByte);
                gU16 = vmovl_u8(gByte);
                rU16 = vmovl_u8(rByte);

                bU16_scaled = vshrq_n_u16(vmulq_n_u16(bU16, 18), 8);
                gU16_scaled = vshrq_n_u16(vmulq_n_u16(gU16, 183), 8);
                rU16_scaled = vshrq_n_u16(vmulq_n_u16(rU16, 54), 8);

                // convert to original 8x8 vector format
                bByte = vqmovn_u16(bU16_scaled);
                gByte = vqmovn_u16(gU16_scaled);
                rByte = vqmovn_u16(rU16_scaled);

                // add up values + store
                grayByte = vadd_u8(vadd_u8(bByte, gByte), rByte);
                vst1_u8(gCurr+c, grayByte);

            }
        }
        pthread_barrier_wait(&barrierC);
        
        // sobel filter
        /*
        IMPLEMENTATION:

            Set Edge Column to Greyscale
            Check for last row if so move backwards // NOT WORKING: gives white line
            Grab 8 vectors at a time starting at seconnd column in
            changes size then scales
            calculate Gx and Gy
            Sum then size change and then store
            repeat until last case

            When reach the end, push vector back like grayscale and sobel


        */
        
        uint8_t *rowT, *rowM, *rowL, *rowS;
        uint8x8_t tByte, g11Byte, g12Byte, g13Byte, g21Byte, g23Byte, g31Byte, g32Byte, g33Byte;
        int16x8_t t16, x16, y16, g11, g12, g13, g21, g23, g31, g32, g33, g11_scaled, g12_scaled, 
                 g21_scaled, g23_scaled, g31_scaled, g32_scaled, g33_scaled;

        for (int r = arguments->startIndex; r < arguments->endIndex; r++){
            if ((r-1) < 0){
                rowT = NULL;
            }else{
                rowT = arguments->gray->ptr<uint8_t>(r-1);
            }
            
            if ((r+1) > arguments->gray->rows-1){ 
                rowL = NULL;
            }else{ 
                rowL = arguments->gray->ptr<uint8_t>(r+1);
            }

            rowM = arguments->gray->ptr<uint8_t>(r);
            rowS = arguments->dest->ptr<uint8_t>(r);
            
            // Set Border Column to Greyscale
            rowS[0] = rowM[0];
            rowS[arguments->source->cols] = rowM[arguments->source->cols];
            
            for (int c = 1; c < arguments->source->cols - 1; c+=8){

                // video is 2159 pixels  
                if (arguments->source->cols - c < 8){
                        c = arguments->source->cols - 9;
                }
                
                // Account for first & last row
                // EDIT: continue function creates a row of all 0s at top and bottom, therefore we opt for 
                if (rowT == NULL || rowL == NULL){
                    vst1_u8(rowS + c, vld1_u8(rowM + c));
                }else{
                    g11Byte = vld1_u8(rowT + c - 1);
                    g12Byte = vld1_u8(rowT + c);
                    g13Byte = vld1_u8(rowT + c + 1);
                    g21Byte = vld1_u8(rowM + c - 1);
                    g23Byte = vld1_u8(rowM + c + 1);
                    g31Byte = vld1_u8(rowL + c - 1);
                    g32Byte = vld1_u8(rowL + c);
                    g33Byte = vld1_u8(rowL + c + 1);


                    // change size to uint16x8_t then to int16x8_t
                    g11 = vreinterpretq_s16_u16(vmovl_u8(g11Byte));
                    g12 = vreinterpretq_s16_u16(vmovl_u8(g12Byte));
                    g13 = vreinterpretq_s16_u16(vmovl_u8(g13Byte));
                    g21 = vreinterpretq_s16_u16(vmovl_u8(g21Byte));
                    g23 = vreinterpretq_s16_u16(vmovl_u8(g23Byte));
                    g31 = vreinterpretq_s16_u16(vmovl_u8(g31Byte));
                    g32 = vreinterpretq_s16_u16(vmovl_u8(g32Byte));
                    g33 = vreinterpretq_s16_u16(vmovl_u8(g33Byte));

                    
                    // scale by 2 or -1 for kernels
                    // doing it this way is faster than multq
                    g11_scaled = vnegq_s16(g11);
                    g12_scaled = vshlq_n_s16(g12, 1);
                    g21_scaled = vnegq_s16(vshlq_n_s16(g21, 1));
                    g23_scaled = vshlq_n_s16(g23, 1);
                    g31_scaled = vnegq_s16(g31);
                    g32_scaled = vnegq_s16(vshlq_n_s16(g32, 1));
                    g33_scaled = vnegq_s16(g33);

                    // Calculate Gx and Gy and total
                    x16 = vaddq_s16(g11_scaled, vaddq_s16(g13, vaddq_s16(g21_scaled, vaddq_s16(g23_scaled, vaddq_s16(g31_scaled, g33)))));
                    y16 = vaddq_s16(g11, vaddq_s16(g12_scaled, vaddq_s16(g13, vaddq_s16(g31_scaled, vaddq_s16(g32_scaled, g33_scaled)))));
                    t16 = vaddq_s16(vabsq_s16(x16), vabsq_s16(y16));

                    // Convert back to original 8x8, vqmovn handles overflow 
                    // Potential error: does vreinterpretq handle negative nums?, would it even need to becos of vabsq or vaddq?
                    tByte = vqmovn_u16(vreinterpretq_u16_s16(t16));
                    vst1_u8(rowS + c, tByte);
                }
            }
        }
    pthread_barrier_wait(&barrierA);
    }
}
