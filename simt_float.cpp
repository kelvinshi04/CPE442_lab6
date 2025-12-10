#include <iostream>
#include <vector>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <CL/cl.h>
#include <papi.h>

using namespace std;
using namespace cv;


void handle_error(int);
void handle_error (int retval)
{
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}



// OpenCL kernel: manual grayscale (float) + Sobel
const char* graySobelKernelSrcFloat = R"CLC(
inline float read_gray_float(__global const uchar* src,
                             int ix, int iy,
                             int width, int height,
                             int srcStep)
{
    // clamp coords
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (ix >= width)  ix = width  - 1;
    if (iy >= height) iy = height - 1;

    int idx = iy * srcStep + ix * 3;
    uchar b = src[idx + 0];
    uchar g = src[idx + 1];
    uchar r = src[idx + 2];

    // manual grayscale (float)
    // approx BT.709: 0.0722, 0.7152, 0.2126
    return 0.0722f * (float)b +
           0.7152f * (float)g +
           0.2126f * (float)r;
}

__kernel void gray_sobel(
    __global const uchar* src,
    __global uchar* dst,
    const int width,
    const int height,
    const int srcStep,
    const int dstStep)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    float gCenter = read_gray_float(src, x, y, width, height, srcStep);

    // borders: just write grayscale value
    if (x == 0 || x == width-1 || y == 0 || y == height-1) {
        float val = gCenter;
        if (val < 0.0f)   val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        int outIdx = y * dstStep + x;
        dst[outIdx] = (uchar)(val + 0.5f);
        return;
    }

    // 3x3 neighborhood grayscale
    float g11 = read_gray_float(src, x-1, y-1, width, height, srcStep);
    float g12 = read_gray_float(src, x,   y-1, width, height, srcStep);
    float g13 = read_gray_float(src, x+1, y-1, width, height, srcStep);

    float g21 = read_gray_float(src, x-1, y,   width, height, srcStep);
    float g23 = read_gray_float(src, x+1, y,   width, height, srcStep);

    float g31 = read_gray_float(src, x-1, y+1, width, height, srcStep);
    float g32 = read_gray_float(src, x,   y+1, width, height, srcStep);
    float g33 = read_gray_float(src, x+1, y+1, width, height, srcStep);

    // Sobel Gx, Gy (float)
    float Gx = (-g11 + g13)
             + (-2.0f * g21 + 2.0f * g23)
             + (-g31 + g33);

    float Gy = ( g11 + 2.0f * g12 + g13 )
             - ( g31 + 2.0f * g32 + g33 );

    float mag = fabs(Gx) + fabs(Gy);  // L1 norm

    if (mag < 0.0f)   mag = 0.0f;
    if (mag > 255.0f) mag = 255.0f;

    int outIdx = y * dstStep + x;
    dst[outIdx] = (uchar)(mag + 0.5f);
}
)CLC";

static void checkStatus(cl_int status, const char* msg) {
    if (status != CL_SUCCESS) {
        cerr << "OpenCL error (" << status << "): " << msg << endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <video path>" << endl;
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
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return 1;
    }

    // --- OpenCL setup ---
    cl_int status;

    cl_uint numPlatforms = 0;
    checkStatus(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs count");
    if (numPlatforms == 0) {
        cerr << "No OpenCL platforms found." << endl;
        return 1;
    }
    vector<cl_platform_id> platforms(numPlatforms);
    checkStatus(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr), "clGetPlatformIDs");

    cl_platform_id platform = platforms[0];

    cl_device_id device = nullptr;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (status != CL_SUCCESS) {
        cout << "No GPU device, trying CPU..." << endl;
        checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr),
                    "clGetDeviceIDs CPU");
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    checkStatus(status, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
    checkStatus(status, "clCreateCommandQueue");

    const char* srcStrs[] = { graySobelKernelSrcFloat };
    size_t lengths[] = { strlen(graySobelKernelSrcFloat) };
    cl_program program = clCreateProgramWithSource(context, 1, srcStrs, lengths, &status);
    checkStatus(status, "clCreateProgramWithSource");

    status = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        cerr << "Build log:\n" << log.data() << endl;
        checkStatus(status, "clBuildProgram");
    }

    cl_kernel kernel = clCreateKernel(program, "gray_sobel", &status);
    checkStatus(status, "clCreateKernel");

    namedWindow("sImage", WINDOW_NORMAL);

    Mat src, dest;
    bool initialized = false;

    
    /* PAPI COUNTER START */
    /* Gets the starting time in clock cycles */
    start_cycles = PAPI_get_virt_cyc();
			
    /* Gets the starting time in microseconds */
    start_usec = PAPI_get_real_usec();

    /*Create an EventSet */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK)
	handle_error(retval);


    while (cap.read(src)) {
        if (src.empty()) break;

        int width  = src.cols;
        int height = src.rows;

        if (!initialized) {
            dest.create(height, width, CV_8UC1);
            initialized = true;
        }

        if (!src.isContinuous()) src = src.clone();
        if (!dest.isContinuous()) dest = dest.clone();

        int srcStep = static_cast<int>(src.step);
        int dstStep = static_cast<int>(dest.step);

        size_t srcBufSize = srcStep * height;
        size_t dstBufSize = dstStep * height;

        cl_mem srcBuf = clCreateBuffer(context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       srcBufSize,
                                       src.data,
                                       &status);
        checkStatus(status, "clCreateBuffer src");

        cl_mem dstBuf = clCreateBuffer(context,
                                       CL_MEM_WRITE_ONLY,
                                       dstBufSize,
                                       nullptr,
                                       &status);
        checkStatus(status, "clCreateBuffer dst");

        int arg = 0;
        checkStatus(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &srcBuf),  "arg 0");
        checkStatus(clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dstBuf),  "arg 1");
        checkStatus(clSetKernelArg(kernel, arg++, sizeof(int), &width),      "arg 2");
        checkStatus(clSetKernelArg(kernel, arg++, sizeof(int), &height),     "arg 3");
        checkStatus(clSetKernelArg(kernel, arg++, sizeof(int), &srcStep),    "arg 4");
        checkStatus(clSetKernelArg(kernel, arg++, sizeof(int), &dstStep),    "arg 5");

        size_t globalWorkSize[2] = { (size_t)width, (size_t)height };
        checkStatus(clEnqueueNDRangeKernel(queue,
                                           kernel,
                                           2,
                                           nullptr,
                                           globalWorkSize,
                                           nullptr,
                                           0,
                                           nullptr,
                                           nullptr),
                    "clEnqueueNDRangeKernel");

        checkStatus(clEnqueueReadBuffer(queue,
                                        dstBuf,
                                        CL_TRUE,
                                        0,
                                        dstBufSize,
                                        dest.data,
                                        0,
                                        nullptr,
                                        nullptr),
                    "clEnqueueReadBuffer");

        clReleaseMemObject(srcBuf);
        clReleaseMemObject(dstBuf);

        imshow("sImage", dest);
        if (waitKey(1) == 27) break; // ESC
    }





    /* Gets the ending time in clock cycles */
    end_cycles = PAPI_get_virt_cyc();
			
    /* Gets the ending time in microseconds */
    end_usec = PAPI_get_real_usec();

    // Print Ending times
    printf("Virtual clock cycles: %lld\n", end_cycles - start_cycles);
    printf("Real clock time in microseconds: %lld\n", end_usec - start_usec);

    /* Executes if all low-level PAPI
    function calls returned PAPI_OK */
    printf("\033[0;32m\n\nPASSED\n\033[0m");





    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
