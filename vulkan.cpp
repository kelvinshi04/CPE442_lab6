#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <vulkan/vulkan.h>
#include <papi.h>

using namespace std;
using namespace cv;

void handle_error(int);
void handle_error (int retval)
{
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}



// Embedded shader code as GLSL strings
const char* grayscaleShaderCode = R"(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout (local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
} pc;

layout(binding = 0, std430) readonly buffer InputBuffer {
    uint8_t data[];
} inputBuf;

layout(binding = 1, std430) writeonly buffer GrayBuffer {
    uint8_t data[];
} grayBuf;

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    
    if (gid.x >= pc.width || gid.y >= pc.height) {
        return;
    }
    
    uint pixelIndex = gid.y * pc.width + gid.x;
    uint rgbIndex = pixelIndex * 3;
    
    uint b = uint(inputBuf.data[rgbIndex]);
    uint g = uint(inputBuf.data[rgbIndex + 1]);
    uint r = uint(inputBuf.data[rgbIndex + 2]);
    
    uint gray = (b * 18 + g * 183 + r * 54) / 256;
    
    grayBuf.data[pixelIndex] = uint8_t(gray);
}
)";

const char* sobelShaderCode = R"(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout (local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PushConstants {
    uint width;
    uint height;
} pc;

layout(binding = 1, std430) readonly buffer GrayBuffer {
    uint8_t data[];
} grayBuf;

layout(binding = 2, std430) writeonly buffer OutputBuffer {
    uint8_t data[];
} outputBuf;

void main() {
    uvec2 gid = gl_GlobalInvocationID.xy;
    
    if (gid.x >= pc.width || gid.y >= pc.height) {
        return;
    }
    
    uint x = gid.x;
    uint y = gid.y;
    uint pixelIndex = y * pc.width + x;
    
    if (x == 0 || x == pc.width - 1 || y == 0 || y == pc.height - 1) {
        outputBuf.data[pixelIndex] = grayBuf.data[pixelIndex];
        return;
    }
    
    uint idx_11 = (y - 1) * pc.width + (x - 1);
    uint idx_12 = (y - 1) * pc.width + x;
    uint idx_13 = (y - 1) * pc.width + (x + 1);
    uint idx_21 = y * pc.width + (x - 1);
    uint idx_23 = y * pc.width + (x + 1);
    uint idx_31 = (y + 1) * pc.width + (x - 1);
    uint idx_32 = (y + 1) * pc.width + x;
    uint idx_33 = (y + 1) * pc.width + (x + 1);
    
    int g11 = int(grayBuf.data[idx_11]);
    int g12 = int(grayBuf.data[idx_12]);
    int g13 = int(grayBuf.data[idx_13]);
    int g21 = int(grayBuf.data[idx_21]);
    int g23 = int(grayBuf.data[idx_23]);
    int g31 = int(grayBuf.data[idx_31]);
    int g32 = int(grayBuf.data[idx_32]);
    int g33 = int(grayBuf.data[idx_33]);
    
    int Gx = -g11 + g13 - 2*g21 + 2*g23 - g31 + g33;
    int Gy = -g11 - 2*g12 - g13 + g31 + 2*g32 + g33;
    
    int magnitude = abs(Gx) + abs(Gy);
    
    outputBuf.data[pixelIndex] = uint8_t(min(magnitude, 255));
}
)";

class VulkanSobelProcessor {
private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex;
    
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    
    VkPipelineLayout pipelineLayout;
    VkPipeline grayscalePipeline;
    VkPipeline sobelPipeline;
    
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    
    VkBuffer inputBuffer;
    VkDeviceMemory inputMemory;
    VkBuffer grayBuffer;
    VkDeviceMemory grayMemory;
    VkBuffer outputBuffer;
    VkDeviceMemory outputMemory;
    
    VkFence fence;
    
    uint32_t width, height;
    size_t imageSize;
    
    void createInstance() {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Sobel Edge Detector";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw runtime_error("Failed to create Vulkan instance");
        }
    }
    
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        
        if (deviceCount == 0) {
            throw runtime_error("No Vulkan-capable GPU found");
        }
        
        vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        physicalDevice = devices[0];
    }
    
    void createLogicalDevice() {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        
        vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
        
        queueFamilyIndex = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queueFamilyIndex = i;
                break;
            }
        }
        
        if (queueFamilyIndex == UINT32_MAX) {
            throw runtime_error("No compute queue found");
        }
        
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        
        VkDeviceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw runtime_error("Failed to create logical device");
        }
        
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
    }
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        
        throw runtime_error("Failed to find suitable memory type");
    }
    
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                     VkMemoryPropertyFlags properties, VkBuffer& buffer, 
                     VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo = {};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw runtime_error("Failed to create buffer");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        
        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw runtime_error("Failed to allocate buffer memory");
        }
        
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }
    
    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding bindings[3] = {};
        
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw runtime_error("Failed to create descriptor set layout");
        }
    }
    
    vector<uint32_t> compileGLSLToSPIRV(const char* shaderCode) {
        // Write shader to temp file
        ofstream tempFile("temp_shader.comp");
        tempFile << shaderCode;
        tempFile.close();
        
        // Compile using glslangValidator
        system("glslangValidator -V temp_shader.comp -o temp_shader.spv 2>/dev/null");
        
        // Read SPIR-V
        ifstream file("temp_shader.spv", ios::ate | ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Failed to compile shader");
        }
        
        size_t fileSize = (size_t)file.tellg();
        vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
        
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
        file.close();
        
        // Cleanup
        system("rm -f temp_shader.comp temp_shader.spv");
        
        return buffer;
    }
    
    VkShaderModule createShaderModule(const vector<uint32_t>& code) {
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode = code.data();
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw runtime_error("Failed to create shader module");
        }
        
        return shaderModule;
    }
    
    void createComputePipelines() {
        vector<uint32_t> grayscaleCode = compileGLSLToSPIRV(grayscaleShaderCode);
        vector<uint32_t> sobelCode = compileGLSLToSPIRV(sobelShaderCode);
        
        VkShaderModule grayscaleModule = createShaderModule(grayscaleCode);
        VkShaderModule sobelModule = createShaderModule(sobelCode);
        
        VkPushConstantRange pushConstant = {};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(uint32_t) * 2;
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw runtime_error("Failed to create pipeline layout");
        }
        
        VkComputePipelineCreateInfo grayscalePipelineInfo = {};
        grayscalePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        grayscalePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        grayscalePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        grayscalePipelineInfo.stage.module = grayscaleModule;
        grayscalePipelineInfo.stage.pName = "main";
        grayscalePipelineInfo.layout = pipelineLayout;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &grayscalePipelineInfo, 
                                    nullptr, &grayscalePipeline) != VK_SUCCESS) {
            throw runtime_error("Failed to create grayscale pipeline");
        }
        
        VkComputePipelineCreateInfo sobelPipelineInfo = {};
        sobelPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        sobelPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        sobelPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        sobelPipelineInfo.stage.module = sobelModule;
        sobelPipelineInfo.stage.pName = "main";
        sobelPipelineInfo.layout = pipelineLayout;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &sobelPipelineInfo, 
                                    nullptr, &sobelPipeline) != VK_SUCCESS) {
            throw runtime_error("Failed to create sobel pipeline");
        }
        
        vkDestroyShaderModule(device, grayscaleModule, nullptr);
        vkDestroyShaderModule(device, sobelModule, nullptr);
    }
    
    void createDescriptorPool() {
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 3;
        
        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = 1;
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw runtime_error("Failed to create descriptor pool");
        }
    }
    
    void createDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;
        
        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            throw runtime_error("Failed to allocate descriptor set");
        }
        
        VkDescriptorBufferInfo bufferInfos[3] = {};
        
        bufferInfos[0].buffer = inputBuffer;
        bufferInfos[0].offset = 0;
        bufferInfos[0].range = VK_WHOLE_SIZE;
        
        bufferInfos[1].buffer = grayBuffer;
        bufferInfos[1].offset = 0;
        bufferInfos[1].range = VK_WHOLE_SIZE;
        
        bufferInfos[2].buffer = outputBuffer;
        bufferInfos[2].offset = 0;
        bufferInfos[2].range = VK_WHOLE_SIZE;
        
        VkWriteDescriptorSet descriptorWrites[3] = {};
        
        for (int i = 0; i < 3; i++) {
            descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[i].dstSet = descriptorSet;
            descriptorWrites[i].dstBinding = i;
            descriptorWrites[i].dstArrayElement = 0;
            descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[i].descriptorCount = 1;
            descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        }
        
        vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, nullptr);
    }
    
    void createCommandBuffer() {
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw runtime_error("Failed to create command pool");
        }
        
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        if (vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer) != VK_SUCCESS) {
            throw runtime_error("Failed to allocate command buffer");
        }
    }
    
    void createSyncObjects() {
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw runtime_error("Failed to create fence");
        }
    }
    
public:
    VulkanSobelProcessor(uint32_t w, uint32_t h) : width(w), height(h) {
        imageSize = width * height;
        
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
        
        // Use byte-aligned buffers
        createBuffer(imageSize * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    inputBuffer, inputMemory);
        
        createBuffer(imageSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    grayBuffer, grayMemory);
        
        createBuffer(imageSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    outputBuffer, outputMemory);
        
        createDescriptorSetLayout();
        createDescriptorPool();
        createDescriptorSet();
        createComputePipelines();
        createCommandBuffer();
        createSyncObjects();
    }
    
    void process(const Mat& src, Mat& dest) {
        void* data;
        vkMapMemory(device, inputMemory, 0, imageSize * 3, 0, &data);
        memcpy(data, src.data, imageSize * 3);
        vkUnmapMemory(device, inputMemory);
        
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &fence);
        
        vkResetCommandBuffer(commandBuffer, 0);
        
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        uint32_t pushConstants[2] = {width, height};
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, grayscalePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
                               pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 
                          0, sizeof(pushConstants), pushConstants);
        
        uint32_t groupCountX = (width + 15) / 16;
        uint32_t groupCountY = (height + 15) / 16;
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
        
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        
        vkCmdPipelineBarrier(commandBuffer, 
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &barrier, 0, nullptr, 0, nullptr);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sobelPipeline);
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);
        
        vkEndCommandBuffer(commandBuffer);
        
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        
        vkMapMemory(device, outputMemory, 0, imageSize, 0, &data);
        memcpy(dest.data, data, imageSize);
        vkUnmapMemory(device, outputMemory);
    }
    
    ~VulkanSobelProcessor() {
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyPipeline(device, grayscalePipeline, nullptr);
        vkDestroyPipeline(device, sobelPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyBuffer(device, inputBuffer, nullptr);
        vkFreeMemory(device, inputMemory, nullptr);
        vkDestroyBuffer(device, grayBuffer, nullptr);
        vkFreeMemory(device, grayMemory, nullptr);
        vkDestroyBuffer(device, outputBuffer, nullptr);
        vkFreeMemory(device, outputMemory, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }
};




int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Error: Invalid input" << endl;
        return 0;
    }
    
    string videoPath = argv[1];
    VideoCapture cap(videoPath);
    
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video file." << endl;
        return 0;
    }
    
    Mat src, dest;
    cap.read(src);
    
    if (src.empty()) {
        cerr << "Error: Empty frame" << endl;
        return 0;
    }


    int retval, EventSet = PAPI_NULL;
    long long start_cycles, end_cycles, start_usec, end_usec;

    retval =  PAPI_library_init(PAPI_VER_CURRENT);   
    if (retval != PAPI_VER_CURRENT)
	handle_error(retval);
			
    /* Gets the starting time in clock cycles */
    start_cycles = PAPI_get_virt_cyc();
			
    /* Gets the starting time in microseconds */
    start_usec = PAPI_get_real_usec();
			
    /*Create an EventSet */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK)
	handle_error(retval);



    
    dest.create(src.rows, src.cols, CV_8UC1);
    
    VulkanSobelProcessor processor(src.cols, src.rows);
    
    namedWindow("sImage", WINDOW_NORMAL);
    
    do {
        processor.process(src, dest);
        imshow("sImage", dest);
        
        if (waitKey(1) == 27) break;
        
    } while (cap.read(src));


    /* Gets the ending time in clock cycles */
    end_cycles = PAPI_get_virt_cyc();
			
    /* Gets the ending time in microseconds */
    end_usec = PAPI_get_real_usec();
			
    printf("Virtual clock cycles: %lld\n", end_cycles - start_cycles);
    printf("Real clock time in microseconds: %lld\n", end_usec - start_usec);

    /* Executes if all low-level PAPI
    function calls returned PAPI_OK */
    printf("\033[0;32m\n\nPASSED\n\033[0m");

    
    return 0;
}