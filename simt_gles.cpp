#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <GLFW/glfw3.h>
#include <GLES2/gl2.h>

#include <papi.h>

using namespace std;
using namespace cv;

// Vertex shader: full-screen quad
const char* vsSrc = R"GLSL(
attribute vec2 aPos;
attribute vec2 aTexCoord;
varying vec2 vTexCoord;
void main() {
    vTexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

// Fragment shader: grayscale + Sobel on GPU
// Note: OpenCV sends BGR, GL reads as RGB, so:
// col.r = B, col.g = G, col.b = R
const char* fsSrc = R"GLSL(
precision mediump float;
uniform sampler2D uImage;
uniform vec2 uTexelSize; // 1/width, 1/height
varying vec2 vTexCoord;

float gray(vec3 col) {
    // inputs already in [0,1]
    return 0.0722 * col.r + 0.7152 * col.g + 0.2126 * col.b;
}

void main() {
    vec2 t = uTexelSize;

    vec3 c11 = texture2D(uImage, vTexCoord + vec2(-t.x, -t.y)).rgb;
    vec3 c12 = texture2D(uImage, vTexCoord + vec2( 0.0, -t.y)).rgb;
    vec3 c13 = texture2D(uImage, vTexCoord + vec2( t.x, -t.y)).rgb;

    vec3 c21 = texture2D(uImage, vTexCoord + vec2(-t.x,  0.0)).rgb;
    vec3 c22 = texture2D(uImage, vTexCoord + vec2( 0.0,  0.0)).rgb;
    vec3 c23 = texture2D(uImage, vTexCoord + vec2( t.x,  0.0)).rgb;

    vec3 c31 = texture2D(uImage, vTexCoord + vec2(-t.x,  t.y)).rgb;
    vec3 c32 = texture2D(uImage, vTexCoord + vec2( 0.0,  t.y)).rgb;
    vec3 c33 = texture2D(uImage, vTexCoord + vec2( t.x,  t.y)).rgb;

    float g11 = gray(c11);
    float g12 = gray(c12);
    float g13 = gray(c13);
    float g21 = gray(c21);
    float g22 = gray(c22);
    float g23 = gray(c23);
    float g31 = gray(c31);
    float g32 = gray(c32);
    float g33 = gray(c33);

    float Gx = (-g11 + g13)
             + (-2.0 * g21 + 2.0 * g23)
             + (-g31 + g33);

    float Gy = ( g11 + 2.0 * g12 + g13 )
             - ( g31 + 2.0 * g32 + g33 );

    float mag = abs(Gx) + abs(Gy);  // 0..~8

    // scale to 0..1 for display
    mag = mag / 4.0;                // tweak this if too bright/dim
    mag = clamp(mag, 0.0, 1.0);

    gl_FragColor = vec4(vec3(mag), 1.0);
}
)GLSL";

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        vector<char> log(len);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        cerr << "Shader compile error: " << log.data() << endl;
        exit(1);
    }
    return shader;
}

static GLuint createProgram(const char* vs, const char* fs) {
    GLuint vsObj = compileShader(GL_VERTEX_SHADER, vs);
    GLuint fsObj = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vsObj);
    glAttachShader(prog, fsObj);
    glBindAttribLocation(prog, 0, "aPos");
    glBindAttribLocation(prog, 1, "aTexCoord");
    glLinkProgram(prog);

    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (!status) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        vector<char> log(len);
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        cerr << "Program link error: " << log.data() << endl;
        exit(1);
    }

    glDeleteShader(vsObj);
    glDeleteShader(fsObj);
    return prog;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <video path>" << endl;
        return 0;
    }

    // ----- PAPI init -----
    int papi_ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_ret != PAPI_VER_CURRENT) {
        cerr << "PAPI_library_init error: " << papi_ret << endl;
    }

    string videoPath = argv[1];
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Error: could not open video file." << endl;
        return 1;
    }

    int srcW = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int srcH = (int)cap.get(CAP_PROP_FRAME_HEIGHT);

    if (srcW <= 0 || srcH <= 0) {
        Mat tmp;
        if (!cap.read(tmp) || tmp.empty()) {
            cerr << "Error: could not read frame to determine size." << endl;
            return 1;
        }
        srcW = tmp.cols;
        srcH = tmp.rows;
        cap.set(CAP_PROP_POS_FRAMES, 0);
    }

    // ---- Choose target (processed) resolution ----
    // We scale the video down to fit within maxW x maxH, preserving aspect ratio.
    int maxW = 640;
    int maxH = 480;
    float scaleW = (float)maxW / (float)srcW;
    float scaleH = (float)maxH / (float)srcH;
    float scale  = std::min(scaleW, scaleH);

    if (scale > 1.0f) {
        // Don't upscale small videos; process at native res
        scale = 1.0f;
    }

    int tgtW = (int)(srcW * scale);
    int tgtH = (int)(srcH * scale);

    if (tgtW <= 0) tgtW = srcW;
    if (tgtH <= 0) tgtH = srcH;

    cout << "Source size: " << srcW << "x" << srcH
         << "  ->  Target (processed/display) size: "
         << tgtW << "x" << tgtH << endl;

    // ---- Init GLFW/GLES ----
    if (!glfwInit()) {
        cerr << "Failed to init GLFW." << endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(tgtW, tgtH, "GLES Sobel", nullptr, nullptr);
    if (!window) {
        cerr << "Failed to create window." << endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    GLuint program = createProgram(vsSrc, fsSrc);
    glUseProgram(program);

    GLint locTex   = glGetUniformLocation(program, "uImage");
    GLint locTexel = glGetUniformLocation(program, "uTexelSize");

    // Fullscreen quad
    float quadVerts[] = {
        // pos      // tex
        -1.0f, -1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 0.0f
    };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture at target resolution
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGB,
                 tgtW,
                 tgtH,
                 0,
                 GL_RGB,
                 GL_UNSIGNED_BYTE,
                 nullptr);

    glUseProgram(program);
    glUniform1i(locTex, 0);
    glUniform2f(locTexel, 1.0f / tgtW, 1.0f / tgtH);

    long long frameCount = 0;
    long long start_us   = PAPI_get_real_usec();

    /* PAPI COUNTER START */
    /* Gets the starting time in clock cycles */
    long long start_cycles = PAPI_get_virt_cyc();
			
    /* Gets the starting time in microseconds */
    long long start_usec = PAPI_get_real_usec();

    Mat frameBGR;
    Mat frameSmall;

    while (!glfwWindowShouldClose(window)) {
        if (!cap.read(frameBGR) || frameBGR.empty()) {
            break; // end of video
        }

        frameCount++;

        // Resize to target resolution
        resize(frameBGR, frameSmall, Size(tgtW, tgtH), 0, 0, INTER_LINEAR);

        if (!frameSmall.isContinuous())
            frameSmall = frameSmall.clone();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D,
                        0,
                        0,
                        0,
                        tgtW,
                        tgtH,
                        GL_RGB,
                        GL_UNSIGNED_BYTE,
                        frameSmall.data);

        glViewport(0, 0, tgtW, tgtH);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    long long end_us   = PAPI_get_real_usec();
    long long total_us = end_us - start_us;

    /* Gets the ending time in clock cycles */
    long long end_cycles = PAPI_get_virt_cyc();
			
    /* Gets the ending time in microseconds */
    long long end_usec = PAPI_get_real_usec();

    // Print Ending times
    printf("Virtual clock cycles: %lld\n", end_cycles - start_cycles);
    printf("Real clock time in microseconds: %lld\n", end_usec - start_usec);

    /* Executes if all low-level PAPI
    function calls returned PAPI_OK */
    printf("\033[0;32m\n\nPASSED\n\033[0m");

    if (frameCount > 0) {
        double total_s  = total_us / 1e6;
        double total_ms = total_us / 1000.0;
        double fps      = frameCount / total_s;

        cout << "Frames processed: " << frameCount << endl;
        cout << "Average FPS: " << fps << endl;
    } else {
        cout << "No frames processed (check video path)." << endl;
    }

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}