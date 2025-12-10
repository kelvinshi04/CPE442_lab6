#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include <GLFW/glfw3.h>
#include <GLES2/gl2.h>

#include <papi.h>

using namespace std;
using namespace cv;

// ===========================
// GLSL SHADERS
// ===========================

// Vertex shader: full-screen quad with scaling
const char* vsSrc = R"GLSL(
attribute vec2 aPos;
attribute vec2 aTexCoord;
varying vec2 vTexCoord;

uniform vec2 uScale; // scales quad to fit window while preserving aspect

void main() {
    vTexCoord = aTexCoord;
    vec2 scaledPos = aPos * uScale;
    gl_Position = vec4(scaledPos, 0.0, 1.0);
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

    // 3x3 neighborhood
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

    // Sobel X and Y
    float Gx = (-g11 + g13)
             + (-2.0 * g21 + 2.0 * g23)
             + (-g31 + g33);

    float Gy = ( g11 + 2.0 * g12 + g13 )
             - ( g31 + 2.0 * g32 + g33 );

    // Edge magnitude (L1)
    float mag = abs(Gx) + abs(Gy);  // 0..roughly 8
    mag = mag / 4.0;
    mag = clamp(mag, 0.0, 1.0);

    gl_FragColor = vec4(vec3(mag), 1.0);
}
)GLSL";

// ===========================
// GL UTILITIES
// ===========================

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
    GLuint prog  = glCreateProgram();

    glAttachShader(prog, vsObj);
    glAttachShader(prog, fsObj);
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

// ===========================
// MAIN
// ===========================

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

    // ---- Determine source size (original resolution) ----
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

    // ---- Use original video resolution as texture size (no resize) ----
    int tgtW = srcW;
    int tgtH = srcH;

    cout << "Processing at original resolution: "
         << tgtW << "x" << tgtH << endl;

    // ---- Init GLFW/GLES ----
    if (!glfwInit()) {
        cerr << "Failed to init GLFW" << endl;
        return 1;
    }

    // Request OpenGL ES 2.0 context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // Window is resizable by default (like OpenCV WINDOW_NORMAL)

    GLFWwindow* window = glfwCreateWindow(
        tgtW, tgtH, "GLES Sobel (Scalable Window)", nullptr, nullptr
    );
    if (!window) {
        cerr << "Failed to create window." << endl;
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    // Disable vsync for max FPS (you can set 1 if you prefer no tearing)
    glfwSwapInterval(0);

    // Create shader program
    GLuint program = createProgram(vsSrc, fsSrc);
    glUseProgram(program);

    // Look up attribute and uniform locations
    GLint locPos     = glGetAttribLocation(program, "aPos");
    GLint locTexCoord= glGetAttribLocation(program, "aTexCoord");
    GLint locTex     = glGetUniformLocation(program, "uImage");
    GLint locTexel   = glGetUniformLocation(program, "uTexelSize");
    GLint locScale   = glGetUniformLocation(program, "uScale");

    // ---- Full-screen quad geometry (two triangles as a strip) ----
    float quadVerts[] = {
        //  x,    y,   u,   v
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 0.0f
    };

    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

    glEnableVertexAttribArray(locPos);
    glVertexAttribPointer(
        locPos,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        (void*)0
    );

    glEnableVertexAttribArray(locTexCoord);
    glVertexAttribPointer(
        locTexCoord,
        2,
        GL_FLOAT,
        GL_FALSE,
        4 * sizeof(float),
        (void*)(2 * sizeof(float))
    );

    // ---- Texture at original resolution ----
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
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
    glUniform1i(locTex, 0);                      // texture unit 0
    glUniform2f(locTexel, 1.0f / tgtW, 1.0f / tgtH);

    // Clear color for letterboxing area
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    long long frameCount = 0;

    long long start_us     = PAPI_get_real_usec();
    long long start_cycles = PAPI_get_virt_cyc();

    Mat frameBGR;

    // ===========================
    // MAIN LOOP
    // ===========================
    while (!glfwWindowShouldClose(window)) {
        // Read next frame
        if (!cap.read(frameBGR) || frameBGR.empty()) {
            break; // end of video
        }
        frameCount++;

        if (!frameBGR.isContinuous())
            frameBGR = frameBGR.clone();

        // Upload original-resolution frame to texture
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
                        frameBGR.data);

        // Get current window framebuffer size
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);

        // Set viewport to full window
        glViewport(0, 0, fbW, fbH);

        // Compute scale to fit image in window, preserving aspect ratio
        float imageAspect   = static_cast<float>(tgtW) / static_cast<float>(tgtH);
        float windowAspect  = static_cast<float>(fbW) / static_cast<float>(fbH);
        float scaleX = 1.0f;
        float scaleY = 1.0f;

        if (windowAspect > imageAspect) {
            // Window is wider than image: fit height, letterbox left/right
            scaleX = imageAspect / windowAspect;
            scaleY = 1.0f;
        } else {
            // Window is taller/narrower: fit width, letterbox top/bottom
            scaleX = 1.0f;
            scaleY = windowAspect / imageAspect;
        }

        // Clear background (for letterboxing areas)
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw Sobel-processed quad with scaling
        glUseProgram(program);
        glUniform2f(locScale, scaleX, scaleY);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    long long end_us       = PAPI_get_real_usec();
    long long total_us     = end_us - start_us;
    long long end_cycles   = PAPI_get_virt_cyc();
    long long total_cycles = end_cycles - start_cycles;

    if (frameCount > 0) {
        double total_s = total_us / 1e6;
        double fps     = frameCount / total_s;

        cout << "Virtual clock cycles: " << total_cycles << endl;
        cout << "Real clock time in microseconds: " << total_us << endl;
        cout << "\033[0;32m\nPASSED\n\033[0m";

        cout << "\nFrames processed: " << frameCount << endl;
        cout << "Average FPS: " << fps << endl;
    } else {
        cout << "No frames processed (check video path)." << endl;
    }

    // Cleanup
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(program);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
