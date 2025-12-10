CC = g++
CFLAGS = -Wall -Wextra -std=c++17 -O3 -I /usr/local/include/opencv4 -I /home/nick/Documents/papi/src

LDFLAGS = -lopencv_imgproc -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_videoio -lglfw -lGLESv2 -lEGL -L/home/nick/Documents/papi/src -lpapi

simt_gles: simt_gles.cpp
	$(CC) $(CFLAGS) simt_gles.cpp -o simt_gles $(LDFLAGS)