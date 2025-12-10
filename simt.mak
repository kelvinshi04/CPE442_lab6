CC = g++
CFLAGS = -Wall -Wextra -std=c++17 -I /usr/local/include/opencv4 -I /home/nick/Documents/papi/src -O3
LDFLAGS = -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_videoio -lOpenCL -L/home/nick/Documents/papi/src -lpapi

TARGET_FLOAT = simt_float
TARGET_INT   = simt_int

all: $(TARGET_FLOAT) $(TARGET_INT)

$(TARGET_FLOAT): simt_float.cpp
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(TARGET_INT): simt_int.cpp
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET_FLOAT) $(TARGET_INT)
