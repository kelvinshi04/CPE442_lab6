CC = g++
CFLAGS = -Wall -Wextra -std=c++17 -I /usr/local/include/opencv4 -I /home/nick/Documents/papi/src -Ofast -pthread -mcpu=cortex-a76
LDFLAGS = -lopencv_imgcodecs -lopencv_core -lopencv_highgui -lopencv_videoio -L/home/nick/Documents/papi/src -lpapi
TARGET = development
SRC = development.cpp

$(TARGET) : $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)