CC=g++
CFLAGS=-Wall -O2 -g
OLIBS=derivative.o motion.o video.o

all: $(OLIBS)
	$(CC)  -o video  $(O_LIBS) $(OLIBS)

video.o:
	$(CC) -c $(CFLAGS) video.cpp

derivative.o:
	$(CC) -c $(CFLAGS) derivative.cpp

motion.o:
	$(CC) -c $(CFLAGS) motion.cpp

clean:
	rm -rf *.o
	rm -rf video
