CC=g++
CFLAGS=-Wall -pedantic
OLIBS=video.o derivative.o motion.o

all: $(OLIBS)
	$(CC) $(OLIBS) $(O_LIBS) -o video

video.o:
	$(CC) -c $(CFLAGS) video.cpp

derivative.o:
	$(CC) -c $(CFLAGS) derivative.cpp

motion.o:
	$(CC) -c $(CFLAGS) motion.cpp

clean:
	rm -rf *.o
	rm -rf video