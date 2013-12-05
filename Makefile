CC=g++
CFLAGS=-Wall -pedantic
OLIBS=video.o

all: video.o
	$(CC) $(OLIBS) $(O_LIBS) -o video

video.o:
	$(CC) -c $(CFLAGS) video.cpp

clean:
	rm -rf *.o
	rm -rf video
