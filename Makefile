CC=gcc
CFLAGS=-Wall -pedantic
OLIBS=video.o derivative.o

all: $(OLIBS)
	$(CC) $(OLIBS) $(O_LIBS) -o video

video.o:
	$(CC) -c $(CFLAGS) video.cpp

derivative.o:
	$(CC) -c $(CFLAGS) derivative.cpp

clean:
	rm -rf *.o
	rm -rf video