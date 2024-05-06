main: main.o
	g++ main.o  -o main

main.o: main.c
	g++ -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lGL -lglut -lGLU  -c main.c

clean:
	rm -f main.o main
