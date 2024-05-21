main: main.o
	g++ main.o  -o main

hdf5: loadh5.o
	gcc loadh5.o -o h5 -L/home/antshiv/Programs/lib -lhdf5 

main.o: main.c
	g++ -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_features2d -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lGL -lglut -lGLU  -c main.c


loadh5.o: loadh5.c
	gcc loadh5.c -c -L/home/antshiv/Programs/lib -lhdf5 -lhdf5_tools 
clean:
	rm -f main.o main
	rm *.o 
	rm h5
