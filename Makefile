CC=clang++
CFLAG=-std=c++14 -O2
LFLAG=-O -std=c++14
OBJ=main.o LSTMNetwork.o kbhit.o progress.o
INCLUDE=-I/usr/local/include/
LINK=-L/usr/local/lib
LIB=

main: $(OBJ)
	$(CC) $(LFLAG) $(LINK) $(LIB) $(OBJ)
	
main.o: main.cpp
	$(CC) $(CFLAG) -c main.cpp $(INCLUDE)

LSTMNetwork.o: LSTMNetwork.cpp
	$(CC) $(CFLAG) -c LSTMNetwork.cpp $(INCLUDE)

kbhit.o: kbhit.cpp
	$(CC) $(CFLAG) -c kbhit.cpp $(INCLUDE)

progress.o: progress.cpp
	$(CC) $(CFLAG) -c progress.cpp $(INCLUDE)
	
clean:
	rm -f *.o
	rm -f *.out
	rm -f main
