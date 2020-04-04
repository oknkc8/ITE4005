TARGET = apriori.exe dataGenerator.exe test.exe

all: $(TARGET)

apriori.exe : apriori.o
	g++ -std=c++11 -o apriori.exe apriori.o

dataGenerator.exe : dataGenerator.o
	g++ -std=c++11 -o dataGenerator.exe dataGenerator.o

test.exe : test.o
	g++ -std=c++11 -o test.exe test.o

apriori.o : apriori.cpp
	g++ -std=c++11 -c -o apriori.o apriori.cpp

dataGenerator.o : dataGenerator.cpp
	g++ -std=c++11 -c -o dataGenerator.o dataGenerator.cpp

test.o : test.cpp
	g++ -std=c++11 -c -o test.o test.cpp

clean:
	rm $(TARGET) apriori.o dataGenerator.o test.o
