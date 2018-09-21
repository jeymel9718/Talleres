CC=g++

all: Simd ejercicio2 ejercicio3

Simd: hello_simd.c
	gcc hello_simd.c -o ej1

ejercicio2: Ejercicio2.cpp
	$(CC) Ejercicio2.cpp -o ej2 -std=c++11

ejercicio3: Ejercicio3.cpp
		$(CC) Ejercicio3.cpp -o ej3 -std=c++11

clean:
	rm ej1 ej2 ej3
