#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void saxpy(int n, float a, int *  x, int *  y)
{
  for (int i = 0; i < n; ++i)
      y[i] = a*x[i] + y[i];
}

int main(int argc, char const *argv[]) {
  double start_time, run_time;
  int x1[1000], y1[1000];
  int x2[2000], y2[2000];
  int x3[3000], y3[3000];
  /*generacion de los tres vectores a utilizar*/
  for(int i=0;i<1000;++i){
    x1[i]=rand()%100;
    y1[i]=rand()%100;
  }

  for(int i=0;i<2000;++i){
    x2[i]=rand()%100;
    y2[i]=rand()%100;
  }

  for(int i=0;i<3000;++i){
    x3[i]=rand()%100;
    y3[i]=rand()%100;
  }

  /*start timer */
	start_time = omp_get_wtime();
  saxpy(1000,2.0,x1,y1);
  run_time = omp_get_wtime() - start_time;
  printf("Saxpy de 1000 elementos serial %lf\n",run_time);

  /*start timer */
	start_time = omp_get_wtime();
  saxpy(2000,2.0,x2,y2);
  run_time = omp_get_wtime() - start_time;
  printf("Saxpy de 2000 elementos serial %lf\n",run_time);

  /*start timer */
	start_time = omp_get_wtime();
  saxpy(3000,2.0,x3,y3);
  run_time = omp_get_wtime() - start_time;
  printf("Saxpy de 3000 elementos serial %lf\n",run_time);

  return 0;
}
