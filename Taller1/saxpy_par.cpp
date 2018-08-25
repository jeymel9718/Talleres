#include <iostream>
#include <cstdlib>
#include <omp.h>

int nprocs=2*omp_get_num_procs();
void saxpy(int n, float a, int *  x, int *  y)
{
  omp_set_num_threads(nprocs);
  #pragma omp parallel
  {
    for (int i = 0; i < n; ++i){
        y[i] = a*x[i] + y[i];
      }
  }
}

int main(int argc, char const *argv[]) {
  double start_time, run_time;
  int x1[100], y1[100];
  int x2[150], y2[150];
  int x3[200], y3[200];
  /*generacion de los tres vectores a utilizar*/
  for(int i=0;i<100;++i){
    x1[i]=rand()%100;
    y1[i]=rand()%100;
  }

  for(int i=0;i<150;++i){
    x2[i]=rand()%100;
    y2[i]=rand()%100;
  }

  for(int i=0;i<200;++i){
    x3[i]=rand()%100;
    y3[i]=rand()%100;
  }

  /*start timer */
	start_time = omp_get_wtime();
  saxpy(100,2.0,x1,y1);
  run_time = omp_get_wtime() - start_time;
  std::cout<<"vector de 100 elementos en un tiempo de: "<<run_time<<'\n';

  /*start timer */
	start_time = omp_get_wtime();
  saxpy(150,2.0,x2,y2);
  run_time = omp_get_wtime() - start_time;
  std::cout<<"vector de 150 elementos en un tiempo de: "<<run_time<<'\n';

  /*start timer */
	start_time = omp_get_wtime();
  saxpy(200,2.0,x3,y3);
  run_time = omp_get_wtime() - start_time;
  std::cout<<"vector de 200 elementos en un tiempo de: "<<run_time<<'\n';

  return 0;
}
