/*
    Programa que calcula el valor de e con la serie:
            sum_from 0 to infinite {1/k!}
*/

#include <iostream>
#include <omp.h>

static long nsteps = 10000000;
int nprocs=2*omp_get_num_procs();
int main(int argc, char const *argv[]) {

  double res = 2.0;
  double fact = 1;

  double start_time, run_time;
  omp_set_num_threads(nprocs);
  /*start time*/
  start_time = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp for reduction(+:res) reduction(*:fact)

    for (int i=2; i<nsteps; i++)
   {
     fact *= i;
     res += 1/fact;
   }
 }
 run_time = omp_get_wtime() - start_time;
 std::cout<<"e is "<<res<<" in "<<run_time<<" seconds"<<'\n';
  return 0;
}
