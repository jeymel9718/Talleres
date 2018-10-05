#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <arm_neon.h>
#include <omp.h>

int nprocs;
int i;
double start_time, run_time;

void saxpy(int n, float a, float *  x, float *  y)
{
  omp_set_num_threads(nprocs);
  float32x4_t scaleVec = vdupq_n_f32(a); //scalar vector
  float32x4_t Xvec,Yvec; //vector of x and y
  start_time = omp_get_wtime();
  #pragma omp parallel
  {
    for (; n!=0; n-=4){
      Xvec = vld1q_f32(x);
      Yvec = vld1q_f32(y);
      Xvec = vmulq_f32(Xvec,scaleVec);
      Yvec = vaddq_f32(Xvec,Yvec);
      vst1q_f32(y,Yvec);
      x+=4;
      y+=4;

    }
  }
  run_time = omp_get_wtime() - start_time;
}

int main(int argc, char const *argv[]) {
  float x1[1000], y1[1000];
  float x2[2000], y2[2000];
  float x3[3000], y3[3000];
  /*generacion de los tres vectores a utilizar*/
  for(i=0;i<1000;++i){
    x1[i]=rand()%100;
    y1[i]=rand()%100;
  }

  for(i=0;i<2000;++i){
    x2[i]=rand()%100;
    y2[i]=rand()%100;
  }

  for(i=0;i<3000;++i){
    x3[i]=rand()%100;
    y3[i]=rand()%100;
  }

  nprocs=2*omp_get_num_procs();

  saxpy(1000,2.0,x1,y1);
  printf("Tiempo saxpy con 1000 elementos paralelo: %lf\n",run_time );

  saxpy(2000,2.0,x2,y2);
  printf("Tiempo saxpy con 2000 elementos paralelo: %lf\n",run_time );

  saxpy(3000,2.0,x3,y3);
  printf("Tiempo saxpy con 3000 elementos paralelo: %lf\n",run_time );

  return 0;
}
