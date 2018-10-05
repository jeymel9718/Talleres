/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

The program was parallelized using OpenMP by adding just
four lines

(1) A line to include omp.h -- the include file that
contains OpenMP's function prototypes and constants.

(2) A pragma that tells OpenMP to create a team of threads

(3) A pragma to cause one of the threads to print the
number of threads being used by the program.

(4) A pragma to split up loop iterations among the team
of threads.  This pragma includes 2 clauses to (1) create a
private variable and (2) to cause the threads to compute their
sums locally and then combine their local sums into a
single global value.

History: Written by Tim Mattson, 11/99.

#---------------------------------------------------------------

Modified by JGG to use threads equal to the number of processors.
SoC 2015.

#-----------------------------------------------------------------

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <arm_neon.h>
#include <omp.h>
int num_steps = 10000000;
float step;
int main ()
{
	int i, nprocs;
	float  pi, sum = 0.0;
	double start_time, run_time;
	float y[4];
	step = 1.0/num_steps;
	/* Use double of system processors (threads) */
	nprocs=2*omp_get_num_procs();
        /*Computes pi for each number of threads*/
	sum = 0.0;
	float x[num_steps],*xptr;
	float32x4_t Xvec;
	float32x4_t scaleVec=vdupq_n_f32(1.0);
	float32x4_t scaleVec2=vdupq_n_f32(4.0);
	float32x4_t sumV=vdupq_n_f32(0.0);

	omp_set_num_threads(nprocs);
	start_time = omp_get_wtime();
	#pragma omp parallel
	{
	#pragma omp for private(i)
	for(i=0;i<num_steps;++i){
		x[i]=(i+1-0.5)*step;
	}
	xptr=x;
	#pragma omp for private(i)
	for (i=0;i<num_steps; i+=4){
		Xvec=vld1q_f32(xptr);
		Xvec=vmlaq_f32(scaleVec,Xvec,Xvec);
		Xvec=vrecpeq_f32(Xvec);
		Xvec=vmulq_n_f32(Xvec,4.0);
		sumV=vaddq_f32(Xvec,sumV);
		xptr+=4;
	}
	}
vst1q_f32(y,sumV);
sum+=y[0];
sum+=y[1];
sum+=y[2];
sum+=y[3];
pi = step * sum;
run_time = omp_get_wtime() - start_time;
printf("\n pi is %f in %f seconds and %d threads\n",pi,run_time,nprocs);

}
