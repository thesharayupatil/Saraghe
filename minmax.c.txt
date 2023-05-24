Sum
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
  int i, n;
  float a[100], b[100], sum;
  /* Some initializations */
  n = 100;
  for (i=0; i < n; i++)
    a[i] = b[i] = i * 1.0;
  sum = 0.0;
  #pragma omp parallel for reduction(+:sum)
  for (i=0; i < n; i++)
    sum = sum + (a[i] * b[i]);
  printf(" Sum = %f\n",sum);
}
min
======
#include <stdio.h>
#include <omp.h>

int main()
{
  double arr[10];
  omp_set_num_threads(4);
  double min_val=9.0;
  int i;
  for( i=0; i<10; i++)
     arr[i] = 2.0 + i;
  #pragma omp parallel for reduction(min : min_val)
  for( i=0;i<10; i++)
  {
     printf("thread id = %d and i = %d \n", omp_get_thread_num(),i); 
     if(arr[i] < min_val)
     {
        min_val = arr[i];
     }
  }
  printf("\nmin_val = %f", min_val);
}
avg
=====
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
   int i, n;
   float a[100], b[100], sum;
   /* Some initializations */
   n = 100;
   for (i = 0; i < n; i++)
      a[i] = b[i] = i * 1.0;
   sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
   for (i = 0; i < n; i++)
      sum = sum + (a[i] * b[i]);
   printf(" Avg = %f\n", sum/n);
}
max 
====
#include <stdio.h>
#include <omp.h>

int main()
{
  double arr[10];
  omp_set_num_threads(4);
  double max_val=0.0;
  int i;
  for( i=0; i<10; i++)
     arr[i] = 2.0 + i;
  #pragma omp parallel for reduction(max : max_val)
  for( i=0;i<10; i++)
  {
      printf("thread id = %d and i = %d \n", omp_get_thread_num(),i);
      if(arr[i] > max_val)
      {
         max_val = arr[i];
      }
  }
  printf("\nmax_val = %f", max_val);
}


