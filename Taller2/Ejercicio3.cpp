/*
*  Matrix multiplicatio using SIMD
*  This code multiple 2 4x4 Matrix
*/

#include <emmintrin.h> //v3
#include <iostream>

int main(int argc, char const *argv[]) {
  float a[4][4]={{1.123,5.621,123.5,0.6412},
  {2.589,3.256,8.568,1.236},
  {8.265,6.994,0.365,3.565},
  {9.365,0.365,0.0002,2.565}};
  float b[4][4]={{6.123,0.621,1.5,4.12},
  {2.589,5.6,8.68,16},
  {0.65,6.994,1.365,3.65},
  {6.5,3.65,0.02,6.5}};

  float c[4][4];
  float tmp[4]; //used to store multiplication result
  __m128  vector1;
  __m128  vector2;
  __m128  result;
  float sum=0;
  for(int i=0;i<4;++i){
    vector1=_mm_load_ps(a[i]); //load row
    for(int j=0;j<4;++j){
      vector2=_mm_set_ps(b[j][3],b[j][2],b[j][1],b[j][0]); //load column
      result=_mm_mul_ps(vector1,vector2);
      _mm_store_ps(tmp,result); //store result
      sum=tmp[0]+tmp[1]+tmp[2]+tmp[3];
      c[i][j]=sum;
    }
  }

  /* Print result matrix */
  for(int i=0;i<4;++i){
    for(int j=0;j<4;++j){
      std::cout << c[i][j] << ' ';
    }
    std::cout << '\n';
  }

  return 0;
}
