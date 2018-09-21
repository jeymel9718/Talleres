/*
*  Make a vector using the max value of each matrix column
*  Matrix size is 2x8
*/
#include <emmintrin.h> //v3
#include <smmintrin.h> //v4
#include <iostream>

int main(int argc, char const *argv[]) {
  short matrix[16];
  short data=0;
  __m128i vector1;
  __m128i vector2;
  __m128i res;
  std::cout << "Ingrese los datos para la matriz 2x8: " << '\n';
  for(int i=0;i<16;++i){
    std::cin >> matrix[i]; //get data from the user
  }

  vector1=_mm_set_epi16(matrix[7],matrix[6],matrix[5],matrix[4],matrix[3],matrix[2],matrix[1],matrix[0]); //store row1
  vector2=_mm_set_epi16(matrix[15],matrix[14],matrix[13],matrix[12],matrix[11],matrix[10],matrix[9],matrix[8]); //store row2
  res=_mm_max_epi16(vector1,vector2); //get max value per column

  //vetor Printing
  std::cout << "Vector resultante" << '\n';

  data=_mm_extract_epi16(res,0);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,1);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,2);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,3);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,4);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,5);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,6);
  std::cout << data << ' ';

  data=_mm_extract_epi16(res,7);
  std::cout << data << ' ';

  std::cout << '\n';

  return 0;
}
