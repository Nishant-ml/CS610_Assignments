#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NSEC_SEC_MUL (1.0e9)

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk);

struct timespec begin_grid, end_main;

// to store values of disp.txt
double a[120];

// to store values of grid.txt
double b[30];

int main() {
  int i, j;

  i = 0;
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fp)) {
    if (!fscanf(fp, "%lf", &a[i])) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
    i++;
  }
  fclose(fp);

  // read grid file
  j = 0;
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  while (!feof(fpq)) {
    if (!fscanf(fpq, "%lf", &b[j])) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
    j++;
  }
  fclose(fpq);

  // grid value initialize
  // initialize value of kk;
  double kk = 0.3;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);
  gridloopsearch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12],
                 b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23], b[24],
                 b[25], b[26], b[27], b[28], b[29], a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
                 a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19],
                 a[20], a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31],
                 a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40], a[41], a[42], a[43],
                 a[44], a[45], a[46], a[47], a[48], a[49], a[50], a[51], a[52], a[53], a[54], a[55],
                 a[56], a[57], a[58], a[59], a[60], a[61], a[62], a[63], a[64], a[65], a[66], a[67],
                 a[68], a[69], a[70], a[71], a[72], a[73], a[74], a[75], a[76], a[77], a[78], a[79],
                 a[80], a[81], a[82], a[83], a[84], a[85], a[86], a[87], a[88], a[89], a[90], a[91],
                 a[92], a[93], a[94], a[95], a[96], a[97], a[98], a[99], a[100], a[101], a[102],
                 a[103], a[104], a[105], a[106], a[107], a[108], a[109], a[110], a[111], a[112],
                 a[113], a[114], a[115], a[116], a[117], a[118], a[119], kk);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);
  printf("Total time = %f seconds\n", (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
                                          (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// grid search function with loop variables

void gridloopsearch(
    double dd1, double dd2, double dd3, double dd4, double dd5, double dd6, double dd7, double dd8,
    double dd9, double dd10, double dd11, double dd12, double dd13, double dd14, double dd15,
    double dd16, double dd17, double dd18, double dd19, double dd20, double dd21, double dd22,
    double dd23, double dd24, double dd25, double dd26, double dd27, double dd28, double dd29,
    double dd30, double c11, double c12, double c13, double c14, double c15, double c16, double c17,
    double c18, double c19, double c110, double d1, double ey1, double c21, double c22, double c23,
    double c24, double c25, double c26, double c27, double c28, double c29, double c210, double d2,
    double ey2, double c31, double c32, double c33, double c34, double c35, double c36, double c37,
    double c38, double c39, double c310, double d3, double ey3, double c41, double c42, double c43,
    double c44, double c45, double c46, double c47, double c48, double c49, double c410, double d4,
    double ey4, double c51, double c52, double c53, double c54, double c55, double c56, double c57,
    double c58, double c59, double c510, double d5, double ey5, double c61, double c62, double c63,
    double c64, double c65, double c66, double c67, double c68, double c69, double c610, double d6,
    double ey6, double c71, double c72, double c73, double c74, double c75, double c76, double c77,
    double c78, double c79, double c710, double d7, double ey7, double c81, double c82, double c83,
    double c84, double c85, double c86, double c87, double c88, double c89, double c810, double d8,
    double ey8, double c91, double c92, double c93, double c94, double c95, double c96, double c97,
    double c98, double c99, double c910, double d9, double ey9, double c101, double c102,
    double c103, double c104, double c105, double c106, double c107, double c108, double c109,
    double c1010, double d10, double ey10, double kk) {
  // results values
  double x1, x2, x3, x4, x5, x6, x7, x8, x9, x10;

  // constraint values
  double q1, q2, q3, q4, q5, q6, q7, q8, q9, q10;

  // results points
  long pnts = 0;

  // re-calculated limits
  double e1, e2, e3, e4, e5, e6, e7, e8, e9, e10;

  // opening the "results-v0.txt" for writing he results in append mode
  FILE* fptr = fopen("./results-v1.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  // initialization of re calculated limits, xi's.
  e1 = kk * ey1;
  e2 = kk * ey2;
  e3 = kk * ey3;
  e4 = kk * ey4;
  e5 = kk * ey5;
  e6 = kk * ey6;
  e7 = kk * ey7;
  e8 = kk * ey8;
  e9 = kk * ey9;
  e10 = kk * ey10;

  x1 = dd1;
  x2 = dd4;
  x3 = dd7;
  x4 = dd10;
  x5 = dd13;
  x6 = dd16;
  x7 = dd19;
  x8 = dd22;
  x9 = dd25;
  x10 = dd28;

  // for loop upper values
  int s1, s2, s3, s4, s5, s6, s7, s8, s9, s10;
  s1 = floor((dd2 - dd1) / dd3);
  s2 = floor((dd5 - dd4) / dd6);
  s3 = floor((dd8 - dd7) / dd9);
  s4 = floor((dd11 - dd10) / dd12);
  s5 = floor((dd14 - dd13) / dd15);
  s6 = floor((dd17 - dd16) / dd18);
  s7 = floor((dd20 - dd19) / dd21);
  s8 = floor((dd23 - dd22) / dd24);
  s9 = floor((dd26 - dd25) / dd27);
  s10 = floor((dd29 - dd28) / dd30);
	
  q1 = 0.0;
  q2 = 0.0;
  q3 = 0.0;
  q4 = 0.0;
  q5 = 0.0;
  q6 = 0.0;
  q7 = 0.0;
  q8 = 0.0;
  q9 = 0.0;
  q10 = 0.0;


  // grid search starts
  for (int r1 = 0; r1 < s1; ++r1) {
    x1 = dd1 + r1 * dd3;
    double q1 = c11*x1, q2 = c21*x1, q3 = c31*x1, q4 = c41*x1, q5 = c51*x1, q6 = c61*x1, q7 = c71*x1, q8 = c81*x1, q9 = c91*x1, q10 = c101*x1;

    for (int r2 = 0; r2 < s2; ++r2) {
      x2 = dd4 + r2 * dd6;
      double r2q1 = q1 + c12*x2, r2q2 = q2 + c22*x2, r2q3 = q3 + c32*x2, r2q4 = q4 + c42*x2, r2q5 = q5 + c52*x2;
      double r2q6 = q6 + c62*x2, r2q7 = q7 + c72*x2, r2q8 = q8 + c82*x2, r2q9 = q9 + c92*x2, r2q10 = q10 + c102*x2;

      for (int r3 = 0; r3 < s3; ++r3) {
        x3 = dd7 + r3 * dd9;
	      double r3q1 = r2q1 + c13*x3, r3q2 = r2q2 + c23*x3, r3q3 = r2q3 + c33*x3, r3q4 = r2q4 + c43*x3, r3q5 = r2q5 + c53*x3; 
     	  double r3q6 = r2q6 + c63*x3, r3q7 = r2q7 + c73*x3, r3q8 = r2q8 + c83*x3, r3q9 = r2q9 + c93*x3, r3q10 = r2q10 + c103*x3;

        for (int r4 = 0; r4 < s4; ++r4) {
          x4 = dd10 + r4 * dd12;
          double r4q1 = r3q1 + c14*x4, r4q2 = r3q2 + c24*x4, r4q3 = r3q3 + c34*x4, r4q4 = r3q4 + c44*x4, r4q5 = r3q5 + c54*x4; 
     	    double r4q6 = r3q6 + c64*x4, r4q7 = r3q7 + c74*x4, r4q8 = r3q8 + c84*x4, r4q9 = r3q9 + c94*x4, r4q10 = r3q10 + c104*x4;

          for (int r5 = 0; r5 < s5; ++r5) {
            x5 = dd13 + r5 * dd15;
            double r5q1 = r4q1 + c15*x5, r5q2 = r4q2 + c25*x5, r5q3 = r4q3 + c35*x5, r5q4 = r4q4 + c45*x5, r5q5 = r4q5 + c55*x5; 
     	      double r5q6 = r4q6 + c65*x5, r5q7 = r4q7 + c75*x5, r5q8 = r4q8 + c85*x5, r5q9 = r4q9 + c95*x5, r5q10 = r4q10+ c105*x5;
            
            for (int r6 = 0; r6 < s6; ++r6) {
              x6 = dd16 + r6 * dd18;
              double r6q1 = r5q1 + c16*x6, r6q2 = r5q2 + c26*x6, r6q3 = r5q3 + c36*x6, r6q4 = r5q4 + c46*x6, r6q5 = r5q5 + c56*x6; 
     	        double r6q6 = r5q6 + c66*x6, r6q7 = r5q7 + c76*x6, r6q8 = r5q8 + c86*x6, r6q9 = r5q9 + c96*x6, r6q10 = r5q10 + c106*x6;

              for (int r7 = 0; r7 < s7; ++r7) {
                x7 = dd19 + r7 * dd21;
                double r7q1 = r6q1 + c17*x7, r7q2 = r6q2 + c27*x7, r7q3 = r6q3 + c37*x7, r7q4 = r6q4 + c47*x7, r7q5 = r6q5 + c57*x7; 
     	          double r7q6 = r6q6 + c67*x7, r7q7 = r6q7 + c77*x7, r7q8 = r6q8 + c87*x7, r7q9 = r6q9 + c97*x7, r7q10 = r6q10 + c107*x7;

                for (int r8 = 0; r8 < s8; ++r8) {
                  x8 = dd22 + r8 * dd24;
                  double r8q1 = r7q1 + c18*x8, r8q2 = r7q2 + c28*x8, r8q3 = r7q3 + c38*x8, r8q4 = r7q4 + c48*x8, r8q5 = r7q5 + c58*x8; 
     	            double r8q6 = r7q6 + c68*x8, r8q7 = r7q7 + c78*x8, r8q8 = r7q8 + c88*x8, r8q9 = r7q9 + c98*x8, r8q10 = r7q10 + c108*x8;

                  for (int r9 = 0; r9 < s9; ++r9) {
                    x9 = dd25 + r9 * dd27;
                    double r9q1 = r8q1 + c19*x9, r9q2 = r8q2 + c29*x9, r9q3 = r8q3 + c39*x9, r9q4 = r8q4 + c49*x9, r9q5 = r8q5 + c59*x9; 
     	              double r9q6 = r8q6 + c69*x9, r9q7 = r8q7 + c79*x9, r9q8 = r8q8 + c89*x9, r9q9 = r8q9 + c99*x9, r9q10 = r8q10 + c109*x9;

                    for (int r10 = 0; r10 < s10; ++r10) {
                      x10 = dd28 + r10 * dd30;
                      double q1 = r9q1 + c110*x10, q2 = r9q2 + c210*x10, q3 = r9q3 + c310*x10, q4 = r9q4 + c410*x10, q5 = r9q5 + c510*x10; 
     	                double q6 = r9q6 + c610*x10, q7 = r9q7 + c710*x10, q8 = r9q8 + c810*x10, q9 = r9q9 + c910*x10, q10 = r9q10 + c1010*x10;


                      // constraints

                      q1 = fabs(q1-d1);

                      q2 = fabs(q2-d2);

                      q3 = fabs(q3-d3);

                      q4 = fabs(q4-d4);

                      q5 = fabs(q5-d5);

                      q6 = fabs(q6-d6);

                      q7 = fabs(q7-d7);

                      q8 = fabs(q8-d8);

                      q9 = fabs(q9-d9);

                      q10 = fabs(q10-d10);

                      if ((q1 <= e1) && (q2 <= e2) && (q3 <= e3) && (q4 <= e4) && (q5 <= e5) &&
                          (q6 <= e6) && (q7 <= e7) && (q8 <= e8) && (q9 <= e9) && (q10 <= e10)) {
                        pnts = pnts + 1;

                        // xi's which satisfy the constraints to be written in file
                        fprintf(fptr, "%lf\t", x1);
                        fprintf(fptr, "%lf\t", x2);
                        fprintf(fptr, "%lf\t", x3);
                        fprintf(fptr, "%lf\t", x4);
                        fprintf(fptr, "%lf\t", x5);
                        fprintf(fptr, "%lf\t", x6);
                        fprintf(fptr, "%lf\t", x7);
                        fprintf(fptr, "%lf\t", x8);
                        fprintf(fptr, "%lf\t", x9);
                        fprintf(fptr, "%lf\n", x10);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  fclose(fptr);
  printf("result pnts: %ld\n", pnts);

  // end function gridloopsearch
}