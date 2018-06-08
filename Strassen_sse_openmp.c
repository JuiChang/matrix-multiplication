//c99 -O3 -fopenmp -Wall foo.c
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>
#include <omp.h>

void gemm(float * restrict a, float * restrict b, float * restrict c, int ma, int na, int nb) {
	int i, j, k;
    for(i = 0; i < ma; i++) {
        for(j = 0; j < na; j++) {
            for(k = 0; k < nb; k++) {
                c[i*nb + k] += a[i*na + j] * b[j*nb + k];
            }
        }
    }
}

void gemm_tlp(float * restrict a, float * restrict b, float * restrict c, int ma, int na, int nb) {
	int i, j, k;
	#pragma omp parallel for private(i, j, k) shared(a, b, c)
    for(i=0; i<ma; i++) {
        for(j=0; j<na; j++) {
            for(k=0; k<nb; k++) {
				c[i*nb + k] += a[i*na + j] * b[j*nb + k];
            }
        }
    }
}   

void gemm_tlp_simd(float * restrict a, float * restrict b, float * restrict c, int ma, int na, int nb) {
	int i, j, k;
	#pragma omp parallel for private(i, j, k) shared(a, b, c)
    for(i=0; i<ma; i++) {
        for(j=0; j<na; j++) {
            __m128 a4 = _mm_set1_ps(a[i*na + j]);
            for(k=0; k<nb; k+=4) {
                __m128 c4 = _mm_load_ps(&c[i*nb + k]);
                __m128 b4 = _mm_load_ps(&b[j*nb + k]);
                c4 = _mm_add_ps(_mm_mul_ps(a4,b4),c4);
                _mm_store_ps(&c[i*nb + k], c4);
            }
        }
    }
}

int to_x_multiple(int x, int in) {
	if(in%x != 0)
		in += 4 - in%4;
	return in;
}

int main(void) {
	int ma_in = 0;	
	int na_in = 0;
	int mb_in = 0;	
	int nb_in = 0;
	int ma = 0;
    int na = 0;
	int mb = 0;
	int nb = 0;
	int i, j, k;

	FILE *infile, *outfile;
	infile = fopen("input_small.txt", "r");
	if(infile == NULL) {
    	printf("Error in opening infile");
    	return EXIT_FAILURE;
	}

	fscanf(infile, "%d %d", &ma_in, &na_in);
	ma = to_x_multiple(4, ma_in);
	na = to_x_multiple(4, na_in);
    float *a = _mm_malloc(ma*na * sizeof *a, 64);
	for(i = 0; i < ma; i++)	{
		for(j = 0; j < na; j++) {
			if(i < ma_in && j < na_in)
				fscanf(infile, "%f", &a[i*na + j]);
			else
				a[i*na + j] = 0;
		}
	}
	fscanf(infile, "%d %d", &mb_in, &nb_in);
	mb = to_x_multiple(4, mb_in);
	nb = to_x_multiple(4, nb_in);
	float *b = _mm_malloc(na*nb * sizeof *b, 64);
	for(i = 0; i < mb; i++)	{
		for(j = 0; j < nb; j++) {
			if(i < mb_in && j < nb_in)
				fscanf(infile, "%f", &b[i*nb + j]);
			else
				b[i*nb + j] = 0;
		}
	}
	fclose(infile);
    
    float *c1 = _mm_malloc(ma*nb * sizeof *c1, 64);
    float *c2 = _mm_malloc(ma*nb * sizeof *c2, 64);
    float *c3 = _mm_malloc(ma*nb * sizeof *c2, 64);


    memset(c1, 0, ma*nb * sizeof *c1);
    memset(c2, 0, ma*nb * sizeof *c2);
    memset(c3, 0, ma*nb * sizeof *c3);
    double dtime;

    dtime = -omp_get_wtime();
    gemm(a,b,c1,ma, na, nb);
    dtime += omp_get_wtime();
    printf("time %f\n", dtime);

    dtime = -omp_get_wtime();
    gemm_tlp(a,b,c2,ma, na, nb);
    dtime += omp_get_wtime();
    printf("time %f\n", dtime);

    dtime = -omp_get_wtime();
    gemm_tlp_simd(a,b,c3,ma, na, nb);
    dtime += omp_get_wtime();
/*
	for(i = 0; i < ma; i++) {
		for(j = 0; j < nb; j++) {
			if(i < ma_in && j < nb_in)			
				printf("%.2f  ", c1[i*nb + j]);
		}		
		printf("\n");	
	}	
	printf("\n");

	for(i = 0; i < ma; i++) {
		for(j = 0; j < nb; j++) {
			if(i < ma_in && j < nb_in)			
				printf("%.2f  ", c2[i*nb + j]);
		}		
		printf("\n");	
	}	
	printf("\n");
*/
	for(i = 0; i < ma; i++) {
		for(j = 0; j < nb; j++) {
			if(i < ma_in && j < nb_in)			
				printf("%.2f  ", c3[i*nb + j]);
		}		
		printf("\n");	
	}	
	printf("\n");

    printf("time %f\n", dtime);
    printf("error %d\n", memcmp(c1,c2, ma*nb*sizeof *c1));
    printf("error %d\n", memcmp(c1,c3, ma*nb*sizeof *c1));
}
