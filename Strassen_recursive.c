/*
#row, #column: only deal with the two input matrices : ma na mb nb

matrix variables in the following code:
	input matrix : A, B
	output matrix: C
	sub-matrix of A: A11, A12, A21, A22
	
	Strassen algorithm:

	P1 = A11 + A22
	P2 = B11 + B22
	P = P1 * P2
	
	Q1 = A21 + A22
	Q = Q1 * B11

	R1 = B12 - B22
	R = A11 * R1
	
	S1 = B21 - B11
	S = A22 * S1

	T1 = A11 + A12
	T = T1 * B22
	
	U1 = A21 - A11
	U2 = B11 + B12
	U = U1 * U2

	V1 = A12 - A22
	V2 = B21 + B22
	V = V1 * V2

	C111 = P + S
	C112 = C111 - T
	C11 = C112 + V

	C12 = R + T

	C21 = Q + S

	C221 = P + R
	C222 =  C221 - Q
	C22 = C222 + U

*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double malloc_cpu_time_used;

void malloc_matrix(int m, int n, double ***matptr);
void add(int m, int n, double **mat1, double **mat2, double **mat3);
void sub(int m, int n, double **mat1, double **mat2, double **mat3);

// split one matrix to four same-shape sub-matrices
void matrix_split(int m, int n, double **mat,
					double **mat1, double **mat2,
					double **mat3, double **mat4);

// merge four same-shape sub-matrices to one matrix
void matrix_merge(int m, int n, double **mat,
					double **mat1, double **mat2,
					double **mat3, double **mat4);

void multiply(int m1, int n1, double **mat1, int m2, int n2, double **mat2, double **mat3);

void Strassen(int ma, int na, double **A, int mb, int nb, double **B, double **C);

int two_power(int in);

int main() {
	malloc_cpu_time_used = 0;
	unsigned ma_in = 0;	
	unsigned na_in = 0;
	unsigned mb_in = 0;	
	unsigned nb_in = 0;
	unsigned ma = 0;	
	unsigned na = 0;
	unsigned mb = 0;	
	unsigned nb = 0;
	int i = 0; // used in loop
	int j = 0; // used in loop
	double **A, **B, **C;
	FILE *infile, *outfile;
	clock_t start, end;
    double cpu_time_used;
	
	infile = fopen("input.txt", "r");
	if(infile == NULL) {
    	printf("Error in Opening infile");
    	return EXIT_FAILURE;
	}

	fscanf(infile, "%u %u", &ma_in, &na_in);	// read A's dimensions
	
	ma = two_power(ma_in);
	na = two_power(na_in);

	malloc_matrix(ma, na, &A);	
	
	for(i = 0; i < ma; i++)	// read A's content
		for(j = 0; j < na; j++) {
			if(i < ma_in && j < na_in)
				fscanf(infile, "%lf", &A[i][j]);
			else
				A[i][j] = 0;
		}

	fscanf(infile, "%u %u", &mb_in, &nb_in);	// read B's dimensions
	
	if(na_in != mb_in) {	// check dimension
		printf("matrices dimension error \n");		
		exit(1);	
	}

	mb = two_power(mb_in);
	nb = two_power(nb_in);

	malloc_matrix(mb, nb, &B);
	
	for(i = 0; i < mb; i++)	// read B's content
		for(j = 0; j < nb; j++) {
			if(i < mb_in && j < nb_in)
				fscanf(infile, "%lf", &B[i][j]);
			else
				B[i][j] = 0;
		}
	
	
	malloc_matrix(ma, nb, &C);
	
	/*** do multiplication in different methods ***/
	outfile = fopen("output.txt", "w+");
	
	// method 1 : traditional 
	
	start = clock();

	multiply(ma, na, A, mb, nb, B, C);
	 
	end = clock();
    cpu_time_used = ((double) (end - start));
	fprintf(outfile, "tradition cost %f clicks \n", cpu_time_used);
	// method 1 end 

	// method 2 : Strassen algorithm 
/*
	start = clock();
	
	Strassen(ma, na, A, mb, nb, B, C);
	
	end = clock();
    cpu_time_used = ((double) (end - start));
	fprintf(outfile, "Strassen cost %f clicks \n", cpu_time_used); */
	// method 2 end
	
	printf("malloc_cpu_time_used = %lf \n", malloc_cpu_time_used);	

	// output the result matrix C 
	fprintf(outfile, "%d %d \n", ma, nb);
	for(i = 0; i < ma; i++) {
		for(j = 0; j < nb; j++)
			fprintf(outfile, "%8.2lf ", C[i][j]);
		fprintf(outfile, "\n");	
	}
	fclose(outfile);

	return 0;
}

int two_power(int in) {
	int expo = 1;
	while(1) {
		if(in <= expo)
			return expo;
		expo *= 2;
	}
}

void Strassen(int ma, int na, double **A, int mb, int nb, double **B, double **C) {

	clock_t start, end;
    

	int i = 0;
	int j = 0;

	if(ma == 1) {
		C[0][0] = 0;
		for(i = 0; i < mb; i++)
			for(j = 0; j < na; j++)
				C[0][j] += A[0][j] * B[i][0];
		return;
	} else if(na == 1) {
		for(i = 0; i < ma; i++)
			for(j = 0; j < nb; j++)
				C[i][j] += A[i][0] * B[0][j];
		return;
	} else if(nb == 1) {
		for(i = 0; i < mb; i++)
			for(j = 0; j < na; j++)
				C[i][j] += A[i][j] * B[i][0];
		return;
	}

	double **A11, **A12, **A21, **A22;
	double **B11, **B12, **B21, **B22;
	double **C11, **C12, **C21, **C22;
	double **C111, **C112, **C221, **C222;
	double **P1, **P2, **P, **Q1, **Q, **R1, **R, **S1, **S, **T1, **T, **U1, **U2, **U, **V1, **V2, **V;

	// malloc
	start = clock();

	malloc_matrix(ma/2, na/2, &A11);
	malloc_matrix(ma/2, na/2, &A12);
	malloc_matrix(ma/2, na/2, &A21);
	malloc_matrix(ma/2, na/2, &A22);
	malloc_matrix(mb/2, nb/2, &B11);
	malloc_matrix(mb/2, nb/2, &B12);
	malloc_matrix(mb/2, nb/2, &B21);
	malloc_matrix(mb/2, nb/2, &B22);
	malloc_matrix(ma/2, nb/2, &C11);
	malloc_matrix(ma/2, nb/2, &C12);
	malloc_matrix(ma/2, nb/2, &C21);
	malloc_matrix(ma/2, nb/2, &C22);
	
	malloc_matrix(ma/2, nb/2, &C111);
	malloc_matrix(ma/2, nb/2, &C112);
	malloc_matrix(ma/2, nb/2, &C221);
	malloc_matrix(ma/2, nb/2, &C222);

	malloc_matrix(ma/2, na/2, &P1);
	malloc_matrix(mb/2, nb/2, &P2);
	malloc_matrix(ma/2, nb/2, &P);

	malloc_matrix(ma/2, na/2, &Q1);
	malloc_matrix(ma/2, nb/2, &Q);

	malloc_matrix(mb/2, nb/2, &R1);
	malloc_matrix(ma/2, nb/2, &R);

	malloc_matrix(mb/2, nb/2, &S1);
	malloc_matrix(ma/2, nb/2, &S);

	malloc_matrix(ma/2, na/2, &T1);
	malloc_matrix(ma/2, nb/2, &T);

	malloc_matrix(ma/2, na/2, &U1);
	malloc_matrix(mb/2, nb/2, &U2);
	malloc_matrix(ma/2, nb/2, &U);

	malloc_matrix(ma/2, na/2, &V1);
	malloc_matrix(mb/2, nb/2, &V2);
	malloc_matrix(ma/2, nb/2, &V);
	
	end = clock();
	malloc_cpu_time_used += ((double) (end - start));

	// calculate (may do parallel below)
	matrix_split(ma, na, A, A11, A12, A21, A22);
	matrix_split(mb, nb, B, B11, B12, B21, B22);

	add(ma/2, na/2, A11, A22, P1); // P
	add(mb/2, nb/2, B11, B22, P2);
	Strassen(ma/2, na/2, P1, mb/2, nb/2, P2, P);

	add(ma/2, na/2, A21, A22, Q1); // Q
	Strassen(ma/2, na/2, Q1, mb/2, nb/2, B11, Q);
	
	sub(mb/2, nb/2, B12, B22, R1); // R
	Strassen(ma/2, na/2, A11, mb/2, nb/2, R1, R);
	
	sub(mb/2, nb/2, B21, B11, S1); // S
	Strassen(ma/2, na/2, A22, mb/2, nb/2, S1, S);
	
	add(ma/2, na/2, A11, A12, T1); // T
	Strassen(ma/2, na/2, T1, mb/2, nb/2, B22, T);
	
	sub(ma/2, na/2, A21, A11, U1); // U
	add(mb/2, nb/2, B11, B12, U2);
	Strassen(ma/2, na/2, U1, mb/2, nb/2, U2, U);

	sub(ma/2, na/2, A12, A22, V1); // V
	add(mb/2, nb/2, B21, B22, V2);
	Strassen(ma/2, na/2, V1, mb/2, nb/2, V2, V);

	add(ma/2, nb/2, P, S, C111); // C11
	sub(ma/2, nb/2, C111, T, C112);
	add(ma/2, nb/2, C112, V, C11);

	add(ma/2, nb/2, R, T, C12); // C12

	add(ma/2, nb/2, Q, S, C21); // C21

	add(ma/2, nb/2, P, R, C221); // C22
	sub(ma/2, nb/2, C221, Q, C222);
	add(ma/2, nb/2, C222, U, C22);

	// merge to C
	matrix_merge(ma, nb, C, C11, C12, C21, C22);
}

void malloc_matrix(int m, int n, double ***matptr) {
	/*
	// the memory of the 2d array is not consecutive (each row)
	int i;	
	*matptr = malloc(m * sizeof(double *));
	for(i = 0; i < m; i++) 
		(*matptr)[i] = malloc(n * sizeof(double));
	*/
	
	// the memory of the 2d array is consecutive (each row)
	int i;
	double *tmp;	
	tmp = malloc(m * n * sizeof(double *));
	*matptr = malloc(m * sizeof(double *));
	for(i = 0; i < m; i++)
		(*matptr)[i] = &(tmp[n*i]);
	
}

void add(int m, int n, double **mat1, double **mat2, double **mat3) {	
	int i, j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			mat3[i][j] = mat1[i][j] + mat2[i][j];	
}

void sub(int m, int n, double **mat1, double **mat2, double **mat3) {	
	int i, j;
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			mat3[i][j] = mat1[i][j] - mat2[i][j];	
}

void multiply(int m1, int n1, double **mat1, int m2, int n2, double **mat2, double **mat3) {	
	int i, j, k;
	for(i = 0; i < m1; i++)
		for(j = 0; j < n2; j++)
			mat3[i][j] = 0;	
	for(i = 0; i < m1; i++)
		for(j = 0; j < n1; j++)
			for(k = 0; k < n2; k++)
				mat3[i][k] += mat1[i][j] * mat2[j][k];
}

void matrix_split(int m, int n, double **mat,
					double **mat11, double **mat12,
					double **mat21, double **mat22) {
	int i, j;
	if(m % 2 != 0 || n % 2 != 0) {
		printf("m = %d, n = %d \n", m, n);
		printf("error, cannot be split to 4 sub-matrices \n");
		exit(1);
	}
	
	
	for(i = 0; i < m/2; i++)
		for(j = 0; j < n/2; j++)
			mat11[i][j] = mat[i][j];

	for(i = 0; i < m/2; i++)
		for(j = n/2; j < n; j++)
			mat12[i][j - n/2] = mat[i][j];

	for(i = m/2; i < m; i++)
		for(j = 0; j < n/2; j++)
			mat21[i - m/2][j] = mat[i][j];

	for(i = m/2; i < m; i++)
		for(j = n/2; j < n; j++)
			mat22[i - m/2][j - n/2] = mat[i][j];
	
}

void matrix_merge(int m, int n, double **mat,
					double **mat11, double **mat12,
					double **mat21, double **mat22) {
	int i, j;
	if(m % 2 != 0 || n % 2 != 0) {
		printf("error, cannot merge by 4 sub-matrices \n");
		exit(1);
	}
	for(i = 0; i < m/2; i++)
		for(j = 0; j < n/2; j++)
			mat[i][j] = mat11[i][j];

	for(i = 0; i < m/2; i++)
		for(j = n/2; j < n; j++)
			mat[i][j] = mat12[i][j - n/2];

	for(i = m/2; i < m; i++)
		for(j = 0; j < n/2; j++)
			mat[i][j] = mat21[i - m/2][j];

	for(i = m/2; i < m; i++)
		for(j = n/2; j < n; j++)
			mat[i][j] = mat22[i - m/2][j - n/2];
}
