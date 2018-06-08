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
#include <pthread.h>
#include <omp.h>

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

void *Strassen_partial_multiply(void *data);

struct arg {
	int id; 
	int thread_count;
	unsigned ma, na, mb, nb;
	double **A11, **A12, **A21, **A22;
	double **B11, **B12, **B21, **B22;
};

double **P, **Q, **R, **S, **T, **U, **V;
double start, end;
double cpu_time_used;

int thread_count=1;

int main(int argc, char*argv []) {
	unsigned ma = 0;	
	unsigned na = 0;
	unsigned mb = 0;	
	unsigned nb = 0;
	int i = 0; // used in loop
	int j = 0; // used in loop
	double **A, **B, **C;
	double **A11, **A12, **A21, **A22;
	double **B11, **B12, **B21, **B22;
	double **C11, **C12, **C21, **C22;
	double **C111, **C112, **C221, **C222;
	double **P1, **P2, **Q1, **R1, **S1, **T1, **U1, **U2, **V1, **V2;
	FILE *infile, *outfile;
	struct arg *data;
	int thread_count = 0;
	printf("thread count: ");
	scanf("%d", &thread_count);
	if(thread_count > 7) 
		thread_count = 7;
	else if(thread_count < 1)
		thread_count = 1;
	pthread_t pt[thread_count];
	
	infile = fopen("input1000.txt", "r");
	if(infile == NULL) {
    	printf("Error in Opening infile");
    	return EXIT_FAILURE;
	}

	fscanf(infile, "%u %u", &ma, &na);	// read A's dimensions
	
	malloc_matrix(ma, na, &A);
	
	for(i = 0; i < ma; i++)	// read A's content
		for(j = 0; j < na; j++)
			fscanf(infile, "%lf", &A[i][j]);
	
	fscanf(infile, "%u %u", &mb, &nb);	// read B's dimensions
	
	if(na != mb) {	// check dimension
		printf("matrices dimension error \n");		
		exit(1);	
	}

	malloc_matrix(mb, nb, &B);
	
	for(i = 0; i < mb; i++)	// read B's content
		for(j = 0; j < nb; j++)
			fscanf(infile, "%lf", &B[i][j]);
	fclose(infile);
	
	
	malloc_matrix(ma, nb, &C);
	
	/*** do multiplication in different methods ***/
	outfile = fopen("output.txt", "w+");
	
	// method 1 : traditional 
	/*
	start = omp_get_wtime();

	multiply(ma, na, A, mb, nb, B, C);
	 
	end = omp_get_wtime();
    cpu_time_used = ((double) (end - start));
	fprintf(outfile, "tradition cost %f \n", cpu_time_used);
	*/
	// method 1 end 

	// method 2 : Strassen algorithm 

	start = omp_get_wtime();

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

	// calculate (may do parallel below)
	matrix_split(ma, na, A, A11, A12, A21, A22);
	matrix_split(mb, nb, B, B11, B12, B21, B22);

	thread_count = strtol(argv[1],NULL,10);
	
	for(i = 0; i < thread_count; i++) {
		data = malloc(sizeof(struct arg));
		*data = (struct arg){.id = i, .thread_count = thread_count,
						.ma = ma, .na = na, .mb = mb, .nb = nb,
						.A11 = A11, .A12 = A12, .A21 = A21, .A22 = A22,
						.B11 = B11, .B12 = B12, .B21 = B21, .B22 = B22};
		pthread_create(&pt[i], NULL, Strassen_partial_multiply,	(void *) data);
		
	}
	
	for(i = 0; i < thread_count; i++) {
		pthread_join(pt[i], NULL);
	}

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
	
	end = omp_get_wtime();
    cpu_time_used = ((double) (end - start));
	fprintf(outfile, "Strassen cost %f \n", cpu_time_used);
	// method 2 end

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


void *Strassen_partial_multiply(void *data) {
	struct arg *in = (struct arg *)data;
	double **P1, **P2, **Q1, **R1, **S1, **T1, **U1, **U2, **V1, **V2;
	int size, taskLB, taskUB;
	if(7/in->thread_count == 0)
		size = 7/in->thread_count;
	else
		size = 7/in->thread_count + 1;
	taskLB = in->id*size;
	taskUB = in->id*size + size;

	printf("taskUB = %d, taskLB = %d \n", taskUB, taskLB);
	
	printf("%d ready!! \n", in->id);

	if(taskLB <= 0 && 0 < taskUB) {
		printf("%d do task 0 \n", in->id);
		malloc_matrix(in->ma/2, in->na/2, &P1);
		malloc_matrix(in->mb/2, in->nb/2, &P2);
		malloc_matrix(in->ma/2, in->nb/2, &P);
		add(in->ma/2, in->na/2, in->A11, in->A22, P1); // P
		add(in->mb/2, in->nb/2, in->B11, in->B22, P2);
		multiply(in->ma/2, in->na/2, P1, in->mb/2, in->nb/2, P2, P);
	} 
	if(taskLB <= 1 && 1 < taskUB) {
		printf("%d do task 1 \n", in->id);
		malloc_matrix(in->ma/2, in->na/2, &Q1);
		malloc_matrix(in->ma/2, in->nb/2, &Q);
		add(in->ma/2, in->na/2, in->A21, in->A22, Q1); // Q
		multiply(in->ma/2, in->na/2, Q1, in->mb/2, in->nb/2, in->B11, Q);
	}
	if(taskLB <= 2 && 2 < taskUB) {
		printf("%d do task 2 \n", in->id);
		malloc_matrix(in->mb/2, in->nb/2, &R1);
		malloc_matrix(in->ma/2, in->nb/2, &R);
		sub(in->mb/2, in->nb/2, in->B12, in->B22, R1); // R
		multiply(in->ma/2, in->na/2, in->A11, in->mb/2, in->nb/2, R1, R);
	}
	if(taskLB <= 3 && 3 < taskUB) {
		printf("%d do task 3 \n", in->id);
		malloc_matrix(in->mb/2, in->nb/2, &S1);
		malloc_matrix(in->ma/2, in->nb/2, &S);
		sub(in->mb/2, in->nb/2, in->B21, in->B11, S1); // S
		multiply(in->ma/2, in->na/2, in->A22, in->mb/2, in->nb/2, S1, S);
	}
	if(taskLB <= 4 && 4 < taskUB) {
		printf("%d do task 4 \n", in->id);
		malloc_matrix(in->ma/2, in->na/2, &T1);
		malloc_matrix(in->ma/2, in->nb/2, &T);
		add(in->ma/2, in->na/2, in->A11, in->A12, T1); // T
		multiply(in->ma/2, in->na/2, T1, in->mb/2, in->nb/2, in->B22, T);
	}
	if(taskLB <= 5 && 5 < taskUB) {
		printf("%d do task 5 \n", in->id);
		malloc_matrix(in->ma/2, in->na/2, &U1);
		malloc_matrix(in->mb/2, in->nb/2, &U2);
		malloc_matrix(in->ma/2, in->nb/2, &U);
		sub(in->ma/2, in->na/2, in->A21, in->A11, U1); // U
		add(in->mb/2, in->nb/2, in->B11, in->B12, U2);
		multiply(in->ma/2, in->na/2, U1, in->mb/2, in->nb/2, U2, U);
	}
	if(taskLB <= 6 && 6 < taskUB) {
		printf("%d do task 6 \n", in->id);
		malloc_matrix(in->ma/2, in->na/2, &V1);
		malloc_matrix(in->mb/2, in->nb/2, &V2);
		malloc_matrix(in->ma/2, in->nb/2, &V);
		sub(in->ma/2, in->na/2, in->A12, in->A22, V1); // V
		add(in->mb/2, in->nb/2, in->B21, in->B22, V2);
		multiply(in->ma/2, in->na/2, V1, in->mb/2, in->nb/2, V2, V);
	}
	pthread_exit(NULL);
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
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat3,mat2,mat1)
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			mat3[i][j] = mat1[i][j] + mat2[i][j];	
}

void sub(int m, int n, double **mat1, double **mat2, double **mat3) {	
	
	int i, j;
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat3,mat2,mat1)
	for(i = 0; i < m; i++)
		for(j = 0; j < n; j++)
			mat3[i][j] = mat1[i][j] - mat2[i][j];	
}

void multiply(int m1, int n1, double **mat1, int m2, int n2, double **mat2, double **mat3) {	
	int i, j, k;
	
	#pragma omp parallel for num_threads(thread_count)private(i,j) shared(mat3)
	for(i = 0; i < m1; i++)
		for(j = 0; j < n2; j++)
			mat3[i][j] = 0;	
	
	#pragma omp parallel for num_threads(thread_count) private(i,j,k) shared(mat3,mat1,mat2,m1,n1,n2)
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
	
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat11,m,n)
	for(i = 0; i < m/2; i++)
		for(j = 0; j < n/2; j++)
			mat11[i][j] = mat[i][j];
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat12,m,n)
	for(i = 0; i < m/2; i++)
		for(j = n/2; j < n; j++)
			mat12[i][j - n/2] = mat[i][j];
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat21,m,n)
	for(i = m/2; i < m; i++)
		for(j = 0; j < n/2; j++)
			mat21[i - m/2][j] = mat[i][j];
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat22,m,n)
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
	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat11,n,m)
	for(i = 0; i < m/2; i++)
		for(j = 0; j < n/2; j++)
			mat[i][j] = mat11[i][j];

	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat12,n,m)
	for(i = 0; i < m/2; i++)
		for(j = n/2; j < n; j++)
			mat[i][j] = mat12[i][j - n/2];

	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat21,n,m)
	for(i = m/2; i < m; i++)
		for(j = 0; j < n/2; j++)
			mat[i][j] = mat21[i - m/2][j];

	#pragma omp parallel for num_threads(thread_count) private(i,j) shared(mat,mat22,n,m)
	for(i = m/2; i < m; i++)
		for(j = n/2; j < n; j++)
			mat[i][j] = mat22[i - m/2][j - n/2];
}
