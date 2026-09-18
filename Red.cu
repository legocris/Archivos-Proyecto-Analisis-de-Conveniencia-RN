#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <time.h>
#if defined(_WIN32)
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define _mkdir(p) mkdir(p, 0755)
#define _chdir chdir
#endif

float Paso = 0.2;
float Gamma = 0.0;
float Parada = 0.1;
int Iteraciones = 1000000;
int CapasDesde = 2; //Desde 2
int CapasHasta = 5; // Hasta 2
int Repeticiones = 1;
int Intervalo = 10000;
int Batch = 1;
float MultiProfunda = 0.7;

cudaDeviceProp Prop;

int sprintConfig(char* str){
    int offset = 0;
    offset += sprintf(str+offset, "Paso: %f\n", Paso);
    offset += sprintf(str+offset, "Gamma: %f\n", Gamma);
    offset += sprintf(str+offset, "Parada: %f\n", Parada);
    offset += sprintf(str+offset, "Iteraciones: %d\n", Iteraciones);
    offset += sprintf(str+offset, "CapasDesde: %d\n", CapasDesde);
    offset += sprintf(str+offset, "CapasHasta: %d\n", CapasHasta);
    offset += sprintf(str+offset, "Repeticiones: %d\n", Repeticiones);
    offset += sprintf(str+offset, "Intervalo: %d\n", Intervalo);
    offset += sprintf(str+offset, "Batch: %d\n", Batch);
    offset += sprintf(str+offset, "MultiplicadorProfunda: %f\n", MultiProfunda);
    offset += sprintf(str+offset, "Tarjeta: %s\n", Prop.name);
    return offset;
}


void traspose(float *B, float *A, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            B[j*n+i] = A[i*m+j];
        }
    }
}

void printMatrix(float *A, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%.9f ", A[i*m+j]);
        }
        printf("\n");
    }
}

void printMatrixColumn(float *A, int n, int m){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            printf("%.9f ", A[i+j*n]);
        }
        printf("\n");
    }
}

__global__ void cuSigmoid(float *out, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x= out[tid];

    //Bajo condiciones ideales, podríamos evitar este check
    if (tid < total){
        out[tid] = 1.0f / (1.0f + exp(-x));
    }
}

__global__ void cuDSigmoidHadamard(float *out, float *A, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = out[tid], y = A[tid];

    if (tid < total){
      out[tid] = y * (1.0f - y) * x;
    }
}

__global__ void cuCopyRepeated(float *out, float *A, int n, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total){
      out[tid] = A[tid%n];
    }
}

//Igual que cuCopyRepeated pero para B redes. out son B bloques de (n x stride) en
//column-major; se llenan las primeras 'cols' columnas de cada bloque con el sesgo
//de esa red (A son B sesgos de largo n, contiguos).
__global__ void cuCopyRepeatedBatched(float *out, float *A, int n, int cols, int stride, int B) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * cols * B;
    if (tid < total){
      int b = tid / (n * cols);
      int rem = tid % (n * cols);
      int col = rem / n;
      int row = rem % n;
      out[b * n * stride + col * n + row] = A[b * n + row];
    }
}

float costo(float *y, float *yhat, int n){
    float sum = 0;
    for(int i=0; i<n; i++){
        sum += (y[i]-yhat[i])*(y[i]-yhat[i]);
    }
    return sum;
}

float score(float* y, float* yhat, int M, float* _returns ){
    int P = 0;
    for(int i=0; i<M; i++){
        if(y[i]>0.5) P++;
    }
    int PP=0;
    for(int i=0; i<M; i++){
        if(yhat[i]>0.5) PP++;
    }
    int TP = 0;
    for(int i=0; i<M; i++){
        if(y[i]>0.5 && yhat[i]>0.5) TP++;
    }
    float precision = (float)TP/(float)P;

    int FN = 0;
    for(int i=0; i<M; i++){
        if(y[i]<0.5 && yhat[i]>0.5) FN++;
    }
    float recall = (float)TP/(float)(PP);

    float f1 = 2.0f*precision*recall/(precision+recall);
      
    _returns[0] = f1;
    _returns[1] = precision;
    _returns[2] = recall;

    
    return f1;
}

int main_test(int M, int m, int L, const int* dim, int B) {
    time_t semilla = time(NULL);
    srand(semilla * 1000000);

    //Datos de entrada
    FILE* fp;
    //char fname[32];
    char fpath[32];
    
    int N = dim[0]; //Alto (entradas)
    //int M; //Ancho (ejemplos)
    int K = dim[L-1]; //Salidas
    //int L; //Número de capas
    //int dim[L]; Arquitecture Neuronas {N,...,K}
    //int m; //Ancho Test

    sprintf(fpath, "Bin/Train/X/%dx%d.bin", N, M);
    fp = fopen(fpath, "rb");
    float* X = (float*)malloc(sizeof(float) * N * M);
    fread(X, sizeof(float), N * M, fp);
    fclose(fp);

    sprintf(fpath, "Bin/Train/Y/%dx%d.bin", K, M);
    fp = fopen(fpath, "rb");
    float* yhat = (float*)malloc(sizeof(float) * K * M);
    fread(yhat, sizeof(float), K * M, fp);
    fclose(fp);

    int l = L-1;

    //Cadena de arquitectura ("_2_3_3_1_") y carpeta donde iran los pesos de esta corrida
    int offset_arq = 1;
    char arquitectura[128] = "_";
    for(int i=0; i<L; i++){
      offset_arq += sprintf(arquitectura+offset_arq, "%d_", dim[i]);
    }
    char pesos_dir[192], ruta[256];
    sprintf(pesos_dir, "Bin/Pesos/%s%ld_b%d", arquitectura, (long)semilla, B);

    //Preparación de datos para el backpropagation
    //Los PESOS (B réplicas contiguas por capa: red 0, red 1, ... red B-1)
    float **h_W = (float**)malloc(sizeof(float*) * l);
    float **h_WV = (float**)malloc(sizeof(float*) * l);
    float **h_B = (float**)malloc(sizeof(float*) * l);
    float **h_BV = (float**)malloc(sizeof(float*) * l);
    float **d_W = (float**)malloc(sizeof(float*) * l);
    float **d_WV = (float**)malloc(sizeof(float*) * l);
    float **d_B = (float**)malloc(sizeof(float*) * l);
    float **d_BV = (float**)malloc(sizeof(float*) * l);

    for(int i=0; i<l; i++){
      float limite = sqrt(6.0f/(dim[i]+dim[i+1]));
      int szW = dim[i] * dim[i+1];
      int szB = dim[i+1];
      h_W[i] = (float*)malloc(sizeof(float) * szW * B);
      h_WV[i] = (float*)malloc(sizeof(float) * szW * B);
      h_B[i] = (float*)malloc(sizeof(float) * szB * B);
      h_BV[i] = (float*)malloc(sizeof(float) * szB * B);

      //Inicializar pesos con valores aleatorios [-limite,limite]
      for (int b = 0; b < B; b++) {
      for (int j = 0; j < szW; j++) {
	h_W[i][b*szW+j] = ( (float) (static_cast<float>(rand())/RAND_MAX)*2.0-1.0 )*limite;
	h_WV[i][b*szW+j] = 0.0;
      }

      for (int j = 0; j < szB; j++) {
	h_B[i][b*szB+j] = 0.0;
	h_BV[i][b*szB+j] = 0.0;
      }
      }

      //En la cpu
      cudaMalloc(&d_W[i], szW * B * sizeof(float));
      cudaMemcpy(d_W[i], h_W[i], szW * B * sizeof(float), cudaMemcpyHostToDevice);

      cudaMalloc(&d_WV[i], szW * B * sizeof(float));
      cudaMemcpy(d_WV[i], h_W[i], szW * B * sizeof(float), cudaMemcpyHostToDevice);

      cudaMalloc(&d_B[i], szB * B * sizeof(float));
      cudaMemcpy(d_B[i], h_B[i], szB * B * sizeof(float), cudaMemcpyHostToDevice);

      cudaMalloc(&d_BV[i], szB * B * sizeof(float));
      cudaMemcpy(d_BV[i], h_B[i], szB * B * sizeof(float), cudaMemcpyHostToDevice);
    }

    //Guardar pesos INICIALES en Bin/Pesos/<arq>_<seed>_b<B>/ini/{W<i>,b<i>}/<filas>x<cols>.bin
    //Cada .bin lleva las B replicas contiguas, mismo layout que en memoria.
    _mkdir("Bin/Pesos");
    _mkdir(pesos_dir);
    sprintf(ruta, "%s/ini", pesos_dir); _mkdir(ruta);
    for(int i=0; i<l; i++){
        sprintf(ruta, "%s/ini/W%d", pesos_dir, i); _mkdir(ruta);
        sprintf(ruta, "%s/ini/W%d/%dx%d.bin", pesos_dir, i, dim[i+1], dim[i]);
        fp = fopen(ruta, "wb"); fwrite(h_W[i], sizeof(float), (size_t)dim[i]*dim[i+1]*B, fp); fclose(fp);

        sprintf(ruta, "%s/ini/b%d", pesos_dir, i); _mkdir(ruta);
        sprintf(ruta, "%s/ini/b%d/%dx1.bin", pesos_dir, i, dim[i+1]);
        fp = fopen(ruta, "wb"); fwrite(h_B[i], sizeof(float), (size_t)dim[i+1]*B, fp); fclose(fp);
    }

    //Variables temporales para el backpropagation
    float *d_yhat;
    cudaMalloc(&d_yhat,       dim[l] * M * B * sizeof(float));
    for(int b=0; b<B; b++)
        cudaMemcpy(d_yhat + b*dim[l]*M , yhat, dim[l] * M * sizeof(float), cudaMemcpyHostToDevice);

    float *y = (float*)malloc(sizeof(float) * dim[l] * M * B);
    float **d_ev, **d_d;
    d_ev = (float**)malloc(sizeof(float*) * L);
    d_d  = (float**)malloc(sizeof(float*) * (L-1));

    float **h_ev = (float**)malloc(sizeof(float*) * L);

    for(int i=0; i<L; i++){
        cudaMalloc(&d_ev[i],  dim[i] * M * B * sizeof(float));
        h_ev[i] = (float*)malloc( dim[i] * M * B * sizeof(float));
    }

    for(int b=0; b<B; b++)
        cudaMemcpy(d_ev[0] + b*dim[0]*M, X,    dim[0] * M * sizeof(float), cudaMemcpyHostToDevice);

    for(int i=0; i<L-1; i++){
        cudaMalloc(&d_d[i],  dim[i+1] * M * B * sizeof(float));
    }

    //Esta matriz Identidad es últil
    float *h_ID1 = (float*)malloc(sizeof(float) * 1 * 1);
    for (int i = 0; i < 1 * 1; i++) {
        h_ID1[i] = 1.0; //IdentidadF
    }
    float *d_ID1;
    cudaMalloc(&d_ID1 , 1 * 1 * sizeof(float));
    cudaMemcpy(d_ID1, h_ID1  , 1 * 1 * sizeof(float), cudaMemcpyHostToDevice);

    //Este a matriz Identidad es últil
    float *h_1M = (float*)malloc(sizeof(float) * 1 * M);
    for (int i = 0; i < 1 * M; i++) {
        h_1M[i] = 1.0; //IdentidadF
    }
    float *d_1M;
    cudaMalloc(&d_1M , 1 * M * sizeof(float));
    cudaMemcpy(d_1M, h_1M, 1 * M * sizeof(float), cudaMemcpyHostToDevice);
    
    //AHORA SÍ
    cublasHandle_t handle;
    cublasCreate(&handle);

    //Stream propio + workspace fijo de cuBLAS: necesario para capturar el paso de
    //entrenamiento como CUDA graph (durante la captura no se puede reservar memoria).
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    void *d_ws;
    size_t ws_size = 32 * 1024 * 1024;
    cudaMalloc(&d_ws, ws_size);
    cublasSetWorkspace(handle, d_ws, ws_size);

    //Un registro de costos por replica
    char** costos = (char**)malloc(sizeof(char*) * B);
    int* offset_cost = (int*)malloc(sizeof(int) * B);
    for(int b=0; b<B; b++){
        costos[b] = (char*)malloc(sizeof(char) * (Iteraciones/Intervalo + 2) * 16);
        offset_cost[b] = 0;
    }

    float multi = powf(MultiProfunda, L-3);

    //El paso se corre 2 veces fuera del entrenamiento: n=0 en vivo (fuerza toda la
    //inicializacion diferida de cuBLAS y de los kernels, que esta prohibida durante la
    //captura) y n=1 grabandolo como CUDA graph. El bucle de entrenamiento (n>=2) solo
    //reproduce el graph, que es lo que quita el overhead de lanzar orden por orden.
    cudaGraph_t graph;
    cudaGraphExec_t graexec;

    int n;
    for(n = 0; n < 2; n++){

        if (n == 1) {
            cudaStreamSynchronize(stream);
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        }

        //SGEMM: C = aAB+bC
        float alpha = 1.0f, beta = 1.0f;
        for(int i=0; i<l; i++){
            cuCopyRepeatedBatched<<<(M*dim[i+1]*B+255)/256, 256, 0, stream>>>(d_ev[i+1], d_B[i], dim[i+1], M, M, B);
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim[i+1], M, dim[i], &alpha,
                d_W[i], dim[i+1], (long long)dim[i+1]*dim[i],
                d_ev[i], dim[i], (long long)dim[i]*M,
                &beta, d_ev[i+1], dim[i+1], (long long)dim[i+1]*M, B);
            cuSigmoid<<<(M*dim[i+1]*B+255)/256, 256, 0, stream>>>(d_ev[1+i], dim[i+1] * M * B);
        }

        //Backpropagation formal.
        alpha = 1.0f; beta = -1.0f;
        cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim[l], M*B, &alpha, d_ev[l], dim[l], &beta,  d_yhat,  dim[l], d_d[l-1], dim[l]);

        cuDSigmoidHadamard<<<(M*dim[l]*B+255)/256, 256, 0, stream>>>(d_d[l-1], d_ev[l], dim[l] * M * B);

        //Ahora si en automatico
        alpha=1.0f; beta = 0.0f;
        for(int i=0; i<l-1; i++){
            cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim[l-i-1], M, dim[l-i], &alpha,
                d_W[l-i-1], dim[l-i], (long long)dim[l-i]*dim[l-i-1],
                d_d[l-i-1], dim[l-i], (long long)dim[l-i]*M,
                &beta, d_d[l-i-2], dim[l-i-1], (long long)dim[l-i-1]*M, B);
            cuDSigmoidHadamard<<<(M*dim[l-i-1]*B+255)/256, 256, 0, stream>>>(d_d[l-i-2], d_ev[l-i-1], dim[l-i-1] * M * B);
        }

        //Terminamos con los gradiantes y actualizamos los pesos
        float neg = -1.0;
        alpha = (Paso*multi)/(M); beta = Gamma;
        for(int i=0; i<l; i++){
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim[i+1], dim[i], M, &alpha,
                d_d[i], dim[i+1], (long long)dim[i+1]*M,
                d_ev[i], dim[i], (long long)dim[i]*M,
                &beta, d_WV[i], dim[i+1], (long long)dim[i+1]*dim[i], B);
            cublasSaxpy(handle, dim[i+1]*dim[i]*B, &neg, d_WV[i], 1, d_W[i], 1);

            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim[i+1], 1, M, &alpha,
                d_d[i], dim[i+1], (long long)dim[i+1]*M,
                d_1M, 1, (long long)0,
                &beta, d_BV[i], dim[i+1], (long long)dim[i+1]*1, B);
            cublasSaxpy(handle, dim[i+1]*B, &neg, d_BV[i], 1, d_B[i], 1);
        }

        if (n == 1) {
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&graexec, graph, NULL, NULL, 0);
        }
    }

    for(n = 2; n < Iteraciones; n++){
        cudaGraphLaunch(graexec, stream);

        //IMPRIMIR
        if (n % Intervalo == 0 && n>0) {
            cudaStreamSynchronize(stream);
            printf("Iteración %d\n", n);
            cudaMemcpy(y, d_ev[l], dim[l]*M*B*sizeof(float), cudaMemcpyDeviceToHost);

            for(int b=0; b<B; b++){
                float cuesta = costo(y + b*dim[l]*M, yhat, M)/M;
                offset_cost[b] += sprintf(costos[b]+offset_cost[b], "%f", cuesta);
            }
            printf("Costo: %f\n", costo(y, yhat, M)/M);
        }
    }
    cudaStreamSynchronize(stream);

    //Guardar pesos FINALES, misma estructura pero en la subcarpeta fin/
    sprintf(ruta, "%s/fin", pesos_dir); _mkdir(ruta);
    for(int i=0; i<l; i++){
        cudaMemcpy(h_W[i], d_W[i], (size_t)dim[i]*dim[i+1]*B*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B[i], d_B[i], (size_t)dim[i+1]*B*sizeof(float), cudaMemcpyDeviceToHost);

        sprintf(ruta, "%s/fin/W%d", pesos_dir, i); _mkdir(ruta);
        sprintf(ruta, "%s/fin/W%d/%dx%d.bin", pesos_dir, i, dim[i+1], dim[i]);
        fp = fopen(ruta, "wb"); fwrite(h_W[i], sizeof(float), (size_t)dim[i]*dim[i+1]*B, fp); fclose(fp);

        sprintf(ruta, "%s/fin/b%d", pesos_dir, i); _mkdir(ruta);
        sprintf(ruta, "%s/fin/b%d/%dx1.bin", pesos_dir, i, dim[i+1]);
        fp = fopen(ruta, "wb"); fwrite(h_B[i], sizeof(float), (size_t)dim[i+1]*B, fp); fclose(fp);
    }

    //Muestra el Train
    // Copiar el resultado de la última capa a la CPU
    float scores[3];
    cudaMemcpy(y, d_ev[l], dim[l]*M*B*sizeof(float), cudaMemcpyDeviceToHost);

    //EVALUAMOS EL TEST (ASUMIMOS m<=M)
    sprintf(fpath, "Bin/Test/X/%dx%d.bin", N, m);
    fp = fopen(fpath, "rb");
    fread(X, sizeof(float), N * m, fp);
    fclose(fp);

    for(int b=0; b<B; b++)
        cudaMemcpy(d_ev[0] + b*dim[0]*M, X, dim[0] * m * sizeof(float), cudaMemcpyHostToDevice);

    float* yhat_test = (float*)malloc(sizeof(float) * K * m);
    sprintf(fpath, "Bin/Test/Y/%dx%d.bin", K, m);
    fp = fopen(fpath, "rb");
    fread(yhat_test, sizeof(float), K * m, fp);
    fclose(fp);

    //Evaluando... (mismo forward batcheado, pero solo m columnas por red)
    float alpha = 1.0f, beta = 1.0f;
    for(int i=0; i<l; i++){
       cuCopyRepeatedBatched<<<(M*dim[i+1]*B+255)/256, 256, 0, stream>>>(d_ev[i+1], d_B[i], dim[i+1], m, M, B);
       cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim[i+1], m, dim[i], &alpha,
           d_W[i], dim[i+1], (long long)dim[i+1]*dim[i],
           d_ev[i], dim[i], (long long)dim[i]*M,
           &beta, d_ev[i+1], dim[i+1], (long long)dim[i+1]*M, B);
       cuSigmoid<<<(M*dim[i+1]*B+255)/256, 256, 0, stream>>>(d_ev[1+i], dim[i+1] * M * B);
    }
    cudaStreamSynchronize(stream);

    float* y_test = (float*)malloc(sizeof(float) * dim[l] * M * B);
    cudaMemcpy(y_test, d_ev[l], dim[l]*M*B*sizeof(float), cudaMemcpyDeviceToHost);

    //Un archivo Reg por replica: _<arq>_<semilla>_<b>.txt
    for(int b=0; b<B; b++){
        int offset = 0;

        char buffer[1024];
        offset += sprintf(buffer+offset, "\nArquitectura: %s\n", arquitectura);
        offset += sprintf(buffer+offset, "Semilla: %ld\n", (long)(semilla));
        offset += sprintf(buffer+offset, "Segundos: %ld\n", (long)(time(NULL)-semilla));
        offset += sprintf(buffer+offset, "NIteraciones: %d\n", n);

        offset += sprintf(buffer+offset, "\nCosto_train: %f\n", costo(y + b*dim[l]*M, yhat, M)/M);
        score(y + b*dim[l]*M, yhat, M, scores);
        offset += sprintf(buffer+offset, "Precision_train: %f\n", scores[1]);
        offset += sprintf(buffer+offset, "Recall_train: %f\n", scores[2]);
        offset += sprintf(buffer+offset, "F1_train: %f\n", scores[0]);


        offset += sprintf(buffer+offset, "\nCosto_test: %f\n", costo(y_test + b*dim[l]*M, yhat_test, m)/m);
        score(y_test + b*dim[l]*M, yhat_test, m, scores);
        offset += sprintf(buffer+offset, "Precision_test: %f\n", scores[1]);
        offset += sprintf(buffer+offset, "Recall_test: %f\n", scores[2]);
        offset += sprintf(buffer+offset, "F1_test: %f\n \n", scores[0]);
        printf(buffer);

        char nombre_archivo[128];
        sprintf(nombre_archivo, "Reg/%s%ld_%d.txt", arquitectura, (long)semilla, b);
        fp = fopen(nombre_archivo, "w");
        sprintConfig(buffer+offset*sizeof(char));
        fprintf(fp, buffer);
        fprintf(fp, "\nCostos:\n");
        fprintf(fp, costos[b]);
        fclose(fp);
    }

    cudaGraphExecDestroy(graexec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_ws);
    free(yhat_test);
    free(y_test);
    for(int b=0; b<B; b++) free(costos[b]);
    free(costos);
    free(offset_cost);
      
    cublasDestroy(handle);  // Destroy cuBLAS handle (optional)

    //Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Free memory
    for(int i=0; i<L-1; i++){
        cudaFree(d_W[i]);
        free(h_W[i]);
    }
    cudaFree(d_yhat);
    cudaFree(d_ID1);
    cudaFree(d_1M);

    for(int i=0; i<L; i++){
        cudaFree(d_ev[i]);
    }
    cudaFree(d_yhat);
    for(int i=0; i<L-1; i++){
        cudaFree(d_d[i]);
    }
    free(X);
    free(y);
    free(yhat);
    free(h_ID1);
    free(h_1M);

    return 0;
}

char* getFname(const char* fpath) {
    FILE* fp;
    static char fname[32]; 
    char cmd[256]; 

    #if defined(_WIN32)
        sprintf(cmd, "cd \"%s\" && dir /b *.bin", fpath);
        fp = _popen(cmd, "r");
    #elif defined(__linux__)
        sprintf(cmd, "cd \"%s\" && ls -1 *.bin", fpath);
        fp = popen(cmd, "r");
    #endif

    fgets(fname, sizeof(fname), fp);

    #if defined(_WIN32)
        _pclose(fp);
    #elif defined(__linux__)
        pclose(fp);
    #endif

    return fname;
}

void mostrar_ayuda(const char *nombre_programa) {
    printf("Uso: %s [CARPETA] [OPCIONES]...\n", nombre_programa);
    printf("  CARPETA             Directorio del problema (opcional, primer argumento). Por defecto: el actual.\n");
    printf("Opciones (nombre corto | nombre largo):\n");
    printf("  -p | --Paso         Valor de Paso (float). Actual: %f\n", Paso);
    printf("  -g | --Gamma        Inercia Gamma (float). Actual: %f\n", Gamma);
    printf("  -r | --Parada       Valor de Parada (float). Actual: %f\n", Parada);
    printf("  -i | --Iteraciones  Número de Iteraciones (int). Actual: %d\n", Iteraciones);
    printf("  -cd| --CapasDesde   Capas Iniciales Experimento (int). Actual: %d\n", CapasDesde);
    printf("  -ch| --CapasHasta   Capas Finales Finales (int). Actual: %d\n", CapasHasta);
    printf("  -R | --Repeticiones Número de Repeticiones Experimento (int). Actual: %d\n", Repeticiones);
    printf("  -I | --Intervalo    Intervalo de registro (int). Actual: %d\n", Intervalo);
    printf("  -b | --Batch        Redes entrenadas en paralelo por repeticion (int). Actual: %d\n", Batch);
    printf("  -h | --ayuda        Mostrar esta ayuda.\n");
}

int main(int argc, char *argv[]) {
    // Primer argumento opcional: la carpeta del problema (si no empieza con '-').
    // Si se da, nos movemos ahi y el resto (Bin/, Reg/, Bin/Pesos/) queda relativo a ella.
    int arg0 = 1;
    if (argc > 1 && argv[1][0] != '-') {
        if (_chdir(argv[1]) != 0) {
            fprintf(stderr, "Error: no se pudo entrar a la carpeta %s\n", argv[1]);
            return 1;
        }
        arg0 = 2;
    }

    // El bucle empieza despues de argv[0] (y de la carpeta, si la hubo)
    for (int i = arg0; i < argc; i++) {
        
        // El argumento actual es el flag (ej: "-p" o "--Paso")
        char *arg = argv[i];

        // ------------------ 1. Paso (float) ------------------
        if (strcmp(arg, "-p") == 0 || strcmp(arg, "--Paso") == 0) {
            if (i + 1 < argc) {
                Paso = atof(argv[++i]); // Incrementa 'i' para tomar el siguiente argumento como valor
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
	// ------------------ 1. Gamma (float) ------------------
        else if (strcmp(arg, "-g") == 0 || strcmp(arg, "--Gamma") == 0) {
            if (i + 1 < argc) {
                Gamma = atof(argv[++i]); 
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 2. Parada (float) ------------------
        else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--Parada") == 0) {
            if (i + 1 < argc) {
                Parada = atof(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 3. Iteraciones (int) ------------------
        else if (strcmp(arg, "-i") == 0 || strcmp(arg, "--Iteraciones") == 0) {
            if (i + 1 < argc) {
                Iteraciones = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 4. CapasDesde (int) ------------------
        else if (strcmp(arg, "-cd") == 0 || strcmp(arg, "--CapasDesde") == 0) {
            if (i + 1 < argc) {
                CapasDesde = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 5. CapasHasta (int) ------------------
        else if (strcmp(arg, "-ch") == 0 || strcmp(arg, "--CapasHasta") == 0) {
            if (i + 1 < argc) {
                CapasHasta = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 6. Repeticiones (int) ------------------
        else if (strcmp(arg, "-R") == 0 || strcmp(arg, "--Repeticiones") == 0) {
            if (i + 1 < argc) {
                Repeticiones = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 7. Intervalo (int) ------------------
        else if (strcmp(arg, "-I") == 0 || strcmp(arg, "--Intervalo") == 0) {
            if (i + 1 < argc) {
                Intervalo = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
	// ------------------ 7. Multiplicadorprofunda (int) ------------------
        else if (strcmp(arg, "-MP") == 0 || strcmp(arg, "--MultiProfunda") == 0) {
            if (i + 1 < argc) {
                MultiProfunda = atof(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 8. Batch (int) ------------------
        else if (strcmp(arg, "-b") == 0 || strcmp(arg, "--Batch") == 0) {
            if (i + 1 < argc) {
                Batch = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: El flag %s requiere un valor.\n", arg);
                return 1;
            }
        }
        // ------------------ 8. Ayuda ------------------
        else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--ayuda") == 0) {
            mostrar_ayuda(argv[0]);
            return 0;
        }
        // ------------------ Argumento desconocido ------------------
        else {
            fprintf(stderr, "Error: Argumento desconocido: %s\n", arg);
            mostrar_ayuda(argv[0]);
            return 1;
        }
    }

    cudaGetDeviceProperties(&Prop, 0);

    
    
    printf("--- Configuración Actual ---\n");
    char config[512];
    sprintConfig(config); 
    printf("%s", config);
    printf("----------------------------\n");
    
    // --- LÓGICA PRINCIPAL ---

    int M, N, K, m;
    
    //Obtenemos los parametros a partir de los nombres de los archivos
    printf("Dimensiones Datos\n");
    
    sscanf(getFname("Bin/Train/X"), "%dx%d", &N, &M);
    printf("trainx: %dx%d \n", N, M);

    sscanf(getFname("Bin/Train/Y"), "%dx%d", &K, &M);
    printf("trainy: %dx%d \n", K, M);

    sscanf(getFname("Bin/Test/X"), "%dx%d", &N, &m);
    printf("test_x: %dx%d \n", N, m);

    int dim1[3];
    int dim2[100];
    for(int i = CapasDesde; i<=CapasHasta; i++){
      dim1[0] = N; dim1[1] = (N+K)*i; dim1[2] = K;
      
      dim2[0] = N; dim2[1+i] = K;
      for(int t = 0; t<i; dim2[1+(t++)] = N+K){}
      
      for(int j = 0; j<Repeticiones; j++){
	//Ahora sí, la prueba.
	printf("\n\n--- Test Angosta %d.%d ---\n", i, j);
	main_test(M, m, 3, dim1, Batch);

	printf("\n\n--- Test Profunda %d.%d ---\n", i, j);
	main_test(M, m, i+2, dim2, Batch);
      }
    }
    
    return 0;
}
