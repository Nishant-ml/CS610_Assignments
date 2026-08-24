#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include <cuda.h>
using namespace std;

#define ITER_CHUNK_SIZE (1<<27)
#define NUM_VAR 10
#define THRESHOLD (numeric_limits<double>::epsilon())

struct EvalOp {
    const double* lim;
    const long long* dim;
    const double* mat;
    const double* grid;

    __host__ __device__
    EvalOp(const double* l,const long long* d,const double* m,const double* g)
        :lim(l),dim(d),mat(m),grid(g){}

    __device__ bool operator()(long long id) const {
        long long idx[10];
        double var[10];
        double acc[10]={0.0};

        long long tmp=id;
        for(int k=9;k>=0;k--){
            idx[k]=tmp%dim[k];
            tmp/=dim[k];
            var[k]=grid[3*k]+idx[k]*grid[3*k+2];
        }

        bool ok=true;
        for(int r=0;r<10;r++){
            double v=0;
            for(int c=0;c<10;c++){
                v+=mat[r*12+c]*var[c];
            }
            v-=mat[r*12+10];
            ok&=(fabs(v)<lim[r]);
        }
        return ok;
    }
};

int main() {
    thrust::host_vector<double> hM(120),hG(30);

    FILE* f1=fopen("./disp.txt","r");
    if(!f1){ printf("Error: could not open file\n"); return 1; }
    for(int i=0;i<120 && fscanf(f1,"%lf",&hM[i])==1;i++);
    fclose(f1);

    FILE* f2=fopen("./grid.txt","r");
    if(!f2){ printf("Error: could not open file\n"); return 1; }
    for(int j=0;j<30 && fscanf(f2,"%lf",&hG[j])==1;j++);
    fclose(f2);

    thrust::device_vector<double> dM=hM,dG=hG;
    thrust::device_vector<double> dL(10);
    thrust::device_vector<long long> dDim(11);

    double s=0.3;
    for(int i=0;i<10;i++) dL[i]=s*dM[11+i*12];

    long long total=1;
    for(int i=0;i<10;i++){
        dDim[i]=floor((dG[3*i+1]-dG[3*i])/dG[3*i+2]);
        total*=dDim[i];
    }
    dDim[10]=total;

    long long count=0;

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    ofstream out("results-vd.txt");
    out<<fixed<<setprecision(6);

    EvalOp fn(
        thrust::raw_pointer_cast(dL.data()),
        thrust::raw_pointer_cast(dDim.data()),
        thrust::raw_pointer_cast(dM.data()),
        thrust::raw_pointer_cast(dG.data())
    );

    for(long long cs=0;cs<total;cs+=ITER_CHUNK_SIZE){
        long long ce=min(cs+ITER_CHUNK_SIZE,total);

        thrust::counting_iterator<long long> b(cs),e(ce);
        thrust::device_vector<long long> buf(ce-cs);

        auto it=thrust::copy_if(b,e,buf.begin(),fn);
        int valid=it-buf.begin();
        count+=valid;

        if(valid){
            thrust::host_vector<long long> hv(valid);
            thrust::copy(buf.begin(),it,hv.begin());

            for(int p=0;p<valid;p++){
                double vv[NUM_VAR];
                long long v=hv[p];

                for(int t=NUM_VAR-1;t>=0;t--){
                    vv[t]=hG[3*t]+(v%dDim[t])*hG[3*t+2];
                    v/=dDim[t];
                }

                for(int z=0;z<NUM_VAR;z++){
                    out<<vv[z];
                    if(z<NUM_VAR-1) out<<"\t";
                    else out<<"\n";
                }
            }
        }
    }

    out.close();
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms=0;
    cudaEventElapsedTime(&ms,t0,t1);

    cout<<"Kernel time "<<ms*1e-3<<"s\n";
    cout<<"Result pnts "<<count<<endl;

    return EXIT_SUCCESS;
}

