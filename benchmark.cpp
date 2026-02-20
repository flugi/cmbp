#define MBP_BENCHMARK
#include "mbp.hpp"

#include <thread>
#include <vector>

using namespace std;

#define REAL float

vector<REAL> f(int OD, const vector<REAL>&v) {
    vector<REAL> result(OD);
    for (int i=0;i<OD;i++) {
        result[i] = sin(v[(i+1)%v.size()]+i)*.5;
    }
    return result;
}

const int threadcount = 16;


int main() {
    int ID = 5;
    int OD = 5;
    MBP<REAL,threadcount > *mbp =
        new MBP<REAL,threadcount > (std::vector<int>({ID,(ID+OD)*(ID+OD),OD}),0);
    int N = 4000;
    vector<vector<REAL> > input(N, vector<REAL>(ID,0.0));
    vector<vector<REAL> > output(N, vector<REAL>(OD,0.0));

    for (int i=0;i<N;i++) {
        for (int j=0;j<ID;j++) {
            input[i][j] = (rand()%1600)/1000.0-0.8;
        }
        output[i] = f(OD, input[i]);
    }
    Trainer<REAL,threadcount > tr(mbp);
    tr.Verbose();
    tr.setInput(input, output); // this one copies the data
    tr.setParams(0,0,0,0,1000,1,100,1);
    //    void Tune(double P_Sat, double P_ASat, double P_R, double P_E0, double P_A0, double P_Psi,double P_Beta, double P_G, double P_Ka, double P_Kd) {
    //tr.Tune(1.0, 1.47,2.0, 0.75, 0.9, 1.05,0.7, 1.05,0.7,0.07);

    //tr.Tune(0.8, 0.47,0.1, 0.75, 0.5, 1.05,0.7, 1.05,0.1,0.01);

    tr.Learn();
//    cout << mbp->Export() << endl;
    delete mbp;
}
