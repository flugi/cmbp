#include "mbp.hpp"

#include <thread>
#include <vector>

using namespace std;

vector<float> f(const vector<float>&v) {
    vector<float> result;
    result.push_back(sin(v[0]+v[1]));
    result.push_back(cos(v[0]-v[1]));
    return result;
}

int main() {
    int ID = 2;
    int OD = 2;
    MBP<float> *mbp = new MBP<float>(std::vector<int>({ID,ID+OD,OD}),0);
    int N = 1000;
    vector<vector<float> > input(N, vector<float>(ID,0.0));
    vector<vector<float> > output(N, vector<float>(OD,0.0));

    for (int i=0;i<N;i++) {
        for (int j=0;j<ID;j++) {
            input[i][j] = (rand()%1000)/1000.0-0.5;
        }
        output[i] = f(input[i]);
    }
    Trainer<float> tr(mbp);
    tr.Verbose();
    tr.setInput(input, output); // this one copies the data
    tr.setParams(0,0,0,0,1000,1,100,1);
    std::thread thr([&](){tr.Learn();});
    input.clear();
    output.clear();
    cout << "input and output vectors are deleted already" << endl;
    thr.join();
    cout << mbp->Export() << endl;
    delete mbp;
}
