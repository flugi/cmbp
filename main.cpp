#include "mbp.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
using namespace std;

const bool extratest = false;
const bool exporttest = false;
const bool exportlearntest = false;

int main(int argc, char* argv[]) {

	stringstream ss;
	for (int i=1; i<argc; i++) {
		ss << argv[i] << " ";
	}
	int hidden_layer_size;
    int iterations;
    ss >> hidden_layer_size;
	if (!ss.good() ) hidden_layer_size=4;
    ss >> iterations;
    if (!ss.good() ) iterations=100;

    vector<int> rn({4,2,1});
    rn[1]=hidden_layer_size;
    MBP<float> * mbp = new MBP<float>(rn,time(0));
    mbp->setweightname("test.w");
    Trainer<float> * tr = new Trainer<float>(mbp);
    tr->Verbose(0);
    tr->setCallback([](StatusCallbackData d){if (d.iter%100==1) cout << d.anaCost << endl;});
#define REAL float
    vector<vector<REAL> > a(16,vector<REAL>(4,0.0));
    vector<vector<REAL> > b(16,vector<REAL>(1,0.0));

    tr->setParams(0, 0, 0.0, -1, iterations, 1, 1, 1);

    for (int i=0;i<16;i++) {
        a[i][0]=0.4*(((i/8)%2)*2-1);
        a[i][1]=0.4*(((i/4)%2)*2-1);
        a[i][2]=0.4*(((i/2)%2)*2-1);
        a[i][3]=0.4*((i%2)*2-1);
        b[i][0]=0.4*((((i/8)%2)*2-1)*(((i/4)%2)*2-1)*(((i/2)%2)*2-1)*((i%2)*2-1));
    }
    tr->setInput(a,b);
    tr->Learn();
    if (extratest) {
        vector<REAL> inps(4,0.0); inps[0]=-1;inps[1]=-1;inps[2]=0;inps[3]=-1;
        vector<REAL> oups(1,0.0); oups[0]=0;
        a.push_back(inps); b.push_back(oups);
    }

    tr->setInput(a,b);
    REAL *inp = new REAL[4];
    REAL *outp = new REAL[1];
    if (extratest) cerr << "Learned 16 samples, last sample was not in the training set:" << endl;

    vector<vector<float> > m = mbp->run(a);
    for (unsigned int i=0;i<a.size();i++) {
        for (int j=0;j<4;j++) cerr <<setw(4)<< a[i][j];
        cerr <<"  -->  " <<setw(14)<< m[i][0] << " (" <<setw(4)<<b[i][0]<<" err:" <<setw(12)<< fabs(m[i][0]-b[i][0])<< ")" << endl;
    }

    cout << mbp->codeExport();
    cout << mbp->Export();
    if (extratest) {
        tr->Learn();
        tr->setParams(0, 0, 0.0, -1, iterations, 1, 1, 0);
        cerr << "Learned all 17 samples:" << endl;
        for (unsigned int i=0;i<a.size();i++) {
            for (int j=0;j<4;j++) inp[j]=a[i][j];
            for (int j=0;j<4;j++) cerr <<setw(4)<< inp[j];
            mbp->run(inp,outp);
            cerr <<"  -->  " <<setw(14)<< outp[0] << " (" <<setw(4)<<b[i][0]<< ")" << endl;
        }
    }
    if (exporttest) {
        string es = mbp->Export();
        MBP<float> *nmbp = new MBP<float>(es);
        cout << "---" << endl << nmbp->Export() << endl;
        for (unsigned int i=0;i<a.size();i++) {
            for (int j=0;j<4;j++) inp[j]=a[i][j];
            for (int j=0;j<4;j++) cerr <<setw(4)<< inp[j];
            nmbp->run(inp,outp);
            cerr <<"  -->  " <<setw(14)<< outp[0] << " (" <<setw(4)<<b[i][0]<<" err:" <<setw(12)<< fabs(outp[0]-b[i][0])<< ")" << endl;
        }
        delete nmbp;
    }
    if (exportlearntest) {
        string es = mbp->Export();
        MBP<float> *nmbp = new MBP<float>(es,time(0));
        for (unsigned int i=0;i<a.size();i++) {
            for (int j=0;j<4;j++) inp[j]=a[i][j];
            for (int j=0;j<4;j++) cerr <<setw(4)<< inp[j];
            nmbp->run(inp,outp);
            cerr <<"  -->  " <<setw(14)<< outp[0] << " (" <<setw(4)<<b[i][0]<<" err:" <<setw(12)<< fabs(outp[0]-b[i][0])<< ")" << endl;
        }
        Trainer<float> *ntr = new Trainer<float>(nmbp);
        ntr->setInput(a,b);
        ntr->setParams(0,0,0,0,iterations,1,10,0);
        ntr->Verbose();
        ntr->Learn();
        for (unsigned int i=0;i<a.size();i++) {
            for (int j=0;j<4;j++) inp[j]=a[i][j];
            for (int j=0;j<4;j++) cerr <<setw(4)<< inp[j];
            nmbp->run(inp,outp);
            cerr <<"  -->  " <<setw(14)<< outp[0] << " (" <<setw(4)<<b[i][0]<<" err:" <<setw(12)<< fabs(outp[0]-b[i][0])<< ")" << endl;
        }
        delete ntr;
        delete nmbp;
    }
    delete[] inp;
    delete[] outp;
    delete tr;
    delete mbp;
    mbp=0;
}
