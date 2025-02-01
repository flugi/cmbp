#include "mbp.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
using namespace std;

#define REAL float

struct Mapping {
    float preadditive, postadditive;
    float multiplicative;
} mapping;

vector<REAL> convert_line(string s) {
    vector<REAL> v;
    stringstream ss;
    ss << s << " ";
    float a;
    ss >> a;
    while (ss.good()) {
//        if (a<0 || a>1) {
//            cerr << "." << flush;
//        }
        v.push_back(a);
        ss >> a;
    }
    return v;
}

vector<vector<REAL> > readfile(string fname) {
    vector<vector<REAL> >res;
    ifstream f(fname);
    string s;
    getline(f,s);
    while (f.good()) {
        res.push_back(convert_line(s));
        getline(f,s);
    }
    return res;
}

void normalize(vector<vector<REAL> >&input, vector<vector<REAL> > &output ) {
    float max=input[0][0], min=input[0][0];

    for (vector<REAL> v : input) {
        for (REAL a : v) {
            if (min > a) min=a;
            if (max < a) max=a;
        }
    }
    for (vector<REAL> v : output) {
        for (REAL a : v) {
            if (min > a) min=a;
            if (max < a) max=a;
        }
    }
    float dist = max-min;
    cerr << min << " .. " << max << " ->" << dist << endl;
    mapping.multiplicative = 1.8/dist;
    mapping.preadditive=-min;
    mapping.postadditive=-0.9;
    for (vector<REAL> &v : input) {
        for (REAL &a : v) {
            a=(a+mapping.preadditive)*mapping.multiplicative+mapping.postadditive;
        }
    }
    for (vector<REAL> &v : output) {
        for (REAL &a : v) {
            a=(a+mapping.preadditive)*mapping.multiplicative+mapping.postadditive;
        }
    }

}

void check1(vector<vector<REAL> >&input, vector<vector<REAL> > &output ) {
    int N = input.size();
    for (int i=0;i<N;i++) {
        vector<REAL> pivot= input[i];
        vector<int> indices;
        for (int j=0;j<N;j++) {
            float sum=0;
            for (int k=0;k<input[j].size();k++) {
                sum+=fabs(input[j][k]-pivot[k]);
            }
            if (sum<0.004) { // 1/255 == 0.004
                indices.push_back(j);
            }
        }
        if (indices.size()>1) {
//            cout << indices.size() << " ";
            for (int d : indices) {
                float sum=0;
                for (int k =0; k<input[d].size();k++) {
                    sum+=fabs(output[i][k]-output[d][k]);
                }
                if (sum>0.2) {
                    for (float a : pivot) cout << a << " ";
                    cout << " -> ";
                    for (float a : output[i]) cout << a << " ";
                    cout << endl;
                    for (float a : input[d]) cout << a << " ";
                    cout << " -> ";
                    for (float a : output[d]) cout << a << " ";
                    cout << endl;
                    cout << sum << endl;
                }
            }
        }
    }
}

const int CL=255;

void check(vector<vector<REAL> >&input, vector<vector<REAL> > &output ) {
    map<vector<int>, vector<REAL>> m;
    int N = input.size();
    for (int i=0;i<N;i++) {
        vector<int> inv;
        for (REAL a : input[i]) {
            inv.push_back(int(a*CL));
        }
        auto it = m.find(inv);
        if (it==m.end()) {
            m[inv]=output[i];
        } else {
            if ((output[i][0]) < (it->second[0])) {
                m[inv]=output[i];
            }
        }
    }
    cout << m.size() << " vs " << input.size() << endl;
    vector<vector<REAL> >new_input, new_output;
    for (pair<vector<int>, vector<REAL>> p : m) {
        vector<REAL> v;
        for (int a : p.first) {
            v.push_back(REAL(a/float(CL)));
        }
        new_input.push_back(v);
        new_output.push_back(p.second);
    }
    input=new_input;
    output=new_output;
}

void decimate(vector<vector<REAL> >&input, vector<vector<REAL> > &output ) {
    const int step = 32*15;
    vector<vector<REAL>> new_input, new_output;
    for (int i=0;i<input.size();i+=step) {
        new_input.push_back(input[i]);
        new_output.push_back(output[i]);
    }
    input=new_input;
    output=new_output;
}


int main(int argc, char* argv[]) {

	stringstream ss;
	for (int i=1; i<argc; i++) {
		ss << argv[i] << " ";
	}
	string inputname, outputname;
	int hidden_layer_size;
    int iterations;
    vector<int> customLayers;
    ss >>inputname>>outputname;
    if (!ss.good() ) {
        cerr << "usage: " << argv[0] << " inputfile outputfile [hidden_layer_size] [iterations]" << endl;
        exit(1);
    }
    ss >> hidden_layer_size;
	if (!ss.good() ) hidden_layer_size=4;
    ss >> iterations;
    if (!ss.good() ) iterations=100;
    if (hidden_layer_size == 0) {
        char c;
        ss>>c;
        string s;
        if (c=='[') {
            getline(ss,s,']');
            stringstream ss2;
            ss2<<s << " ";
            int a;
            ss2 >> a;
            while (ss2.good()) {
                customLayers.push_back(a);
                ss2 >> a;
            }
        }
    }

    vector<vector<REAL> > input = readfile(inputname);
    cout << input.size() <<flush<< "x" << input.at(0).size() << endl;

    vector<vector<REAL> > output = readfile(outputname);
    cout << output.size() <<flush<< "x" << output.at(0).size() << endl;

    normalize(input, output);


    int ID = input.at(0).size();
    int OD = output.at(0).size();

    vector<int> rn(3,0);
    rn[0]=ID;
    rn[1]=hidden_layer_size;
    rn[2]=OD;
    if (customLayers.size()) {
        rn=vector<int> (customLayers.size()+2);
        rn[0]=ID;
        rn[rn.size()-1]=OD;
        for (size_t i=0;i<customLayers.size();i++) {
            rn[i+1]=customLayers[i];
        }
    }
    MBP<float> * mbp = new MBP<float>(rn,2);
    mbp->setweightname("test.w");
    Trainer<float> * tr = new Trainer<float>(mbp);
    tr->LoadIfPossible(true);
    tr->Verbose(1);
    tr->setParams(0, 0, 0.0, -1, iterations, 1, 10, 1);
    tr->setInput(input,output);
    tr->Learn();
    mbp->SaveWeights();
    ofstream outfile(outputname+".estimated");
    for (unsigned int i=0;i<input.size();i++) {
        vector<REAL> output_line(OD);
        REAL * inp = &(input[i][0]);
        mbp->run(inp,&(output_line[0]));
        for (int j=0;j<OD;j++) outfile << (output_line[j]-mapping.postadditive)/mapping.multiplicative-mapping.preadditive << " ";
        outfile << "\n";
    }

}
