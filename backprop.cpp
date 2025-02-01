#include <iostream>
using namespace std;

#include "mbp.h"
#include "mm.h"
#include <cmath>
#include "random.h"


/* **************************************************************************/
/*                              f()                                         */
/* **************************************************************************/

REAL tanh_f (REAL x) {
    return (REAL)tanh((double)x);     /* Tanh */

}

REAL tanh_apprxf (REAL x) {

    /* IF YOU WANT TO USE THE APPROXIMATION USE THE FOLLOWING CODE INSTEAD */

    REAL Lambda = 0.26037;
    REAL Xs     = 1.92033;
    REAL Ys     = 0.96016;

    if ( x > Xs )
    return Ys;
    else if ( x < -Xs )
    return -Ys;
    else if ( x >= 0.0 )
    return Ys-Lambda*(x-Xs)*(x-Xs);
    else
    return Lambda*(x+Xs)*(x+Xs)-Ys;

}

/* **************************************************************************/
/*                              df()                                        */
/* **************************************************************************/

REAL tanh_df (REAL x) {
    return (REAL)(1.0-x*x);      /* Tanh' */
}

FuncLib::FuncLib() {
    nlfunction tanh;
    tanh.f=tanh_f;
    tanh.df=tanh_df;
    tanh.name="tanh";
    push_back(tanh);

    nlfunction tanhapx;
    tanh.f=tanh_apprxf;
    tanh.df=tanh_df;
    tanh.name="tanh_approx";
    push_back(tanh);

}

nlfunction FuncLib::getNLF(string P_name) {
    nlfunction res = at(0);
    for (unsigned int i=1;i<size();i++)
        if (at(i).name==P_name) res=at(i);
    return res;
}

vector<string> FuncLib::names() const {
    vector<string> res(size(),"");
    for (unsigned int i=0;i<size();i++)
        res[i]=at(i).name;
    return res;
}

FuncLib funclib;



MBP::MBP (vector<int> layerSizes, int P_aSeed) {

    Delta=0;
    weightname="MBPdefault.log";
    nLayer=layerSizes.size()-1;
    nUnit=new int[nLayer+1];
    for (int i=0;i<=nLayer;i++) nUnit[i]=layerSizes[i];

    nl=funclib.getNLF("tanh");

    rand= new Random(P_aSeed);

    Weight         = new REAL*[nLayer+1];
    Bias           = new REAL*[nLayer+1];
    OldWeight      = new REAL*[nLayer+1];
    OldBias        = new REAL*[nLayer+1];
    DeltaWeight    = new REAL*[nLayer+1];
    DeltaBias      = new REAL*[nLayer+1];
    OldDeltaWeight = new REAL*[nLayer+1];
    OldDeltaBias   = new REAL*[nLayer+1];
    StepWeight     = new REAL*[nLayer+1];
    StepBias       = new REAL*[nLayer+1];

    RunStatus = new REAL*[nLayer+1];

    for (int i=1; i <= nLayer; i++) {
        Weight[i]         = new REAL[nUnit[i]*nUnit[i-1]];
        OldWeight[i]      = new REAL[nUnit[i]*nUnit[i-1]];
        DeltaWeight[i]    = new REAL[nUnit[i]*nUnit[i-1]];
        OldDeltaWeight[i] = new REAL[nUnit[i]*nUnit[i-1]];
        StepWeight[i]     = new REAL[nUnit[i]*nUnit[i-1]];
        for (int j=0;j<nUnit[i]*nUnit[i-1];j++) {
            Weight[i][j]=OldWeight[i][j]=DeltaWeight[i][j]=OldDeltaWeight[i][j]=StepWeight[i][j]=0;
        }
    }
    for (int i=0; i <= nLayer; i++) {
        Bias[i]           = new REAL[nUnit[i]];
        OldBias[i]        = new REAL[nUnit[i]];
        DeltaBias[i]      = new REAL[nUnit[i]];
        OldDeltaBias[i]   = new REAL[nUnit[i]];
        StepBias[i]       = new REAL[nUnit[i]];
        RunStatus[i] = new REAL[nUnit[i]];
        for (int j=0;j<nUnit[i];j++) {
            RunStatus[i][j]=Bias[i][j]=OldBias[i][j]=DeltaBias[i][j]=OldDeltaBias[i][j]=StepBias[i][j]=0;
        }
    }
//cerr << nUnit[0] << "cc ";
}


MBP::~MBP()
{
    delete[] nUnit;
    delete rand;

    for (int i=1; i <= nLayer; i++) {
        delete[] Weight[i];
        delete[] OldWeight[i];
        delete[] DeltaWeight[i];
        delete[] OldDeltaWeight[i];
        delete[] StepWeight[i];
    }
    for (int i=0; i <= nLayer; i++) {
        delete[] Bias[i];
        delete[] OldBias[i];
        delete[] DeltaBias[i];
        delete[] OldDeltaBias[i];
        delete[] StepBias[i];
        delete[] RunStatus[i];
    }

    delete[] Weight;
    delete[] Bias;
    delete[] OldWeight;
    delete[] OldBias;
    delete[] DeltaWeight;
    delete[] DeltaBias;
    delete[] OldDeltaWeight ;
    delete[] OldDeltaBias;
    delete[] StepWeight;
    delete[] StepBias;

    delete[] RunStatus;
    for (int i=1; i <= nLayer; i++) delete[] Delta[i];
    delete[] Delta;
}

void MBP::setPattern(int nIPattern) {
    if (Delta) {
        for (int i=1; i <= nLayer; i++) delete[] Delta[i];
        delete[] Delta;
    }

    Delta = new REAL*[nLayer+1];
    for (int i=1; i <= nLayer; i++) {
        Delta[i] = new REAL[nUnit[i]*nIPattern];
        for (int j=0;j<nUnit[i]*nIPattern;j++)
            Delta[i][j]=0;
    }
}


/* **************************************************************************/
/*                           FeedForward()                                  */
/* **************************************************************************/

void MBP::FeedForward (REAL **NewStatus, int nRowsNS)
{
    for (int idx=1; idx <= nLayer; idx++) {
        REAL *OldStatus=NewStatus[idx-1];
        int nColsNS=nUnit[idx];
        int nColsOS=nUnit[idx-1];
        int nColsW=nColsNS;
        /* Fast matrix multiplication routine to compute neurons' net input */

//        MM2x2PB (NewStatus[idx], OldStatus, Weight[idx], nRowsNS, nColsNS, nColsOS,
//                 nColsOS,   nColsW, BLOCK_DIMENSION);

        MM1x1P (NewStatus[idx], OldStatus, Weight[idx], nRowsNS, nColsNS, nColsOS,
                 nColsOS,   nColsW);

        /* Compute neurons' output */

        for (int i=0; i<nRowsNS; i++) {
            for (int j=0; j<nColsNS; j++) {
                NewStatus[idx][i*nColsNS+j] = nl.f(NewStatus[idx][i*nColsNS+j]+Bias[idx][j]);
            }
        }
    }
}

void MBP::run (REAL *inp, REAL* outp)//REAL **NewStatus, int nRowsNS)
{
//cerr << nUnit[0]<<"..." << flush;
    //TODO: first and last round can be inserted into the loop
    for (int i=0;i<nUnit[0];i++) { //cerr << i << " ";
        RunStatus[0][i]=inp[i];}
//cerr << "inploaded" << flush;
    for (int idx=1; idx <= nLayer; idx++) {
//cerr << "layer" << idx << "  " << flush;
        REAL *OldStatus=RunStatus[idx-1];
        int nColsNS=nUnit[idx];
        int nColsOS=nUnit[idx-1];
        int nColsW=nColsNS;
        /* Fast matrix multiplication routine to compute neurons' net input */

//        MM2x2PB (RunStatus[idx], OldStatus, Weight[idx], 1, nColsNS, nColsOS,
//                 nColsOS,   nColsW, BLOCK_DIMENSION);
        MM1x1P (RunStatus[idx], OldStatus, Weight[idx], 1, nColsNS, nColsOS,
                 nColsOS,   nColsW);

        /* Compute neurons' output */
        for (int j=0; j<nColsNS; j++) {
            RunStatus[idx][j] = nl.f(RunStatus[idx][j]+Bias[idx][j]);
        }

    }
    for(int i=0;i<nUnit[nLayer];i++) outp[i]=RunStatus[nLayer][i];
//cerr << nUnit[0]<<"..." << flush;
}




void MBP::Step(REAL **Status, int nIPattern) {
    for (int il=1; il <= nLayer; il++) {
    /* Compute the gradient */

//        MTM2x2PB(DeltaWeight[il], Status[il-1], Delta[il], nUnit[il-1],nUnit[il], nIPattern, nUnit[il-1],
//                 nUnit[il], BLOCK_DIMENSION);
        MTM1x1P(DeltaWeight[il], Status[il-1], Delta[il], nUnit[il-1],nUnit[il], nIPattern, nUnit[il-1],
                 nUnit[il]);

        for (int i=0; i<nUnit[il]; i++) {
            DeltaBias[il][i] = 0.0;
            for (int j=0; j<nIPattern; j++) {
                DeltaBias[il][i] += Delta[il][j*nUnit[il]+i];
            }
        }

    }
}




/* **************************************************************************/
/*                           ComputeGradientNorm()                          */
/* **************************************************************************/

REAL MBP::ComputeGradientNorm () {
    int  i,j;
    REAL result = 0.0;
    REAL *pDeltaWeight;   /* Some pointers to speed things up */
    REAL *pDeltaBias;

    for ( i=1; i <= nLayer; i++ ) {
        pDeltaWeight = DeltaWeight[i];
        pDeltaBias   = DeltaBias[i];

        for ( j=0; j < nUnit[i]*nUnit[i-1]; j++) {
            result += (pDeltaWeight[j])*(pDeltaWeight[j]);
        }
        for ( j=0; j < nUnit[i]; j++) {
            result += (pDeltaBias[j])*(pDeltaBias[j]);
        }
    }

    result = (REAL) sqrt((double)result);

    return result;

}


void MBP::LearningStep(REAL Eta, REAL Alpha) {
    for (int il=1; il <= nLayer; il++) {
        /* A step in the right direction (hopefully) */
        for (int i=0; i < nUnit[il-1]*nUnit[il]; i++ ) {
            OldDeltaWeight[il][i] = DeltaWeight[il][i];
            StepWeight[il][i]     = Eta*DeltaWeight[il][i] + Alpha*StepWeight[il][i];
            Weight[il][i]        += StepWeight[il][i];
        }
        for (int i=0; i < nUnit[il]; i++ ) {
            OldDeltaBias[il][i] = DeltaBias[il][i];
            StepBias[il][i]     = Eta*DeltaBias[il][i] + Alpha*StepBias[il][i];
            Bias[il][i]        += StepBias[il][i];
        }
    }
}


void MBP::BackStep()
{
    for (int i=1; i <= nLayer; i++) {
        /* Makes a learning step back */
        for (int j=0; j < nUnit[i-1]*nUnit[i]; j++) {
            DeltaWeight[i][j] =  OldDeltaWeight[i][j];
            Weight[i][j]      -= StepWeight[i][j];
        }
        for (int j=0; j < nUnit[i]; j++) {
            DeltaBias[i][j] =  OldDeltaBias[i][j];
            Bias[i][j]      -= StepBias[i][j];
        }
    }
}

/*#include <fstream>
using namespace std;
ofstream bplog("backproplog.txt");
*/

///Error back-propagation phase.
/// 1. Compute for the last layer
///\f[ [Delta]_N = df([Status]_N) \times ([Target]-[Status]_N) \f]
/// 2. For the hidden layers:
///\f[ [Delta]_n = df([Status]_n) \times ([Weight]^t_{n+1}*[Delta]_{n+1}) \f]

void MBP::ErrorBackProp(REAL** Status, REAL*  Target, int nIPattern)
{
    for (int i=0; i < nIPattern*nUnit[nLayer]; i++) {
        Delta[nLayer][i] = 2.0/((REAL)(nIPattern*nUnit[nLayer]))*
                nl.df(Status[nLayer][i])*(Target[i]-Status[nLayer][i]);
    }
/*
for (int i=0; i < nIPattern*nUnit[nLayer]; i++) bplog << Status[nLayer][i] << " ";
for (int i=0; i < nIPattern*nUnit[nLayer]; i++) bplog << Target[i] << " ";
bplog<<endl;
*/
    /* and the hidden layers */

    for (int i=nLayer-1; i >= 1; i--) {

        //Compute deltas of hidden neurons
                                                                     //ez talan problemas
//        MMT2x2P ( Delta[i], Delta[i+1], Weight[i+1], nIPattern, nUnit[i], nUnit[i+1], nUnit[i+1], nUnit[i+1]);
        MMT1x1P ( Delta[i], Delta[i+1], Weight[i+1], nIPattern, nUnit[i], nUnit[i+1], nUnit[i+1],nUnit[i+1]);
        for (int j=0; j < nIPattern*nUnit[i]; j++) {
            Delta[i][j] *= nl.df(Status[i][j]);
        }
    }
}


/* **************************************************************************/
/*                         RandomWeights()                                  */
/* **************************************************************************/

void MBP::RandomWeights (REAL Range) {

    for (int il=1; il <= nLayer; il++ ) {
        REAL R=Range/(nUnit[il-1]+1.0);
        for (int i=0; i < nUnit[il]; i++) {
            for (int j=0; j < nUnit[il-1]; j++) {
                Weight[il][i+j*nUnit[il]] = R*(REAL)(1.0-2.0*rand->RandomNumber());
            }
        }
        for (int i=0; i < 1; i++) {
            for (int j=0; j < nUnit[il]; j++) {
                Bias[il][i+j*1] = R*(REAL)(1.0-2.0*rand->RandomNumber());
            }
        }

    }
}


/* **************************************************************************/
/*                             SaveWeights()                                */
/* **************************************************************************/

void MBP::SaveWeights() {
    FILE* fWeight = fopen(weightname.c_str(),"wt");

    int i,j;

    for (i=1; i <= nLayer; i++) {
        for (j=0; j < nUnit[i]*nUnit[i-1]; j++) {
            fprintf(fWeight,"%lf ",(double)Weight[i][j]);
        }
        for (j=0; j < nUnit[i]; j++) {
            fprintf(fWeight,"%lf ",(double)Bias[i][j]);
        }
    }
            fclose(fWeight);

}

#include <fstream>
#include <iostream>
//using std::ofstream;
using std::ifstream;
using std::cerr;

int MBP::LoadWeights() {
    ifstream inp(weightname.c_str());
    if (inp.fail()) {
        cerr << "Warning: No weight file";
        return -1;
    }

    for (int i=1; i <= nLayer; i++) {
        for (int j=0; j < nUnit[i]*nUnit[i-1]; j++) {
            inp >> Weight[i][j];
        }
        for (int j=0; j < nUnit[i]; j++) {
            inp >> Bias[i][j];
        }
    }
    inp.close();
    return 0;
}


