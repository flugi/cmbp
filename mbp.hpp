#ifndef _H_MBP
#define _H_MBP

#include "time.h"
#include <cmath>
#include <fstream>
#include <functional>

#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#define ASSERT(condition, s) if (!(condition)) { std::cerr << __FILE__ << ":" <<  __LINE__ << " >"<< s << "<"<<std::endl; exit(1);}




/* **************************************************************************/
/*                           MM1x1P                                         */
/* **************************************************************************/
/**Matrix multiply with no unrolling, no blocking but use of explicit
 * pointers.
 *
 * @param            c          result matrix
 * @param            a          first matrix
 * @param            b          second matrix
 * @param            Ni         # of rows of c and a
 * @param            Nj         # of cols of c and b
 * @param            Nk         # of cols of a and # of rows of b
 * @param            NaOffs     rows overlap of a
 * @param            NbOffs     rows overlap of b
 *
 * **************************************************************************/
template<typename REAL>
void MM1x1P(REAL* c, REAL* a, REAL* b,
            int Ni,int Nj,int Nk, int NaOffs, int NbOffs) {
    REAL s00;
    REAL *pa,*pb,*pc;
#pragma omp parallel for
    for (int i=0; i<Ni; i++) {
        int k=0;
        for (int j=0; j<Nj; j++) {
            pc = c+j+Nj*i;
            s00 = 0.0;
            for (k=0,pb=b+j+k*NbOffs, pa=a+k+NaOffs*i; k<Nk; k++,pa++,pb+=NbOffs) {
                s00 += (*pa)*(*pb);
            }
            *pc = s00;
        }
    }
}




/**Matrix multiply with no unrolling, no blocking and use of explicit
 * pointers.
 * The second matrix is transposed. \f[ c=a*b^T \f]
 *
 * @param            c          result matrix
 * @param            a          first matrix
 * @param            b          second matrix
 * @param            Ni         # of rows of c and a
 * @param            Nj         # of cols of c and # of rows of b
 * @param            Nk         # of cols of a and b
 * @param            NaOffs     rows overlap of a
 * @param            NbOffs     rows overlap of b
 *
 * **************************************************************************/
template <typename REAL>
void MMT1x1P(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs) {
    REAL s00;
    REAL *pa,*pb,*pc;

#pragma omp parallel for
    for (int i=0; i<Ni; i++) {
        int k=0;
        for (int j=0; j<Nj; j++) {
            pc = c+j+Nj*i;
            s00 = 0.0;
            for (k=0,pb=b+k+NbOffs*j, pa=a+k+NaOffs*i; k<Nk; k++,pa++,pb++) {
                s00 += (*pa)*(*pb);
            }
            *pc = s00;
        }
    }
}


/** Matrix multiply with no unrolling, no blocking and use of explicit
 * pointers.
 * The first matrix is transposed. \f[ c = a^T *b \f]
 *
 * @param            c          result matrix
 * @param            a          first matrix
 * @param            b          second matrix
 * @param            Ni         # of rows of c and # of cols of a
 * @param            Nj         # of cols of c and b
 * @param            Nk         # of rows of a and # of rows of b
 * @param            NaOffs     rows overlap of a
 * @param            NbOffs     rows overlap of b
 * @param
 * **************************************************************************/

template <typename REAL>
void MTM1x1P(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs) {
    REAL *pa,*pb,*pc;
    REAL s00;

#pragma omp parallel for
    for (int i=0; i<Ni; i++) {
        int k=0;
        for (int j=0; j<Nj; j++) {
            pc = c+j+Nj*i;
            s00 = 0.0;
            for (k=0,pb=b+j+k*NbOffs, pa=a+k*NaOffs+i; k<Nk; k++,pa+=NaOffs,pb+=NbOffs) {
                s00 += (*pa)*(*pb);
            }
            *pc = s00;
        }
    }
}



template<typename REAL>
inline REAL nlf (REAL x) {
//    return (REAL)tanh(x);     /* Tanh */
    return x/(1.0+fabs(x));
}

template<typename REAL>
inline REAL nldf (REAL x) {
  //  return (REAL)(1.0-x*x);      /* Tanh' */
    return 1.0/(1.0+fabs(x))/(1.0+fabs(x));
}




template<typename REAL>
class Random {
public:
    REAL RandomNumber( ) {
        double number;
        int64_t aa = 16807L;
        int64_t mm = 2147483647L;
        int64_t qq = 127773L;
        int64_t rr = 2836L;
        int64_t hh = theSeed/qq;
        int64_t lo = theSeed-hh*qq;
        int64_t test = aa*lo-rr*hh;
        if (test > 0) {
            theSeed = test;
        } else {
            theSeed = test+mm;
        }
        number = (REAL)theSeed/(REAL)mm;

        return number;
    }
    Random (long aSeed) {
        theSeed = aSeed;
    }
    int64_t GetSeed () {
        return theSeed;
    }

private:
    long int theSeed;
};






//typedef double REAL;



/// MBP is Matrix Backpropagation neural network by Davide Anguita. C++ wrapper by Gergely Feldhoffer.

/// class MBP is responsible for weight data and calculations as feed forward and error backpropagation

/// @brief Matrix Backpropagation neural network

template <typename REAL>
class MBP {
public:
    /// MBP must be initialised by
    /// @param P_nLayer: the number of hidden and output layers. If P_nLayer==1, the network is a single layer perceptron
    /// @param p_nUnit: the number of neurons in each layer. P_nUnit[0] is the input layer, P_nUnit[P_nLayer] is the output layer
    /// @param P_aSeed: you can specify the random seed.
    MBP (std::vector<int> layerSizes, int P_aSeed) {

        Delta=0;
        weightname="MBPdefault.log";
        nLayer=layerSizes.size()-1;
        nUnit=new int[nLayer+1];
        for (int i=0;i<=nLayer;i++) nUnit[i]=layerSizes[i];


        rand= new Random<REAL>(P_aSeed);

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
    }
    MBP(std::string imports, int P_aSeed = 0) {
        std::stringstream ss;
        ss << imports << " ";
        ss >> nLayer;
        nUnit=new int[nLayer+1];
        for (int i=0;i<=nLayer;i++)
            ss >> nUnit[i];

        rand= new Random<REAL>(P_aSeed);
        Delta=0;

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

        for (int i=1; i <= nLayer; i++) {
            for (int j=0; j < nUnit[i]*nUnit[i-1]; j++) {
                ss >> Weight[i][j];
            }
            for (int j=0; j < nUnit[i]; j++) {
                ss >> Bias[i][j];
            }
        }
        setPattern(0);
    }

    ~MBP()
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
        if (Delta) {
            for (int i=1; i <= nLayer; i++) delete[] Delta[i];
            delete[] Delta;
        }
    }

    ///Allocate space for Delta array. Don't call this method directly, use the member functions of Trainer, unless you know what you are doing
    void setPattern(int nIPattern) {
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

    ///Compute the lerning step.
    /// For each layer:
    /// \f[ [DeltaWeight](n) = [Delta](n)*[Status]^t(n-1) \f]
    /// \f[ DeltaBias(n)     = [Delta](n)*1 \f]
    void Step(REAL **Status, int nIPattern) {
        for (int il=1; il <= nLayer; il++)
        {
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

    ///Feed-forward phase of BP. Computes
    /// \f[ [Status]_n = f([Weight]_n*[Status]_{n-1}+Bias_{n-1}*1^t) \f]
    ///for each layer
    void FeedForward (REAL **NewStatus, int nRowsNS)
    {
        for (int idx=1; idx <= nLayer; idx++) {
            REAL *OldStatus=NewStatus[idx-1];
            int nColsNS=nUnit[idx];
            int nColsOS=nUnit[idx-1];
            int nColsW=nColsNS;

            MM1x1P (NewStatus[idx], OldStatus, Weight[idx], nRowsNS, nColsNS, nColsOS,
                   nColsOS,   nColsW);

            /* Compute neurons' output */

            for (int i=0; i<nRowsNS; i++) {
                for (int j=0; j<nColsNS; j++) {
                    NewStatus[idx][i*nColsNS+j] = nlf(NewStatus[idx][i*nColsNS+j]+Bias[idx][j]);
                }
            }
        }
    }

    ///Runs the network for a single input. outp should be allocated according
    ///the last layer of the network
    void run(REAL *inp, REAL* outp)//REAL **NewStatus, int nRowsNS)
    {
        for (int i=0;i<nUnit[0];i++) {
            RunStatus[0][i]=inp[i];
        }

        for (int idx=1; idx <= nLayer; idx++) {
            REAL *OldStatus=RunStatus[idx-1];
            int nColsNS=nUnit[idx];
            int nColsOS=nUnit[idx-1];
            int nColsW=nColsNS;

            MM1x1P (RunStatus[idx], OldStatus, Weight[idx], 1, nColsNS, nColsOS,
                   nColsOS,   nColsW);

            /* Compute neurons' output */
            for (int j=0; j<nColsNS; j++) {
                RunStatus[idx][j] = nlf(RunStatus[idx][j]+Bias[idx][j]);
            }

        }
        for(int i=0;i<nUnit[nLayer];i++)
            outp[i]=RunStatus[nLayer][i];
    }
    std::vector<std::vector<REAL> > run(const std::vector<std::vector<REAL> > &input) {
        int n = input.size();
        int id = input.at(0).size();
        ASSERT(nUnit[0]==id,"input vector size should be [input dimension][sample size]");
        REAL** buf = new REAL*[Layer()+1];
        for (int i=0; i <= Layer(); i++) {
            (buf)[i]= new REAL[Unit(i)*(n)];
        }
        for (int i=0;i<n;i++) {
            for (int j=0;j<id;j++) {
                buf[0][i*id+j] = input[i][j];
            }
        }
        FeedForward(buf, n);
        int od = Unit(nLayer);
        std::vector<std::vector<REAL> > res(n, std::vector<REAL>(od));
        for (int i=0;i<n;i++) {
            for (int j=0;j<od;j++) {
                res[i][j] = buf[nLayer][i*od+j];
            }
        }
        for (int i=0;i<=Layer();i++) delete[] buf[i];
        delete[] buf;
        return res;
    }

    std::string codeExport()//REAL **NewStatus, int nRowsNS)
    {
        std::stringstream ss;
        for (int i=0;i<nUnit[0];i++) {
            ss << "float rs0" <<i << "=input[" <<i<<"];\n";
        }

        for (int idx=1; idx <= nLayer; idx++) {
            for (int j=0; j<nUnit[idx]; j++) {
                ss << "float rs" << idx << j << "=f(";
                for (int k=0;k<nUnit[idx-1]; k++) {
                    ss << "rs" << idx-1<<k <<"*" << Weight[idx][j+nUnit[idx]*k] <<"+";
//                    ss << "rs" << idx-1<<k <<"*" << "w["<<idx-1<<"]["<<j<<"]["<<k<<"]+";
                }
//                ss << "bias["<<idx-1<<"]["<<j<<"]);\n";
                ss << Bias[idx][j] <<");\n";
            }
        }
        for(int i=0;i<nUnit[nLayer];i++) {
            //outp[i]=RunStatus[nLayer][i];
            ss << "output[" <<i <<"]=rs"<<nLayer<<i<<";\n";
        }
        return ss.str();
    }


    ///updates the weights according to the new learning step.
    void LearningStep(REAL Eta, REAL Alpha) {
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

    ///recalls the weights of the previous learning step.
    void BackStep()
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
    ///Compute the gradient norm.
    REAL ComputeGradientNorm () {
        REAL result = 0.0;

        for (int i=1; i <= nLayer; i++ ) {
            for (int j=0; j < nUnit[i]*nUnit[i-1]; j++) {
                result += (DeltaWeight[i][j])*(DeltaWeight[i][j]);
            }
            for (int j=0; j < nUnit[i]; j++) {
                result += (DeltaBias[i][j])*(DeltaBias[i][j]);
            }
        }

        result = (REAL) sqrt((double)result);

        return result;

    }

    ///Error back-propagation phase.
    /// 1. Compute for the last layer
    ///\f[ [Delta]_N = df([Status]_N) \times ([Target]-[Status]_N) \f]
    /// 2. For the hidden layers:
    ///\f[ [Delta]_n = df([Status]_n) \times ([Weight]^t_{n+1}*[Delta]_{n+1}) \f]
    void ErrorBackProp(REAL** Status, REAL*  Target, int nIPattern)
    {
        for (int i=0; i < nIPattern*nUnit[nLayer]; i++) {
            Delta[nLayer][i] = 2.0/((REAL)(nIPattern*nUnit[nLayer]))*
                               nldf(Status[nLayer][i])*(Target[i]-Status[nLayer][i]);
        }

        for (int i=nLayer-1; i >= 1; i--) {
            MMT1x1P ( Delta[i], Delta[i+1], Weight[i+1],
                    nIPattern, nUnit[i], nUnit[i+1], nUnit[i+1],nUnit[i+1]);
            for (int j=0; j < nIPattern*nUnit[i]; j++) {
                Delta[i][j] *= nldf(Status[i][j]);
            }
        }
    }

    ///Set the filename of the result. Call before SaveWieghts();
    void inline setweightname(std::string a) {weightname=a;}
    ///loads the weights of the network
    int LoadWeights() {
        std::ifstream inp(weightname.c_str());
        if (inp.fail()) {
            std::cerr << "Warning: No weight file";
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

    ///saves the weights of the network
    void SaveWeights() {
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
    std::string Export() {
        std::stringstream ss;
        ss << nLayer << " ";
        for (int i=0;i<=nLayer;i++)
            ss << nUnit[i] << " ";
        for (int i=1; i <= nLayer; i++) {
            for (int j=0; j < nUnit[i]*nUnit[i-1]; j++) {
                ss << Weight[i][j] << " ";
            }
            for (int j=0; j < nUnit[i]; j++) {
                ss << Bias[i][j] << " ";
            }
        }

        return ss.str();
    }

    /// Fill the weight matrix with random values in the interval [-Range,+Range].
    void RandomWeights (REAL Range) {
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
    /// Returns the number of layers
    inline int Layer() {return nLayer;}
    /// Returns the number of neurons in a layer;
    inline int Unit(int idx) {return nUnit[idx];}

protected:

    std::string weightname;

    REAL** Weight;         /**< Weights                                         */
    REAL** Bias;           /**< Biases                                          */
    REAL** OldWeight;      /**< Weights at step n-1                             */
    REAL** OldBias;        /**< Biases at step n-1                              */
    REAL** Delta;          /**< Error back-propagated                           */
    REAL** DeltaWeight;    /**< Weight delta (gradient)                         */
    REAL** DeltaBias;      /**< Bias delta (gradient)                           */
    REAL** OldDeltaWeight; /**< Weight delta at step n-1                        */
    REAL** OldDeltaBias;   /**< Bias delta at step n-1                          */
    REAL** StepWeight;     /**< Weight updating step                            */
    REAL** StepBias;       /**< Bias updating step                              */

    REAL** RunStatus; /**< for run() function*/

    Random<REAL> *rand;

    int    nLayer;         /**< # of layers                                     */
    int*   nUnit;          /**< # of neurons per layer                          */


private:


};

struct StatusCallbackData {
    StatusCallbackData(int i, float a, float m, float d): iter(i), anaCost(a), maxCost(m), digCost(d) {}
    int iter;
    float anaCost, maxCost, digCost;
};


/// Trainer class
/// Trainer is responsible for the training data and traning strategy. YPROP and VOGL mode supported.
/// @brief Neural network training tool

template<typename REAL>
class Trainer {
public:
    ///Trainer must be initialised with an MBP object. This will be the neural network to train
    Trainer(MBP<REAL> *P_mbp) : mbp(P_mbp), AnaTestCost(0.0),MaxTestCost(0.0),DigTestCost(0.0){

        Tune(1.0, 1.47,2.0, 0.75, 0.9, 1.05,0.7, 1.05,0.7,0.07);

        verbosemode=false;
        loadifpossible=false;

        nIPattern=nTPattern=0;
        Test=false;
        Status=StatusTest=0;
        Target=TargetTest=0;
        endMBP      = false;
        nRun        = 0;
        BestAnaCost = 1e38; //big
        BestMaxCost = 1e38;
        BestDigCost = 1e38;
        statuscallback=[](StatusCallbackData){};
    }

    void setCallback(std::function<void(StatusCallbackData)> c) {
        statuscallback = c;
    }

    ~Trainer() {
        if (Status) {
            for (int i=0; i <= mbp->Layer(); i++) delete[] Status[i];
            delete[] Status;
        }
        if (Target) delete[] Target;
        if (StatusTest) {
            for (int i=0; i <= mbp->Layer(); i++) delete[] StatusTest[i];
            delete[] StatusTest;
        }
        if (TargetTest) delete[] TargetTest;

    }

    ///Sets the training data. input std::vector must be sized as [number of samples][input layer size]
    ///output std::vector must be sized as [number of samples][output layer size]
    void setInput(const std::vector<std::vector<REAL> > &input, const std::vector<std::vector<REAL> > &output) {
        unsigned int s=input.size();
        ASSERT(s>0,"input size can not equal 0");
        ASSERT(s==output.size(),"no pairs : input:"<<s<<" output:"<<output.size());
        unsigned int din = input[0].size();
        unsigned int dou = output[0].size();
        for (unsigned int i=0;i<s;i++) {
            ASSERT(input[i].size()==din,"input is not matrix");
            ASSERT(output[i].size()==dou,"input is not matrix");
        }

        Allocate(s);
        for (unsigned int i=0;i<s;i++) {
            for (unsigned int j=0;j<din;j++) {
                Status[0][i*din+j]=input[i][j];
            }
            for (unsigned int j=0;j<dou;j++) {
                Target[i*dou+j]=output[i][j];
            }
        }
        NewSession();
    }


    ///Sets the parameters of the learning
    /// @param P_GradTh Stop criteria: if gradient is less than P_Gradth, the trainig is over
    /// @param P_AnaTh Stop criteria: if analog error is less than P_Anath, the trainig is over
    /// @param P_MaxTh Stop criteria: if maximum error is less than P_Maxth, the trainig is over
    /// @param P_DigTh Stop criteria: if digital error is less than P_Digth, the trainig is over
    /// @param P_IterTh Number of iterations if none of the above stop criterias is true
    /// @param P_RunTh Number of independent runs. Note that in the case of LoadIFPossible(true) the runs will be not independent!
    /// @param P_nPrints If console logging is turned on, in every P_nPrints iterations status will be printed
    /// @param P_YProp YPROP mode. If false Vogl's mode is chosen
    void setParams(REAL P_GradTh, REAL P_AnaTh, REAL P_MaxTh, REAL P_DigTh, long int P_IterTh,
                       int P_RunTh, int P_nPrints, int P_YProp){
        GradTh= P_GradTh;
        AnaTh= P_AnaTh;
        MaxTh= P_MaxTh;
        DigTh= P_DigTh;
        IterTh= P_IterTh;
        RunTh= P_RunTh;
        nPrints= P_nPrints;
        YProp= P_YProp;
    }

    /// sets Learning and accelerating parameters
    void Tune(double P_Sat, double P_ASat, double P_R, double P_E0, double P_A0, double P_Psi,
                    double P_Beta, double P_G, double P_Ka, double P_Kd) {
        Sat = P_Sat;
        ASat= P_ASat;
        R=P_R;
        E0=P_E0;
        A0=P_A0;
        Psi=P_Psi;
        Beta=P_Beta;
        G=P_G;
        Ka=P_Ka;
        Kd=P_Kd;

    }



    /// Initialise new training run. Iteration counter will be reseted, Run counter don't change.
    void NewRun(){
        nIter      = 0;
        OldAnaCost = 1e38; // big
        Eta        = E0;
        endRun     = false;
        endMBP = false;
    }
    /// New Session. Run counter is reseted and new run will be initialised.
    void NewSession() {nRun=0; NewRun();}
    /// Main training method. Use this if unsure.
    void Learn(std::string logfilename="") {
        //std::cerr<<"Learning.. "<<nIPattern<< " patterns"<<std::endl;

        std::ofstream flog;
        if (logfilename!="")flog.open(logfilename.c_str(), std::ios::app);
        while (! endMBP) {
            NewRun();
            /* Print initial information */
            //OutStart (nRun+1);
            /* Compute a new starting point */
            bool loaded=false;
            if (loadifpossible) loaded=!mbp->LoadWeights();
            if (!loaded) mbp->RandomWeights (ASat*R);
            /* Start the timer */
            clock_t Start;     /* Speed measurements                              */
            Start = clock();
            while (! endRun) {
                Step();
                /* logging*/
                if (logfilename!="")flog << nIter << " " <<AnaCost << " " << AnaTestCost << std::endl;
                Train(Start);
            }
        }
    }

    /// if set to true before each training run the network attempt to load the weights. This is useful for
    /// long training sessions, the partial results will be saved regularly, so if anything happens by
    /// restarting the training can be continued
    void LoadIfPossible(bool l=true) {loadifpossible=l;}
    /// Verbose mode. Console logging.
    void Verbose(bool l=true) {verbosemode=l;}



    /// % of wrong outputs of the network
    REAL ComputeDigitalCost  (REAL *Target, int nRowsT, REAL *Status) {
        int nColsT;
        nColsT=mbp->Unit(mbp->Layer());
        int  i;
        REAL result = 0.0;

        /* Compute the % of wrong answers of the net */

        for ( i=0; i < nRowsT*nColsT; i++ ) {
            if (Target[i]*Status[i] <= 0.0) {
                result += 100.0/(REAL)(nRowsT*nColsT);
            }
        }
        lastdigcost=result;
        return result;

    }
    /// quadratic error
    REAL ComputeAnalogCost   (REAL *Target, int nRowsT, REAL *Status) {
        int nColsT;
        nColsT=mbp->Unit(mbp->Layer());

        int  i;
        REAL result = 0.0;

        /* Compute the quadratic error of the net */

        for ( i=0; i < nRowsT*nColsT; i++) {
            result += (Target[i]-Status[i])*(Target[i]-Status[i]);
        }
        result /= (REAL) (nRowsT*nColsT);

        lastanacost=result;
        return result;
    }
    ///maximum absolute error
    REAL ComputeMaximumCost  (REAL *Target, int nRowsT, REAL *Status) {
        int nColsT;
        nColsT=mbp->Unit(mbp->Layer());
        int  i;
        REAL result = 0.0;
        REAL diff;

        /* Compute the maximum abs error of the net */

        for ( i=0; i < nRowsT*nColsT; i++) {
            diff = Target[i] - Status[i];
            if (diff < 0.0) {
                diff = -diff;
            }
            result = (result > diff) ? result : diff;
        }
        lastmaxcost=result;
        return result;
    }

    void setData(int P_nIPattern, REAL** P_Status, REAL* P_Target,
                 int P_nTPattern=0, REAL** P_StatusTest=0, REAL* P_TargetTest=0){
        nIPattern=P_nIPattern;
        Status=P_Status;
        Target=P_Target;
        mbp->setPattern(nIPattern);
        if (P_nTPattern) {
            Test=true;
            nTPattern=P_nTPattern;
            StatusTest=P_StatusTest;
            TargetTest=P_TargetTest;
        } else Test=false;
    }

    double lastAnaCost() const{ return lastanacost;}
    double lastDigCost() const{ return lastdigcost;}
    double lastMaxCost() const{ return lastmaxcost;}

protected:
    double lastanacost, lastdigcost, lastmaxcost;
    ///prints the learning speed.
    void OutTime (double Time) {
        double    nC=0;

        for (int i=1; i<= mbp->Layer(); i++) {
            nC += (double) (mbp->Unit(i-1)+1) * mbp->Unit(i);
        }

        nC *= (double) nIter*nIPattern;

        if (Time > 0.0) {
            printf ("\n\n* MCUPS = %6.4lf\n",nC/(1.0e6*Time));
        } else {
            printf ("\n\n* MCUPS = <not measurable>\n");
        }

    }
    void PrintStep(std::ostream &out = std::cout) {
        out<< "#" << std::setw(4) << nIter
            << " grad:" <<std::setw(13) << GradientNorm
            << " ana:"  <<std::setw(13) << AnaCost
            << " max:"  <<std::setw(13) << MaxCost
            << " dig:"  <<std::setw(5)  << DigCost << "% "
            ;
        if (Test) {
            out << DigTestCost << " " << AnaTestCost << " " << MaxTestCost << " " << std::endl;
        } else {
            out << std::endl;
        }

    }
    void Allocate(int P_nIPattern, bool P_Test=false, int P_nTPattern=0){
        nIPattern=P_nIPattern;
        nTPattern=P_nTPattern;
        Test=P_Test;

        mbp->setPattern(nIPattern);

        if (Status) {
            for (int i=0; i <= mbp->Layer(); i++) delete[] Status[i];
            delete[] Status;
        }
        if (Target) delete[] Target;
        if (StatusTest) {
            for (int i=0; i <= mbp->Layer(); i++) delete[] StatusTest[i];
            delete[] StatusTest;
        }
        if (TargetTest) delete[] TargetTest;

        Status = new REAL*[mbp->Layer()+1];
        for (int i=0; i <= mbp->Layer(); i++) {
            (Status)[i]= new REAL[mbp->Unit(i)*(nIPattern)];
        }

        Target = new REAL[mbp->Unit(mbp->Layer())*(nIPattern)];

        if (Test) {
            TargetTest = new REAL[mbp->Unit(mbp->Layer())*(nTPattern)];
            StatusTest = new REAL*[mbp->Layer()+1];
            for (int i=0; i <= mbp->Layer(); i++) {
                (StatusTest)[i] = new REAL[mbp->Unit(i)*(nTPattern)];
            }
        }


    }
    /// Vogl's or YPROP accelerations
    /// Don't use directly unless you know what you are doing
    void Train(clock_t Start){
        /* Test if a threshold has been reached */
        if (GradientNorm <= GradTh || DigCost  <= DigTh  ||  AnaCost <= AnaTh  ||
            MaxCost  <= MaxTh  || nIter  >= IterTh ) {
            clock_t End = clock();
            endRun = true;

            nRun++;
            if (nRun >= RunTh) {
                endMBP = true;
            }

            if (verbosemode) {
                PrintStep();
                OutTime ((double)(End-Start)/CLOCKS_PER_SEC);
            }
            //mbp->SaveWeights();
        }

        /* else makes a step following the Vogl's or YPROP algorithm */
        else {
            /* If cost is better increase Eta */
            if (AnaCost < OldAnaCost) {
                if (YProp) {
                    Eta *= 1+Ka/(Ka+Eta);
                } else {
                    Eta *= Psi;
                }
                Alpha      = A0;
                OldAnaCost = AnaCost;
            }

            /* If cost is worse than a few percent decrease Eta
           and zeroes momentum */

            else if (AnaCost <= G*OldAnaCost) {
                if (YProp) {
                    Eta *= Kd/(Kd+Eta);
                } else {
                    Eta *= Beta;
                }
                Alpha       = 0.0;
                OldAnaCost = AnaCost;
            }

            /* If cost is really worse, backstep */

            else {
                if (YProp) {
                    Eta *= Kd/(Kd+Eta);
                } else {
                    Eta *= Beta;
                }
                Alpha       = 0.0;
                mbp->BackStep();
            }

            /* Makes the learning step */
            mbp->LearningStep(Eta, Alpha);

        }

        nIter++;


    }
    /// Make a training step. Error will be calculated and backpropagated
    /// Don't use directly unless you know what you are doing
    void Step(){
        /* for each layer */
        /* compute the forward phase for the learning patterns */
        mbp->FeedForward (Status,   nIPattern);

        /* and the test patterns too */
        if (Test) {
            mbp->FeedForward (StatusTest,   nTPattern);
        }

        /* Back-propagate the error of the output layer */
        mbp->ErrorBackProp(Status, Target, nIPattern);

        /* Compute the gradient */
        mbp->Step(Status, nIPattern);

        /* Compute the costs and the gradient norm */

        DigCost  = ComputeDigitalCost (Target, nIPattern, Status[mbp->Layer()]);
        AnaCost  = ComputeAnalogCost  (Target, nIPattern, Status[mbp->Layer()]);
        MaxCost  = ComputeMaximumCost (Target, nIPattern, Status[mbp->Layer()]);
        GradientNorm = mbp->ComputeGradientNorm ();

        /* for the test patterns too */

        if (Test) {
            AnaTestCost = ComputeAnalogCost (TargetTest, nTPattern, StatusTest[mbp->Layer()]);
            MaxTestCost = ComputeMaximumCost (TargetTest, nTPattern, StatusTest[mbp->Layer()]);
            DigTestCost = ComputeDigitalCost (TargetTest, nTPattern, StatusTest[mbp->Layer()]);
        }

        /* Test if the actual cost is the best */

        if (AnaCost < BestAnaCost) {
            BestAnaCost = AnaCost;
            nBestAna    = nRun+1;
        }
        if (MaxCost < BestMaxCost) {
            BestMaxCost = MaxCost;
            nBestMax    = nRun+1;
        }
        if (DigCost < BestDigCost) {
            BestDigCost = DigCost;
            nBestDig    = nRun+1;
        }

        /* Print the starting step */
        /* and every nPrints steps */

        statuscallback(StatusCallbackData(nIter, AnaCost, MaxCost, DigCost));

        if (verbosemode) if ((nIter == 0)||(nPrints != 0 && nIter > 0 && (nIter % nPrints == 0))) {
                PrintStep();
            }
    }

    MBP<REAL> * mbp;
    bool loadifpossible;
    bool verbosemode;
    std::function<void(StatusCallbackData)> statuscallback;

    REAL** Status;         /**< Neurons status for learning patterns            */
    REAL** StatusTest;     /**< Neurons status for test patterns                */
    REAL*  Target;         /**< Target of learning patterns                     */
    REAL*  TargetTest;     /**< Target of test patterns                         */


    bool   endMBP;          /**< End MBP?                                       */
    bool   endRun;          /**< End single run ?                               */
    bool   Test;            /**< Test while learning ? (Cross-validation)       */
    bool   YProp;           /**< YProp or Vogl algorithm ?                      */
    REAL   Sat;             /**< [-1,+1] -> [-Sat,+Sat]                         */
    REAL   ASat;            /**< Tanh(ASat) == Sat                              */
    REAL   R;               /**< Range of initial weights                       */
    REAL   E0;              /**< Initial learning step                          */
    REAL   A0;              /**< Initial momentum                               */
    REAL   Psi;             /**< Vogl's parameter psi                           */
    REAL   Beta;            /**< Vogl's parameter beta                          */
    REAL   G;               /**< Acceptance of error increase (Vogl)            */
    REAL   Ka;              /**< Acceleration factor (YPROP)                    */
    REAL   Kd;              /**< Deceleration factor (YPROP)                    */
    REAL   Eta;             /**< Learning step                                  */
    REAL   Alpha;           /**< Momentum                                       */

    REAL   GradTh;          /**< Threshold on gradient norm                     */
    REAL   AnaTh;           /**< Threshold on analog error   (t-o)^2            */
    REAL   MaxTh;           /**< Threshold on maximum error  max|t-o|           */
    REAL   DigTh;           /**< Threshold on digital error  sign(t*o) %        */
    long   IterTh;          /**< Threshold on # of iterations per run           */
    int    RunTh;           /**< Threshold on # of runs                         */

    int    nPrints;         /**< # of iterations/prints  0=no output            */


    REAL   GradientNorm;    /**< Gradient norm                                  */
    REAL   AnaCost;         /**< Analog cost                    (t-o)^2         */
    REAL   OldAnaCost;      /**< Analog cost at step n-1                        */
    REAL   MaxCost;         /**< Maximum absolute cost          max|t-o|        */
    REAL   DigCost;         /**< Digital cost                   sign(t*o) %     */
    REAL   AnaTestCost;     /**< Analog cost for test patterns                  */
    REAL   MaxTestCost;     /**< Maximum absolute cost for test patterns        */
    REAL   DigTestCost;     /**< Digital cost for test patterns                 */
    long   nIter;           /**< Iteration #                                    */
    int    nRun;            /**< Run #                                          */

    REAL   BestAnaCost;    /**< Best analog cost                                */
    REAL   BestMaxCost;    /**< Best maximum absolute cost                      */
    REAL   BestDigCost;    /**< Best digital cost                               */
    int    nBestAna;       /**< Run # with best analog cost                     */
    int    nBestMax;       /**< Run # with best max abs cost                    */
    int    nBestDig;       /**< Run # with best digital cost                    */

    int    nIPattern;      /**< # of input patterns                             */
    int    nTPattern;      /**< # of test patterns                              */


};




#endif
