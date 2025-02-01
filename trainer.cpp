#ifndef TRAINER_H
#define TRAINER_H

#include "mbp.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
using namespace std;

Trainer::Trainer(MBP *P_mbp) : mbp(P_mbp), AnaTestCost(0.0),MaxTestCost(0.0),DigTestCost(0.0){

    Tune(1.0, 1.47,2.0, 0.75, 0.9, 1.05,0.7, 1.05,0.7,0.07);

    verbosemode=false;
    loadifpossible=false;

    nIPattern=nTPattern=0;
    Test=false;
    Status=StatusTest=0;
    Target=TargetTest=0;
    endMBP      = false;
    nRun        = 0;
    BestAnaCost = MAXREAL;
    BestMaxCost = MAXREAL;
    BestDigCost = MAXREAL;



}

void Trainer::Tune(double P_Sat, double P_ASat, double P_R, double P_E0, double P_A0, double P_Psi,
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

Trainer::~Trainer() {
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

void Trainer::Allocate(int P_nIPattern, bool P_Test, int P_nTPattern) {
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

void Trainer::setParams(REAL P_GradTh, REAL P_AnaTh, REAL P_MaxTh,
                        REAL P_DigTh, long int P_IterTh,
                        int P_RunTh, int P_nPrints, int P_YProp) {
    GradTh= P_GradTh;
    AnaTh= P_AnaTh;
    MaxTh= P_MaxTh;
    DigTh= P_DigTh;
    IterTh= P_IterTh;
    RunTh= P_RunTh;
    nPrints= P_nPrints;
    YProp= P_YProp;
}


void Trainer::Step() {
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

    if (verbosemode) if ((nIter == 0)||(nPrints != 0 && nIter > 0 && (nIter % nPrints == 0))) {
        PrintStep();
    }


}

void Trainer::Train(clock_t Start) {
    /* Test if a threshold has been reached */

    if (GradientNorm <= GradTh || DigCost  <= DigTh  ||  AnaCost <= AnaTh  ||
            MaxCost  <= MaxTh  || nIter  >= IterTh ) {
        /* Stop the timer */
        clock_t End = clock();
        /* and the current run */
        endRun = true;

        nRun++;


        /* End MBP ? */

        if (nRun >= RunTh) {
            endMBP = true;
        }
        if (verbosemode) {
            /* Print the last step */
            PrintStep();

            /* Print the speed of the lerning */
            OutTime ((double)(End-Start)/CLOCKS_PER_SEC);
        }

        /* Save the weights of this run */

        mbp->SaveWeights();
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

void Trainer::NewRun() {
    nIter      = 0;
    OldAnaCost = MAXREAL;
    Eta        = E0;
    endRun     = false;
    endMBP = false;
}

void Trainer::setData(int P_nIPattern, REAL** P_Status, REAL* P_Target,
                 int P_nTPattern, REAL** P_StatusTest, REAL* P_TargetTest) {
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

void Trainer::Learn(string logfilename) {
//cerr<<"Learning.. "<<nIPattern<< " patterns"<<endl;

    ofstream flog;
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
            if (logfilename!="")flog << nIter << " " <<AnaCost << " " << AnaTestCost << endl;
            Train(Start);
        }
    }



}


/* **************************************************************************/
/*                           ComputeDigitalCost()                           */
/* **************************************************************************/

REAL Trainer::ComputeDigitalCost (REAL *Target, int nRowsT,
                                  REAL *Status) {
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

/* **************************************************************************/
/*                           ComputeAnalogCost()                            */
/* **************************************************************************/

REAL Trainer::ComputeAnalogCost  (REAL *Target, int nRowsT,
                                  REAL *Status) {
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

/* **************************************************************************/
/*                           ComputeMaximumCost()                           */
/* **************************************************************************/

REAL Trainer::ComputeMaximumCost (REAL *Target, int nRowsT,
                                  REAL *Status) {
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



/* **************************************************************************/
/*                             OutTime()                                    */
/* **************************************************************************/

void Trainer::OutTime (double Time) {
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

void Trainer::PrintStep(ostream &trf) {
    trf<< "#" << setw(4) << nIter
        << " grad:" <<setw(13) << GradientNorm
        << " ana:"  <<setw(13) << AnaCost
        << " max:"  <<setw(13) << MaxCost
        << " dig:"  <<setw(5)  << DigCost << "% "
        ;
    if (Test) {
        trf << DigTestCost << " " << AnaTestCost << " " << MaxTestCost << " " << endl;
    } else {
        trf << endl;
    }

}

#define ASSERT(felt, s) if (!(felt)) { cerr << __FILE__ << ":" <<  __LINE__ << " >"<< s << "<"<<endl; exit(1);}

void Trainer::setInput(const vector<vector<REAL> > &input, const vector<vector<REAL> > &output) {
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

#endif // TRAINER_H

