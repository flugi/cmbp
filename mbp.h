#ifndef _H_MBP
#define _H_MBP

#include "time.h"

#include <string>
using std::string;

#include <vector>
using std::vector;

typedef double REAL;
#define MAXREAL 1.0e38

typedef double(*NLF)(double);

struct nlfunction {
    NLF f, df;
    string name;
};

class FuncLib : public vector<nlfunction> {
public:
    FuncLib();
    nlfunction getNLF(string P_name);
    vector<string> names() const;
};

extern FuncLib funclib;

class Random;


/// MBP is Matrix Backpropagation neural network by Davide Anguita. C++ wrapper by Gergely Feldhoffer.

/// class MBP is responsible for weight data and calculations as feed forward and error backpropagation

/// @brief Matrix Backpropagation neural network

class MBP {
public:
    /// MBP must be initialised by
    /// @param P_nLayer: the number of hidden and output layers. If P_nLayer==1, the network is a single layer perceptron
    /// @param p_nUnit: the number of neurons in each layer. P_nUnit[0] is the input layer, P_nUnit[P_nLayer] is the output layer
    /// @param P_aSeed: you can specify the random seed.
    MBP (int P_nLayer, int* P_nUnit, int P_aSeed=0);

    ///Allocate space for Delta array. Don't call this method directly, use the member functions of Trainer, unless you know what you are doing
    void setPattern(int P_nIPattern);

    ///Compute the lerning step.
    /// For each layer:
    /// \f[ [DeltaWeight](n) = [Delta](n)*[Status]^t(n-1) \f]
    /// \f[ DeltaBias(n)     = [Delta](n)*1 \f]
    void Step(REAL **Status, int nIPattern);

    ///Feed-forward phase of BP. Computes
    /// \f[ [Status]_n = f([Weight]_n*[Status]_{n-1}+Bias_{n-1}*1^t) \f]
    ///for each layer
    void FeedForward (REAL **NewStatus, int nRowsNS);

    ///Runs the network for a single input. outp should be allocated according
    ///the last layer of the network
    void run(REAL *inp, REAL *outp);


    ///updates the weights according to the new learning step.
    void LearningStep(REAL Eta, REAL Alpha);

    ///recalls the weights of the previous learning step.
    void BackStep();
    ///Compute the gradient norm.
    REAL ComputeGradientNorm ();

    ///Error back-propagation phase.
    /// 1. Compute for the last layer
    ///\f[ [Delta]_N = df([Status]_N) \times ([Target]-[Status]_N) \f]
    /// 2. For the hidden layers:
    ///\f[ [Delta]_n = df([Status]_n) \times ([Weight]^t_{n+1}*[Delta]_{n+1}) \f]
    void ErrorBackProp(REAL** Status, REAL*  Target, int nIPattern);

    ///Set the filename of the result. Call before SaveWieghts();
    void inline setweightname(string a) {weightname=a;}
    ///loads the weights of the network
    int LoadWeights();
    ///saves the weights of the network
    void SaveWeights();
    /// Fill the weight matrix with random values in the interval [-Range,+Range].
    void RandomWeights (REAL Range);
    /// Returns the number of layers
    inline int Layer() {return nLayer;}
    /// Returns the number of neurons in a layer;
    inline int Unit(int idx) {return nUnit[idx];}

protected:

    nlfunction nl;
    string weightname;

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

    Random *rand;

    int    nLayer;         /**< # of layers                                     */
    int*   nUnit;          /**< # of neurons per layer                          */


private:


};


/// Trainer class
/// Trainer is responsible for the training data and traning strategy. YPROP and VOGL mode supported.
/// @brief Neural network training tool

class Trainer {
public:
    ///Trainer must be initialised with an MBP object. This will be the neural network to train
    Trainer(MBP *P_mbp);
    ///Sets the training data. input vector must be sized as [number of samples][input layer size]
    ///output vector must be sized as [number of samples][output layer size]
    void setInput(const vector<vector<double> > &input, const vector<vector<double> > &output);

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
                       int P_RunTh, int P_nPrints, int P_YProp);

    /// sets Learning and accelerating parameters
    void Tune(double P_Sat, double P_ASat, double P_R, double P_E0, double P_A0, double P_Psi,
                    double P_Beta, double P_G, double P_Ka, double P_Kd);

    /// Make a training step. Error will be calculated and backpropagated
    /// Don't use directly unless you know what you are doing
    void Step();

    /// Vogl's or YPROP accelerations
    /// Don't use directly unless you know what you are doing
    void Train(clock_t Start);
    /// Initialise new training run. Iteration counter will be reseted, Run counter don't change.
    void NewRun();
    /// New Session. Run counter is reseted and new run will be initialised.
    void NewSession() {nRun=0; NewRun();}
    /// Main training method. Use this if unsure.
    void Learn(string logfilename="");
    /// if set to true before each training run the network attempt to load the weights. This is useful for
    /// long training sessions, the partial results will be saved regularly, so if anything happens by
    /// restarting the training can be continued
    void LoadIfPossible(bool l=true) {loadifpossible=l;}
    /// Verbose mode. Console logging.
    void Verbose(bool l=true) {verbosemode=l;}



    /// % of wrong outputs of the network
    REAL ComputeDigitalCost  (REAL *Target, int nRowsT, REAL *Status);
    /// quadratic error
    REAL ComputeAnalogCost   (REAL *Target, int nRowsT, REAL *Status);
    ///maximum absolute error
    REAL ComputeMaximumCost  (REAL *Target, int nRowsT, REAL *Status);

    void setData(int P_nIPattern, REAL** P_Status, REAL* P_Target,
                 int P_nTPattern=0, REAL** P_StatusTest=0, REAL* P_TargetTest=0);

    double lastAnaCost() const{ return lastanacost;}
    double lastDigCost() const{ return lastdigcost;}
    double lastMaxCost() const{ return lastmaxcost;}

protected:
    double lastanacost, lastdigcost, lastmaxcost;
    ///prints the learning speed.
    void OutTime (double Time);
    void PrintStep();
    void Allocate(int P_nIPattern, bool P_Test=false, int P_nTPattern=0);

    MBP * mbp;
    bool loadifpossible;
    bool verbosemode;



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
