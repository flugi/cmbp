#ifndef RANDOM_H
#define RANDOM_H

class Random {
public:
    /// Initialize the seed of the random number generator.
    Random(long aSeed);
    /// Returns a random number in the range [0,1).
    double RandomNumber (void);

    /// Returns the seed of the number generator.
    long GetSeed (void);

private:
    long int theSeed;


};

#endif
