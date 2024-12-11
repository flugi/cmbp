#include "random.h"

/// This is the seed of the random number generator.

double Random::RandomNumber( void ) {
    double number;
    long   aa = 16807L;
    long   mm = 2147483647L;
    long   qq = 127773L;
    long   rr = 2836L;
    long   lo;
    long   hh;
    long   test;

    hh = theSeed/qq;
    lo = theSeed-hh*qq;
    test = aa*lo-rr*hh;
    if (test > 0) {
        theSeed = test;
    } else {
        theSeed = test+mm;
    }
    number = (double)theSeed/(double)mm;

    return number;
}

Random::Random (long aSeed) {
    theSeed = aSeed;
}


long Random::GetSeed (void) {
    return theSeed;
}
