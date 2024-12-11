#ifndef _H_MM
#define _H_MM

#include "mbp.h"

#define BLOCK_DIMENSION 50

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

void MM1x1P(REAL* c, REAL* a, REAL* b,
            int Ni,int Nj,int Nk, int NaOffs, int NbOffs);




/**Matrix multiply with 2x2 unrolling, no blocking and use of explicit
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

void MM2x2P(REAL* c, REAL* a, REAL* b,
            int Ni, int Nj, int Nk, int NaOffs, int NbOffs);


/**Matrix multiply with 2x2 unrolling, blocking and use of explicit
 * pointers.
 *
 * @param            c          result matrix
 * @param            a          first matrix
 * @param            b          second matrix
 * @param            Ni         # of rows of c and a
 * @param            Nj         # of cols of c and b
 * @param            Nk         # of cols of a and # of rows of b
 * @param           NaOffs     rows overlap of a
 * @param            NbOffs     rows overlap of b
 * @param            NB         Block dimension (must be even)
 * **************************************************************************/
void MM2x2PB(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs, int NB);




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
void MMT1x1P(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs);




/**Matrix multiply with 2x2 unrolling, no blocking and use of explicit
 * pointers.
 * The second matrix is transposed. \f[ c = a*b^T \f]
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
void MMT2x2P(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs);




/** Matrix multiply with 2x2 unrolling, blocking and use of explicit
 * pointers.
 * The second matrix is transposed. \f[ c = a*b^T \f]
 *
 * @param            c          result matrix
 * @param            a          first matrix
 * @param            b          second matrix
 * @param            Ni         # of rows of c and a
 * @param            Nj         # of cols of c and # of rows of b
 * @param            Nk         # of cols of a and b
 * @param            NaOffs     rows overlap of a
 * @param            NbOffs     rows overlap of b
 * @param            NB         Block dimension (must be even)
 *
 * **************************************************************************/
void MMT2x2PB(REAL* c, REAL* a, REAL* b,
              int Ni, int Nj, int Nk, int NaOffs, int NbOffs, int NB);



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
void MTM1x1P(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs);



/**Matrix multiply with 2x2 unrolling, no blocking and use of explicit
 * pointers.
 * The first matrix is transposed. \f[ c = a^T *b \f]
 *
 *
 * @param            c          result matrix
 * @param            a          first matrix
 * @param            b          second matrix
 * @param            Ni         # of rows of c and # of cols of a
 * @param            Nj         # of cols of c and b
 * @param            Nk         # of rows of a and # of rows of b
 * @param            NaOffs     rows overlap of a
 * @param            NbOffs     rows overlap of b

*/
void MTM2x2P(REAL* c, REAL* a, REAL* b,
             int Ni, int Nj, int Nk, int NaOffs, int NbOffs);



/**
Matrix multiply with 2x2 unrolling, blocking and use of explicit pointers.
The first matrix is transposed. \f[ c =  a^T * b \f]

 @param c result matrix
 @param a first matrix
 @param b second matrix
 @param Ni # of rows of c and # of cols of a
 @param Nj # of cols of c and b
 @param Nk # of rows of a and # of rows of b
 @param NaOffs rows overlap of a
 @param NbOffs rows overlap of b
 @param NB Block dimension (must be even)
*/
void MTM2x2PB(REAL* c, REAL* a, REAL* b,
              int Ni, int Nj, int Nk, int NaOffs, int NbOffs, int NB);

#endif

/* **************************************************************************/
/*                                 END OF FILE                              */
/* **************************************************************************/
