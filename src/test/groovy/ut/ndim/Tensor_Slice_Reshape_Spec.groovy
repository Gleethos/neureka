package ut.ndim

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.view.NDPrintSettings
import spock.lang.Specification

class Tensor_Slice_Reshape_Spec extends Specification
{
    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'A slice of a tensor changes as expected when reshaping it.'()
    {
        given : 'A parent tensor.'
            Tsr A = Tsr.of([
                    [  1,  5,  3, -6, -3,  8, -9,  4  ],
                    [  0, -2,  2,  1, -1,  0,  5,  4  ],
                    [ -6,  7,  7, -2,  9,  0,  1, -1  ],
                    [  4,  4, -1,  8,  4, -3,  2, -9  ],
                    [  7,  5, -2, -3,  7, -8,  5,  0  ]
            ])

        when : 'A slice of this tensor is being created...'
            Tsr a = A[1..3, 4..5]
        and : 'And also a slice of "a" with the same dimensionality, namely : "b".'
            Tsr b = a[]

        then : 'The slice and the parent are as expected.'
            A.toString() == "(5x8):[1.0, 5.0, 3.0, -6.0, -3.0, 8.0, -9.0, 4.0, 0.0, -2.0, 2.0, 1.0, -1.0, 0.0, 5.0, 4.0, -6.0, 7.0, 7.0, -2.0, 9.0, 0.0, 1.0, -1.0, 4.0, 4.0, -1.0, 8.0, 4.0, -3.0, 2.0, -9.0, 7.0, 5.0, -2.0, -3.0, 7.0, -8.0, 5.0, 0.0]"
            a.toString() == "(3x2):[-1.0, 0.0, 9.0, 0.0, 4.0, -3.0]"
        and : 'The slice "b" of the slice "a" is a different tensor that references the same underlying data!'
            a != b
            a.toString() == b.toString()

        when : 'The slice "a" is being reshaped... (transposed)'
            Tsr c = a.T()

        then : 'The returned tensor is a different one than "a".'
            a != c

        and : 'The order of the value of "c" reflect the reshaping...'
            c.toString() == "(2x3):[-1.0, 9.0, 4.0, 0.0, 0.0, -3.0]"

        and : 'The tensor "b" and the tensor "a" have not changed.'
            b.toString() == "(3x2):[-1.0, 0.0, 9.0, 0.0, 4.0, -3.0]"
            a.toString() == "(3x2):[-1.0, 0.0, 9.0, 0.0, 4.0, -3.0]"
    }



    def 'Two slices of one big tensor perform matrix multiplication flawless.'()
    {
        given : 'A parent tensor.'
            Tsr X = Tsr.of([
                    [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                    [1000,   -2,    2, 1000, 1000, 1000, 1000, 1000],
                    [1000,    7,    7, 1000, 1000, 1000, 1000, 1000],
                    [1000, 1000, 1000,    8,    4, 1000, 1000, 1000],
                    [1000, 1000, 1000,   -3,    7, 1000, 1000, 1000]
            ])

        when : 'Extracting and transposing two distinct slices from the tensor above...'
            Tsr a = X[1..2, 1..2].T()
            Tsr b = X[3..4, 3..4].T()

        then : 'These matrices as String instances look as follows.'
            a.toString() ==
                "(2x2):[" +
                    "-2.0, 7.0, " + // -16 + 28 = 12 ;  6 + 49 = 55
                    "2.0, 7.0" +    //  16 + 28 = 44 ; -6 + 49 = 43
                "]"
            b.toString() ==
                 "(2x2):[" +
                     "8.0, -3.0, " +
                     "4.0, 7.0" +
                 "]"

        when : 'Both 2D matrices are being multiplied via the dot operation...'
            Tsr c = a.convDot(b)

        then : 'This produces the following matrix: '
            c.toString() == "(2x1x2):[12.0, 55.0, 44.0, 43.0]"

    }


    def 'Reshaping a slice works as expected.'()
    {
        given : 'A parent tensor.'
            Tsr X = Tsr.of([
                    [1000, 1000, 1000, 1000, ],
                    [1000,   -1,    4, 1000, ],
                    [1000,    2,    7, 1000, ],
                    [1000,    5,   -9, 1000, ],
                    [1000, 1000, 1000, 1000, ]
            ])

        when :
            Tsr a = X[1..3, 1..2].T()

        then :
            a.toString() ==
                    "(2x3):[" +
                    "-1.0, 2.0, 5.0, " +
                    "4.0, 7.0, -9.0" +
                    "]"

        when :
            Tsr b = Function.of("[-1, 0, 1]:(I[0])")(a)

        then :
            a.toString() == "(2x3):[" +
                    "-1.0, 2.0, 5.0, " +
                    "4.0, 7.0, -9.0" +
                    "]"
            b.toString() == "(1x2x3):[" +
                    "-1.0, 2.0, 5.0, " +
                    "4.0, 7.0, -9.0" +
                    "]"

    }

}
