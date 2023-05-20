package ut.ndim

import neureka.Neureka
import neureka.Tensor
import neureka.math.Function
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Permuting Slices of Tensors")
@Narrative('''

    Neureka provides a convenient way to permuting tensors
    even if they are slices of other tensors sharing the same underlying data.
    This is possible because of the under the hood indexing 
    abstractions provided by the `NDConfiguration` interface and its various implementations.

''')
@Subject([Tensor])
class Tensor_Slice_Permute_Spec extends Specification
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
        Tensor A = Tensor.of([
                    [  1d,  5d,  3d, -6d, -3d,  8d, -9d,  4d  ],
                    [  0d, -2d,  2d,  1d, -1d,  0d,  5d,  4d  ],
                    [ -6d,  7d,  7d, -2d,  9d,  0d,  1d, -1d  ],
                    [  4d,  4d, -1d,  8d,  4d, -3d,  2d, -9d  ],
                    [  7d,  5d, -2d, -3d,  7d, -8d,  5d,  0d  ]
            ])

        when : 'A slice of this tensor is being created...'
            Tensor a = A[1..3, 4..5]
        and : 'And also a slice of "a" with the same dimensionality, namely : "b".'
            Tensor b = a[]

        then : 'The slice and the parent are as expected.'
            A.toString() == "(5x8):[1.0, 5.0, 3.0, -6.0, -3.0, 8.0, -9.0, 4.0, 0.0, -2.0, 2.0, 1.0, -1.0, 0.0, 5.0, 4.0, -6.0, 7.0, 7.0, -2.0, 9.0, 0.0, 1.0, -1.0, 4.0, 4.0, -1.0, 8.0, 4.0, -3.0, 2.0, -9.0, 7.0, 5.0, -2.0, -3.0, 7.0, -8.0, 5.0, 0.0]"
            a.toString() == "(3x2):[-1.0, 0.0, 9.0, 0.0, 4.0, -3.0]"
        and : 'The slice "b" of the slice "a" is a different tensor that references the same underlying data!'
            a != b
            a.toString() == b.toString()

        when : 'The slice "a" is being permuted... (transposed)'
            Tensor c = a.T()

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
            var X = Tensor.of([
                        [1000d, 1000d, 1000d, 1000d, 1000d, 1000d, 1000d, 1000d],
                        [1000d,   -2d,    2d, 1000d, 1000d, 1000d, 1000d, 1000d],
                        [1000d,    7d,    7d, 1000d, 1000d, 1000d, 1000d, 1000d],
                        [1000d, 1000d, 1000d,    8d,    4d, 1000d, 1000d, 1000d],
                        [1000d, 1000d, 1000d,   -3d,    7d, 1000d, 1000d, 1000d]
                    ])

        when : 'Extracting and transposing two distinct slices from the tensor above...'
            var a = X[1..2, 1..2].T()
            var b = X[3..4, 3..4].T()

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
            var c = a.convDot(b)

        then : 'This produces the following matrix: '
            c.toString() == "(2x1x2):[12.0, 55.0, 44.0, 43.0]"

    }


    def 'Reshaping a slice works as expected.'()
    {
        given : 'A parent tensor from which we want to create slices.'
            var X = Tensor.of([
                        [1000d, 1000d, 1000d, 1000d, ],
                        [1000d,   -1d,    4d, 1000d, ],
                        [1000d,    2d,    7d, 1000d, ],
                        [1000d,    5d,   -9d, 1000d, ],
                        [1000d, 1000d, 1000d, 1000d, ]
                    ])

        when : 'We extract a slice from the tensor above...'
            var a = X[1..3, 1..2].T()

        then :
            a.toString() ==
                    "(2x3):[" +
                        "-1.0, 2.0, 5.0, " +
                        "4.0, 7.0, -9.0" +
                    "]"

        when :
            var b = Function.of("[-1, 0, 1]:(I[0])")(a)

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
