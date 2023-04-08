package ut.tensors

import neureka.Neureka
import neureka.Shape
import neureka.Tsr
import spock.lang.*

@Title("Tensor Dot Products")
@Narrative('''

    A tensor can also be a simple vector, which is a tensor of rank 1.
    This specification demonstrates how to perform dot products on tensors of rank 1.
    
''')
@Subject([Tsr])
class Tensor_Dot_Product_Spec extends Specification
{
    def 'The "dot" method calculates the dot product between vectors.'()
    {
        given : 'two vectors, a and b, of length 2.'
            var a = Tsr.of(3f, 2f)
            var b = Tsr.of(1f, -0.5f)
        when : 'we calculate the dot product of a and b.'
            var result = a.dot(b)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ 3f * 1f + 2f * -0.5f ]
    }

    def 'You can slice a Matrix into vectors and then used them for dot products.'()
    {
        given : 'A matrix we want to slice.'
            var m = Tsr.of(1f..4f).reshape(2,2)
        when : 'we slice the matrix into two vectors.'
            var a = m.slice().axis(0).at(0).get()
            var b = m.slice().axis(0).at(1).get()
        and : 'We perform a dot product on the two vectors.'
            var c = a.dot(b)

        then : 'the result is a scalar.'
            c.shape == Shape.of(1)
            c.items == [ 1f * 3f + 2f * 4f ]
    }

    def 'The "dot" operation supports autograd.'()
    {
        reportInfo """
            The "dot" operation supports autograd.
            This means that you can use it to calculate the gradient of a weight tensor.
            This is useful for when you want to build a neural network or some other machine learning model.
        """
        given : 'two vectors, a and b, of length 2.'
            var a = Tsr.of(4f, -1f, 2f)
            var w = Tsr.of(1f, 0f, 0f).setRqsGradient(true)
        when : 'we calculate the dot product of a and w.'
            var result = a.dot(w)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ 4f ]

        when : 'we calculate the gradient of the result with respect to w, divided by 2.'
            result.backward(0.5f)
        then : 'the gradient of w is a vector of length 3.'
            w.gradient.isPresent()
            w.gradient.get().shape == Shape.of(3)
            w.gradient.get().items == [ 2f, -0.5f, 1f ]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == 'GPU' }) // We need to assure that this system supports OpenCL!
    def 'The dot product operation runs on any device.'( String device )
    {
        reportInfo """
            The dot product operation runs on any device that 
            supports OpenCL (meaning that it has OpenCL drivers installed).
        """
        given : 'A pair of vector tensors which we move to the device!'
            var a = Tsr.of(-1f, -3f, 0f, 4f, 2f).to( device )
            var b = Tsr.of( 1f,  2f, 7f, -1f, 3f).to( device )
        when : 'we calculate the dot product of a and b.'
            var result = a.dot(b)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ -1f * 1f + -3f * 2f + 0f * 7f + 4f * -1f + 2f * 3f ]

        where :
            device << [ 'CPU', 'GPU' ]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == 'GPU' }) // We need to assure that this system supports OpenCL!
    def 'The dot operation works for virtual tensors as well.'( String device )
    {
        given : 'A pair of vector tensors which we move to the device!'
            var a = Tsr.of(Shape.of(8), 3f).to(device)
            var b = Tsr.of(Shape.of(8), 3f).to(device)
        expect : 'the tensors are virtual.'
            a.isVirtual() // They are scalars in disguise!
            b.isVirtual()
        when : 'we calculate the dot product of a and b.'
            var result = a.dot(b)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ 3f * 3f * 8f ]
        where :
            device << [ 'CPU', 'GPU' ]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == 'GPU' }) // We need to assure that this system supports OpenCL!
    def 'The dot operation work even when one tensor is virtual.'( String device )
    {
        given : 'A pair of vector tensors which we move to the device!'
            var a = Tsr.of(Shape.of(8), 3f).to(device)
            var b = Tsr.of(Shape.of(8), new float[]{3f, 4f, -1f}).to(device)
        expect : 'the tensors are virtual.'
            a.isVirtual() // They are scalars in disguise!
            !b.isVirtual()
        when : 'we calculate the dot product of a and b.'
            var result = a.dot(b)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == [ 57f ]
        where :
            device << [ 'CPU', 'GPU' ]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == 'GPU' }) // We need to assure that this system supports OpenCL!
    def 'The dot product works across different types and devices.'(
        String device, Object data1, Object data2, List expected
    ) {
        given : 'A pair of vector tensors which we move to the device!'
            var a = Tsr.of(data1).to(device)
            var b = Tsr.of(data2).to(device)
        when : 'we calculate the dot product of a and b.'
            var result = a.dot(b)
        then : 'the result is a scalar.'
            result.shape == Shape.of(1)
            result.items == expected
        where :
            device | data1                       | data2                      | expected
            'CPU'  | [ 8f, -4f, -1f ] as float[] | [ 1f, 2f, 4f ] as float[]  | [ 8f * 1f + -4f * 2f + -1f * 4f ]
            'CPU'  | [ 42f ] as float[]          | [ 56f ] as float[]         | [ 42f * 56f ]
            'CPU'  | [ 8d, -4d, -1d ] as double[]| [ 1d, 2d, 4d ] as double[] | [ 8d * 1d + -4d * 2d + -1d * 4d ]
            'CPU'  | [ 2d, 3d, 4d ] as double[]  | [ 0d, 2d, 3d ] as double[] | [ 2d * 0d + 3d * 2d + 4d * 3d ]
            'CPU'  | [ 1d, -4d ] as double[]     | [ 4d, 2d ] as double[]     | [ 1d * 4d + -4d * 2d ]
            'CPU'  | [ 8, -4, -1 ] as int[]      | [ 1, 2, 4 ] as int[]       | [ 8 * 1 + -4 * 2 + -1 * 4 ]
            'CPU'  | [ 42 ] as int[]             | [ 56 ] as int[]            | [ 42 * 56 ]
            'CPU'  | [ 2, 3, 4 ] as long[]       | [ 0, 2, 3 ] as long[]      | [ 2 * 0 + 3 * 2 + 4 * 3 ]
            'CPU'  | [ 1, -4 ] as long[]         | [ 4, 2 ] as long[]         | [ 1 * 4 + -4 * 2 ]
            'CPU'  | [ 42 ] as long[]            | [ 56 ] as long[]           | [ 42 * 56 ]

            'GPU'  | [ 8f, -4f, -1f ] as float[] | [ 1f, 2f, 4f ] as float[]  | [ 8f * 1f + -4f * 2f + -1f * 4f ]
            'GPU'  | [ 42f ] as float[]          | [ 56f ] as float[]         | [ 42f * 56f ]
    }

}
