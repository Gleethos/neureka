package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.dtype.DataType
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Why not have a tensor of words?")
@Narrative('''

    Technically, tensors are merely fancy ND-arrays with some useful mathematical operations
    applicable to them...
    Therefore, there is no reason why a tensor would not also be able to store
    other kinds of objects besides numbers like strings for example.
    This specification ensures that tensors can hold and index many other things...

''')
class Tensor_As_Container_Spec extends Specification
{

    def setupSpec()
    {
        Neureka.get().reset()

        reportHeader """
           Although you can create a tensor of almost anything, you will
           not be able to execute operations on every kind of tensor...        
        """
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


    def 'Plus operator on String tensors works element-wise.'()
    {
        given : 'Two tensors filled with String objects containing various letters.'
            Tsr a = Tsr.of([2, 3], 'a'..'e')
            Tsr b = Tsr.of([2, 3], 'f'..'k')

        expect : 'These two tensors then look as expected.'
            a.toString() == '(2x3):[a, b, c, d, e, a]'
            b.toString() == '(2x3):[f, g, h, i, j, k]'

        and : 'They have a data type wrapping the String class.'
            a.dataType == DataType.of( String.class )
            b.dataType == DataType.of( String.class )

        when : 'We now apply the "+" operator to the tensors...'
            Tsr c = a + b

        then: 'This translates to the individual elements :'
            c.toString() == '(2x3):[af, bg, ch, di, ej, ak]'

    }

    def 'Tensor operations translate to custom data type "ComplexNumber".'()
    {
        given : 'We create 2 simple complex numbers'
            def c1 = new ComplexNumber(2.3, -1.54)
            def c2 = new ComplexNumber(1.0, 0.5)
        and : 'Then we use them to populate 2 matrices.'
            Tsr a = Tsr.of([3, 2], c1)
            Tsr b = Tsr.of([3, 2], c2)

        expect:
            a.toString() == "(3x2):[2.3-1.54i, 2.3-1.54i, 2.3-1.54i, 2.3-1.54i, 2.3-1.54i, 2.3-1.54i]"
            b.toString() == "(3x2):[1.0+0.5i, 1.0+0.5i, 1.0+0.5i, 1.0+0.5i, 1.0+0.5i, 1.0+0.5i]"
            a.isVirtual()
            b.isVirtual()
            (a+b).toString() == "(3x2):[3.3-1.04i, 3.3-1.04i, 3.3-1.04i, 3.3-1.04i, 3.3-1.04i, 3.3-1.04i]"
            (a-b).toString() == "(3x2):[1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i]"
            (a*b).toString() == "(3x2):[3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i]"
    }

    def 'More tensor operations translate to custom data type "ComplexNumber".'()
    {
        given : 'We create 2 simple matrices of complex numbers using filler lambdas '
            Tsr a = Tsr.of(
                        DataType.of(ComplexNumber.class),
                        [3, 2],
                        ( int i, int[] indices ) -> new ComplexNumber(indices[0], indices[1])
                    )
            Tsr b = Tsr.of(
                        DataType.of(ComplexNumber.class),
                        [3, 2],
                        ( int i, int[] indices ) -> new ComplexNumber(indices[1], indices[0])
                    )

        expect:
            a.toString() == "(3x2):[0.0+0.0i, 0.0+1.0i, 1.0+0.0i, 1.0+1.0i, 2.0+0.0i, 2.0+1.0i]"
            b.toString() == "(3x2):[0.0+0.0i, 1.0+0.0i, 0.0+1.0i, 1.0+1.0i, 0.0+2.0i, 1.0+2.0i]"
            !a.isVirtual()
            !b.isVirtual()
            (a+b).toString() == "(3x2):[0.0+0.0i, 1.0+1.0i, 1.0+1.0i, 2.0+2.0i, 2.0+2.0i, 3.0+3.0i]"
            (a-b).toString() == "(3x2):[0.0+0.0i, -1.0+1.0i, 1.0-1.0i, 0.0+0.0i, 2.0-2.0i, 1.0-1.0i]"
            (a*b).toString() == "(3x2):[0.0+0.0i, 0.0+1.0i, 0.0+1.0i, 0.0+2.0i, 0.0+4.0i, 0.0+5.0i]"
    }

    def 'We can apply predicates on the values of a tensor.'()
    {
        given : 'We create 3 different kinds of tensors.'
            var a = Tsr.of(Character).withShape(2, 3).andFill('x' as char, 'y' as char, 'z' as char)
            var b = Tsr.of(String).withShape(3, 2).andFill("Hey", "how", "are", "you", "?")
            var c = Tsr.ofDoubles().vector(42, 0.53, -2.5, 9, -12.92, 73)

        expect : 'We can check if every value of a tensor satisfies a condition.'
            a.every( item -> item instanceof Character )
            b.every( item -> item instanceof String )
            c.every( item -> item instanceof Double )
        and : 'We can see that this is only true of it applies to every value.'
            !a.every( item -> item == 'x' as char )
            !b.every( item -> item == "Hey" )
            !c.every( item -> item == 42 )
        and : 'When we want to check if at least one item meets the condition we can use the following:'
            a.any( item -> item == 'x' as char )
            c.any( item -> item == 42 )
            b.any( item -> item == "how" )
        and : 'As expected the negation will ensure that there is no item matching the provided condition.'
            !a.any( item -> item == '?' as char )
            !b.any( item -> item == "is" )
            !c.any( item -> item == 666 )
    }

    /**
     *  A class modeling a complex number which we can use as
     *  a custom number data type for tensors.
     */
    class ComplexNumber
    {
        private double real = 0.0
        private double imaginary = 0.0

        ComplexNumber(double real, double imaginary)
        {
            this.real = real
            this.imaginary = imaginary
        }

        ComplexNumber plus(ComplexNumber z2)
        {
            return new ComplexNumber(this.real + z2.real, this.imaginary + z2.imaginary)
        }

        ComplexNumber minus(ComplexNumber z2)
        {
            return new ComplexNumber(this.real - z2.real, this.imaginary - z2.imaginary)
        }

        ComplexNumber multiply(ComplexNumber z2)
        {
            double _real = this.real*z2.real - this.imaginary*z2.imaginary
            double _imaginary = this.real*z2.imaginary + this.imaginary*z2.real
            return new ComplexNumber(_real,_imaginary)
        }

        ComplexNumber divide(ComplexNumber z2)
        {
            ComplexNumber output = multiply(z2.conjugate())
            double div = Math.pow(z2.mod(),2)
            return new ComplexNumber(output.real/div,output.imaginary/div)
        }

        double mod()
        {
            return Math.sqrt(Math.pow(this.real,2) + Math.pow(this.imaginary,2))
        }

        ComplexNumber pow(int power)
        {
            ComplexNumber output = new ComplexNumber(this.real,this.imaginary)
            for(int i = 1; i < power; i++)
            {
                double _real = output.real*this.real - output.imaginary*this.imaginary
                double _imaginary = output.real*this.imaginary + output.imaginary*this.real
                output = new ComplexNumber(_real,_imaginary)
            }
            return output
        }

        ComplexNumber inverse()
        {
            return new ComplexNumber(1,0).divide(this)
        }

        ComplexNumber conjugate()
        {
            return new ComplexNumber(this.real,-this.imaginary)
        }

        ComplexNumber square()
        {
            double _real = this.real*this.real - this.imaginary*this.imaginary
            double _imaginary = 2*this.real*this.imaginary
            return new ComplexNumber(_real,_imaginary)
        }

        @Override
        String toString()
        {
            String re = this.real + ""
            String im
            if (this.imaginary < 0) im = this.imaginary+"i"
            else im = "+"+this.imaginary+"i"
            return re + im
        }

        @Override
        final boolean equals(Object z)
        {
            if (!(z instanceof ComplexNumber)) return false
            ComplexNumber a = (ComplexNumber) z
            return (real == a.real) && (imaginary == a.imaginary)
        }
    }
}
