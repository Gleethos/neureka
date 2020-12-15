package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.dtype.DataType
import spock.lang.Specification

class Tensor_As_Container_Integration_Tests extends Specification
{

    def setupSpec()
    {
        Neureka.instance().reset()

        reportHeader """
                         
        """
    }


    def 'Operations on tensors of String objects work elementwise.'()
    {
        given : ''
            Tsr a = new Tsr([2, 3], 'a'..'e')
            Tsr b = new Tsr([2, 3], 'f'..'k')

        expect :
            a.toString() == '(2x3):[a, b, c, d, e, a]'
            b.toString() == '(2x3):[f, g, h, i, j, k]'
            a.dataType == DataType.instance( String.class )
            b.dataType == DataType.instance( String.class )

        when :
            Tsr c = a + b

        then:
            c.toString() == '(2x3):[af, bg, ch, di, ej, ak]'

    }

    def 'Basic operations work with custom data types.'()
    {
        given :
            def c1 = new ComplexNumber(2.3, -1.54)
            def c2 = new ComplexNumber(1.0, 0.5)
        and :
            Tsr a = new Tsr([3, 2], c1)
            Tsr b = new Tsr([3, 2], c2)

        expect:
            a.toString() == "(3x2):[2.3-1.54i, 2.3-1.54i, 2.3-1.54i, 2.3-1.54i, 2.3-1.54i, 2.3-1.54i]"
            b.toString() == "(3x2):[1.0+0.5i, 1.0+0.5i, 1.0+0.5i, 1.0+0.5i, 1.0+0.5i, 1.0+0.5i]"
            a.isVirtual()
            b.isVirtual()
            (a+b).toString() == "(3x2):[3.3-1.04i, 3.3-1.04i, 3.3-1.04i, 3.3-1.04i, 3.3-1.04i, 3.3-1.04i]"
            (a-b).toString() == "(3x2):[1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i, 1.2999999999999998-2.04i]"
            (a*b).toString() == "(3x2):[3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i, 3.07-0.3900000000000001i]"
    }


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
