package it.autograd

import neureka.Neureka
import neureka.Tsr
import neureka.utility.TsrAsString
import spock.lang.Specification

/**
 * These tests were originally JUnit test cases.
 * They have been ported to Spock tests almost without syntax modification.
 */
class Autograd_Tensor_Integration_Tests extends Specification
{
    def setupSpec()
    {
        reportHeader """
            <h2> Autograd Tensor Behavior </h2>
            <p>
                Specified below is the behavior of the autograd system.
            </p>
        """
    }

    def setup() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = "dgc"
    }

    def 'Test basic autograd behaviour. (Not on device)'()
    {
        given: 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false);
        and: 'Tensor legacy view is set to true.'
            Neureka.instance().settings().view().setIsUsingLegacyView(true);

        and: 'Three scalar tensors "x", "b", "w" are being instantiated, and "x" requires gradients.'
            Tsr x = new Tsr(new int[]{1}, 3).setRqsGradient(true);
            Tsr b = new Tsr(new int[]{1}, -4);
            Tsr w = new Tsr(new int[]{1}, 2);
        /**
         *      ((3-4)*2)^2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        when: 'A new tensor is being calculated by the equation "((i0+i1)*i2)^2".'
            Tsr y = new Tsr(new Tsr[]{x, b, w}, "((i0+i1)*i2)^2");
        then: 'The resulting tensor should contain "[1]:(4.0); ->d[1]:(-8.0), " where the last part is a derivative.'
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0), ");

        when: 'We call the "backward" method on this tensor...'
            y.backward(new Tsr(2));
        then : 'The source tensor which requires gradients will have the gradient "-16".'
            x.toString().contains("-16.0");

        when : 'We create a new tensor via the same equation but applied in a different way...'
            y = new Tsr("(","(",x,"+",b,")","*",w,")^2");
        then : 'The will produce the same result once again.'
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0), ");

        when : 'Whe also call the "backward" method again...'
            y.backward(new Tsr(1));
        then : 'Then the accumulated gradient in the source tensor which requires gradients will be as expected.'
            x.toString().contains("-24.0");

        when : 'We execute the same equation once more...'
            y = new Tsr("((",x,"+",b,")*",w,")^2");
        then : 'The result will be as expected.'
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0), ");

        when : 'We call "backward" with -1 as error...'
            y.backward(new Tsr(-1));
        then : 'This will change the gradient of "x" accordingly.'
            x.toString().contains("-16.0");
    }

    def 'Second-Test "x-mul" autograd behaviour. (Not on device)'()
    {
        given : 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.instance().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false);
        and: 'Tensor legacy view is set to true.'
            Neureka.instance().settings().view().setIsUsingLegacyView(true)
        when :
            def x = new Tsr(
                    new int[]{3, 3},
                    new double[]{
                            1, 2, 5,
                            -1, 4, -2,
                            -2, 3, 4,
                    }
            );
            def y = new Tsr(
                    new int[]{2, 2},
                    new double[]{
                            -1, 3,
                            2, 3,
                    }).setRqsGradient(true);

        then : y.toString().contains(":g:(null)");
        when : def z = new Tsr(new Tsr[]{x, y}, "I0xi1");
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z = new Tsr(new Object[]{x, "x", y});
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z.backward(new Tsr(new int[]{2, 2}, 1));
        then : y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)");
        //---
        when :
        //--- again but now reverse: (outcome should not change...)
        x = new Tsr(
                new int[]{3, 3},
                new double[]{
                        1, 2, 5,
                        -1, 4, -2,
                        -2, 3, 4,
                }
        );
        y = new Tsr(
                new int[]{2, 2},
                new double[]{
                        -1, 3,
                        2, 3,
                }).setRqsGradient(true);

        then : y.toString().contains(":g:(null)");
        when : z = new Tsr(new Tsr[]{y, x}, "I0xi1");
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z = new Tsr(new Object[]{y, "x", x});
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)");

        when : z.backward(new Tsr(new int[]{2, 2}, 1));
        then : y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)");
        //====
        when :
        x = new Tsr(new int[]{1}, 3);
        Tsr b = new Tsr(new int[]{1}, -5);
        Tsr w = new Tsr(new int[]{1}, -2);
        z = new Tsr(new Tsr[]{x, b, w}, "I0*i1*i2");
        then : z.toString().contains("[1]:(30.0)");

        when :
        x = new Tsr(new int[]{1}, 4).setRqsGradient(true);
        b = new Tsr(new int[]{1}, 0.5);
        w = new Tsr(new int[]{1}, 0.5);
        y = new Tsr(new Tsr[]{x, b, w}, "(2^i0^i1^i2^2");
        then :
            y.toString().contains("[1]:(4.0);");
            y.toString().contains(" ->d[1]:(1.38629E0), ");
        //===
        //TODO: add tests using more then 1 function and check if the graph is being built correctly!
    }

    def 'A tensor used as derivative within a computation graph will throw exception when trying to deleting it.'()
    {
        given : 'A new tensor "a" requiring autograd.'
            Tsr a = new Tsr(1).setRqsGradient(true)

        and : 'A second tensor "b".'
            Tsr b = new Tsr(2)

        when : 'Both tensors are being multiplied via the "dot" method.'
            Tsr c = a.dot(b)

        and : 'One tries to delete tensor "b"...'
            b.delete()

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message == "Cannot delete a tensor which is used as derivative by the AD computation graph!"
    }


}
