package ut.autograd

import neureka.Neureka
import neureka.Shape
import neureka.Tensor
import neureka.view.NDPrintSettings
import spock.lang.Specification

/**
 * These tests were originally JUnit test cases.
 * They have been ported to Spock tests almost without syntax modification.
 */
class Autograd_Tensor_Spec extends Specification
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

    def 'Test basic autograd behaviour. (Not on device)'()
    {
        given: 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
        and: 'Tensor legacy view is set to true.'
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)

        and: 'Three scalar tensors "x", "b", "w" are being instantiated, and "x" requires gradients.'
            Tensor x = Tensor.of(Shape.of(1), 3).setRqsGradient(true)
            Tensor b = Tensor.of(Shape.of(1), -4)
            Tensor w = Tensor.of(Shape.of(1), 2)
        /**
         *      ((3-4)*2)**2 = 4
         *  dx:   8*3 - 32  = -8
         * */
        when: 'A new tensor is being calculated by the equation "((i0+i1)*i2)**2".'
            Tensor y = Tensor.of("((i0+i1)*i2)**2", x, b, w)
        then: 'The resulting tensor should contain "[1]:(4.0); ->d[1]:(-8.0), " where the last part is a derivative.'
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when: 'We call the "backward" method on this tensor...'
            y.backward(Tensor.of(2d))
        then : 'The source tensor which requires gradients will have the gradient "-16".'
            x.toString().contains("-16.0")

        when : 'We create a new tensor via the same equation but applied in a different way...'
            y = Tensor.of("(","(",x,"+",b,")","*",w,")**2")
        then : 'The will produce the same result once again.'
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when : 'We also call the "backward" method again...'
            y.backward(Tensor.of(1d))
        then : 'The accumulated gradient in the source tensor which requires gradients will be as expected.'
            x.toString().contains("-24.0")

        when : 'We execute the same equation once more...'
            y = Tensor.of("((",x,"+",b,")*",w,")**2")
        then : 'The result will be as expected.'
            y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when : 'We call "backward" with -1 as error...'
            y.backward(Tensor.of(-1d))
        then : 'This will change the gradient of "x" accordingly.'
            x.toString().contains("-16.0")
    }

    def 'Second-Test "x-mul" autograd behaviour. (Not on device)'( Class<?> type )
    {
        given : 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
        and: 'Tensor legacy view is set to true.'
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
        when :
            def x = Tensor.ofDoubles()
                            .withShape(3, 3)
                            .andFill(
                                    1.0, 2.0, 5.0,
                                    -1.0, 4.0, -2.0,
                                    -2.0, 3.0, 4.0,
                            )
            def y = Tensor.of(
                    Shape.of(2, 2),
                    new double[]{
                            -1, 3,
                            2, 3,
                    }).setRqsGradient(true)

        then : y.toString().contains(":g:(null)")
        when : def z = Tensor.of("I0xi1", x, y)
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)")

        when : z = Tensor.of(new Object[]{x, "x", y})
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)")

        when : z.backward(Tensor.of(Shape.of(2, 2), 1))
        then : y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)")

        when :
            // again but now reverse: (outcome should not change...)
            x = Tensor.of(Shape.of(3, 3),
                        new double[]{
                                1, 2, 5,
                                -1, 4, -2,
                                -2, 3, 4,
                        }
                ).mut.toType(type)
            y = Tensor.of(Shape.of(2, 2),
                    new double[]{
                            -1, 3,
                            2, 3,
                    }).setRqsGradient(true).mut.toType(type)

        then : y.toString().contains(":g:(null)")
        when : z = Tensor.of("I0xi1", y, x)
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)")
        and : z.itemType == type

        when : z = Tensor.of(y, "x", x)
        then : z.toString().contains("[2x2]:(15.0, 15.0, 18.0, 8.0)")
        and : z.itemType == type

        when : z.backward(Tensor.of(Shape.of(2, 2), 1d))
        then : y.toString().contains("[2x2]:(-1.0, 3.0, 2.0, 3.0):g:(6.0, 9.0, 4.0, 9.0)")
        //====
        when :
            x = Tensor.of(Shape.of(1), 3d).mut.toType(type)
            Tensor b = Tensor.of(Shape.of(1), -5d).mut.toType(type)
            Tensor w = Tensor.of(Shape.of(1), -2d).mut.toType(type)
            z = Tensor.of("I0*i1*i2", x, b, w)
        then : z.toString().contains("[1]:(30.0)")
        and : z.itemType == type

        when :
            x = Tensor.of(Shape.of(1), 4d).setRqsGradient(true).mut.toType(type)
            b = Tensor.of(Shape.of(1), 0.5).mut.toType(type)
            w = Tensor.of(Shape.of(1), 0.5).mut.toType(type)
            y = Tensor.of("(2**i0**i1**i2**2", x, b, w)
        then :
            y.toString().contains("[1]:(9.24238);")
            y.toString().contains(" ->d[1]:(4.32078)")
        and :
            y.itemType == type
            //TODO: add tests using more then 1 function and check if the graph is being built correctly!
        where :
            type << [Double, Float]

    }

    def 'A tensor used as derivative within a computation graph will throw exception when trying to deleting it.'()
    {
        given : 'A new tensor "a" requiring autograd.'
            Tensor a = Tensor.of(1d).setRqsGradient(true)

        and : 'A second tensor "b".'
        Tensor b = Tensor.of(2d)

        when : 'Both tensors are being multiplied via the "dot" method.'
            Tensor c = a.convDot(b)

        and : 'One tries to delete tensor "b"...'
            b.getMut().delete()

        then : 'An exception is being thrown.'
            def exception = thrown(IllegalStateException)
            exception.message == "Cannot delete a tensor which is used as derivative by the AD computation graph!"
    }


}
