package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import neureka.common.utility.SettingsLoader
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.dtype.DataType
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Specification

import java.util.function.BiFunction

class Tensor_Operation_Integration_Spec extends Specification
{

    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
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

    def 'Test "x-mul" operator produces expected results. (Not on device)'(
            String expected
    ) {
        reportInfo """
            The 'x' operator performs convolution on the provided operands.
        """

        given: 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
        and: 'Tensor legacy view is set to true.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and: 'Two new 3D tensor instances with the shapes: [2x3x1] & [1x3x2].'
            //Same test again but this time with reversed indexing:
            def x = Tsr.of(
                new int[]{2, 3, 1},
                new double[]{
                        3,  2, -1, //<=- Format of legacy : false
                        -2,  2,  4
                        /* Format otherwise :
                             3,  2,
                            -1, -2,
                             2,  4
                         */
                }
            );
            def y = Tsr.of(
                new int[]{1, 3, 2},
                new double[]{
                        4, -1,
                        3,  2,
                        3, -1
                })
            /*
                15, 2,
                10, 2
            */
        when : 'The x-mul result is being instantiated by passing a simple equation to the tensor constructor.'
            def z = Tsr.of("I0xi1", x, y)
        then: 'The result contains the expected String.'
            z.toString().contains(expected)

        when: 'The x-mul result is being instantiated by passing a object array containing equation parameters and syntax.'
            z = Tsr.of(new Object[]{x, "x", y})
        then: 'The result contains the expected String.'
            z.toString().contains(expected)

        where :
            expected << ["[2x1x2]:(15.0, 2.0, 10.0, 2.0)"]
    }


    def 'The "dot" operation reshapes and produces valid "x" operation result.'()
    {
        given : 'Two multi-dimensional tensors.'
            def a = Tsr.of([1, 4, 4, 1   ], 4d..12d)
            def b = Tsr.of([1, 3, 5, 2, 1], -5d..3d)

        when : 'The "dot" method is being called on "a" receiving "b"...'
            Tsr c = a.convDot(b)

        then : 'The result tensor contains the expected shape.'
            c.toString().contains("(4x2x5x2)")
    }


    def 'The "matMul" operation produces the expected result.'(
            Double[] A, Double[] B, int M, int K, int N, double[] expectedC
    ) {
        given : 'Two 2-dimensional tensors.'
            Tsr a = Tsr.of(Double.class).withShape(M, K).andFill(A)
            Tsr b = Tsr.of(Double.class).withShape(K, N).andFill(B)

        when : 'The "matMul" method is being called on "a" receiving "b"...'
            Tsr c = a.matMul(b)

        then : 'The result tensor contains the expected shape and values.'
            c.toString() == "(${M}x${N}):$expectedC"

        where : 'We use the following data and matrix dimensions!'
            A            | B                  | M | K | N || expectedC
            [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 1, 0, 0, 1 ]
            [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 0.5 ]
            [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2.0, 3.0, -1.0, -1.5 ]
    }

    def 'New method "asFunction" of String added at runtime is callable by groovy and also works.'(
            String code, String expected
    ) {
        given :
            Tsr a = Tsr.of([1,2], [3, 2])
            Tsr b = Tsr.of([2,1], [-1, 4])
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        when : 'The groovy code is being evaluated.'
            Tsr c = new GroovyShell(binding).evaluate((code)) as Tsr

        then : 'The resulting tensor (toString) will contain the expected String.'
            c.toString().contains(expected)

        where :
            code                                       || expected
            '"I[0]xI[1]".asFunction()([a, b])'         || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            '"I[0]xI[1]"[a, b]'                        || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            '"i0 x i1"%[a, b]'                         || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            '"i0"%a'                                   || "(1x2):[3.0, 2.0]"
    }

    def 'New operator methods added to "SDK-types" at runtime are callable by groovy and also work.'(
            String code, String expected
    ) {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Tsr a = Tsr.of(5)
            Tsr b = Tsr.of(3)
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        when : '...calling methods on types like Double and Integer that receive Tsr instances...'
            Tsr c = new GroovyShell(binding).evaluate((code)) as Tsr

        then : 'The resulting tensor (toString) will contain the expected String.'
            c.toString().contains(expected)

        where :
            code       || expected
            '(2+a)'    || "7.0"
            '(2*b)'    || "6.0"
            '(6/b)'    || "2.0"
            '(2^b)'    || "8.0"
            '(2**b)'   || "8.0"
            '(4-a)'    || "-1.0"
            '(2.0+a)'  || "7.0"
            '(2.0*b)'  || "6.0"
            '(6.0/b)'  || "2.0"
            '(2.0^b)'  || "8.0"
            '(2.0**b)' || "8.0"
            '(4.0-a)'  || "-1.0"

    }

    def 'Overloaded operation methods on tensors produce expected results when called.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Tsr a = Tsr.of(2).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Tsr c = Tsr.of(3).setRqsGradient(true)

        expect :
            ( a / a                     ).toString().contains("[1]:(1.0)")
            ( c % a                     ).toString().contains("[1]:(1.0)")
            ( ( ( b / b ) ^ c % a ) * 3 ).toString().contains("[1]:(3.0)")
            ( a *= b                    ).toString().contains("(-8.0)")
            ( a += -c                   ).toString().contains("(-11.0)")
            ( a -= c                    ).toString().contains("(-14.0)")
            ( a /= Tsr.of(2)      ).toString().contains("(-7.0)")
            ( a %= c                    ).toString().contains("(-1.0)")
    }

    def 'Manual convolution produces expected result.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
            Tsr a = Tsr.of([100, 100], 3..19)
            Tsr x = a[1..-2,0..-1]
            Tsr y = a[0..-3,0..-1]
            Tsr z = a[2..-1,0..-1]

        when :
            Tsr rowconvol = x + y + z // (98, 100) (98, 100) (98, 100)
            Tsr k = rowconvol[0..-1,1..-2]
            Tsr v = rowconvol[0..-1,0..-3]
            Tsr j = rowconvol[0..-1,2..-1]
            Tsr u = a[1..-2,1..-2]
            Tsr colconvol = k + v + j - 9 * u // (98, 98)+(98, 98)+(98, 98)-9*(98, 98)
            String xAsStr = x.toString()
            String yAsStr = y.toString()
            String zAsStr = z.toString()
            String rcAsStr = rowconvol.toString()
            String kAsStr = k.toString()
            String vAsStr = v.toString()
            String jAsStr = j.toString()
            String uAsStr = u.toString()

        then :
            assert xAsStr.contains("(98x100)")
            assert xAsStr.contains("):[18.0, 19.0, 3.0, 4.0, 5.0")

            assert yAsStr.contains("(98x100)")
            assert yAsStr.contains("):[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0")

            assert zAsStr.contains("(98x100)")
            zAsStr.contains("):[16.0, 17.0, 18.0, 19.0, 3.0")

            assert rcAsStr.contains("(98x100)")
            assert rcAsStr.contains("):[37.0, 40.0, 26.0, 29.0, 15.0, 18.0")

            assert kAsStr.contains("(98x98)")
            kAsStr.contains("):[40.0, 26.0, 29.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0")

            assert vAsStr.contains("(98x98)")
            assert vAsStr.contains("):[37.0, 40.0, 26.0, 29.0, 15.0, 18.0, 21.0")

            assert jAsStr.contains("(98x98)")
            jAsStr.contains("):[26.0, 29.0, 15.0, 18.0, 21.0, 24.0")

            assert uAsStr.contains("(98x98)")
            assert uAsStr.contains("):[19.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, ")
            String ccAsStr = colconvol.toString()
            assert ccAsStr.contains("(98x98)")
            ccAsStr.contains("(98x98):[-68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, " +
                    "-34.0, -68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, -34.0, " +
                    "-68.0, 68.0, 34.0, 17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -17.0, ... + 9554 more]")
    }


    def 'Very simple manual convolution produces expected result.'(
         Device device
    ) {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
            Tsr a = Tsr.of([4, 4], 0..16).to( device )

            Tsr x = a[1..-2,0..-1]
            Tsr y = a[0..-3,0..-1]
            Tsr z = a[2..-1,0..-1]

        when :
            Tsr rowconvol = x + y + z
            Tsr k = rowconvol[0..-1,1..-2]
            Tsr v = rowconvol[0..-1,0..-3]
            Tsr j = rowconvol[0..-1,2..-1]
            Tsr u = a[1..-2,1..-2]
            Tsr colconvol = k + v + j - 9 * u
            String xAsStr = x.toString()
            String yAsStr = y.toString()
            String zAsStr = z.toString()
            String rcAsStr = rowconvol.toString()
            String kAsStr = k.toString()
            String vAsStr = v.toString()
            String jAsStr = j.toString()
            String uAsStr = u.toString()

        then :
            assert xAsStr.contains("(2x4):[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]")
            assert yAsStr.contains("(2x4):[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]")
            assert zAsStr.contains("(2x4):[8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]")
            assert rcAsStr.contains("(2x4):[12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0]")
            assert kAsStr.contains("(2x2):[15.0, 18.0, 27.0, 30.0]")
            assert vAsStr.contains("(2x2):[12.0, 15.0, 24.0, 27.0]")
            assert jAsStr.contains("(2x2):[18.0, 21.0, 30.0, 33.0]")
            assert uAsStr.contains("(2x2):[5.0, 6.0, 9.0, 10.0]")
            String ccAsStr = colconvol.toString()
            assert ccAsStr.contains("(2x2):[0.0, 0.0, 0.0, 0.0]")

        where : 'The following data is being used for tensor instantiation :'
            device  << [CPU.get(), Device.find("openCL") ]
    }

    //This needs verification!
    //def 'Simple manual convolution produces expected result.'()
    //{
    //    given :
    //        Neureka.instance().reset()
    //        Neureka.instance().settings().view().setIsUsingLegacyView(false)
    //        Tsr a = Tsr.of([8, 8], 0..63)
    //        Tsr x = a[1..-2,0..-1]
    //        Tsr y = a[0..-3,0..-1]
    //        Tsr z = a[2..-1,0..-1]
//
    //    when :
    //        Tsr rowconvol = x + y + z
    //        Tsr k = rowconvol[0..-1,1..-2]
    //        Tsr v = rowconvol[0..-1,0..-3]
    //        Tsr j = rowconvol[0..-1,2..-1]
    //        Tsr u = a[1..-2,1..-2]
    //        Tsr colconvol = k + v + j - 9 * u
    //        String rcAsStr = rowconvol.toString()
//
    //    then :
    //        assert rcAsStr.contains("(6x8)")
    //        assert rcAsStr.contains("[0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, ")
    //        String ccAsStr = colconvol.toString()
    //        assert ccAsStr.contains("(6x6)")
    //        assert ccAsStr.contains("[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]")
    //}


    def 'Simple slice addition produces expected result.'(
            Device device
    ) {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
            Tsr a = Tsr.of([11, 11], 3..19).to( device )
            Tsr x = a[1..-2,0..-1]
            Tsr y = a[0..-3,0..-1]

        when :
            Tsr rowconvol = x + y
            String rcAsStr = rowconvol.toString({it.setRowLimit(50)})

        then :
            assert rcAsStr.contains("(9x11):[17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, " +
                    "26.0, 28.0, 30.0, 32.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, " +
                    "26.0, 28.0, 30.0, 32.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, ... + 49 more]")

        where : 'The following data is being used for tensor instantiation :'
            device  << [
                    CPU.get(),
                    Device.find("openCL")
            ]
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCL() && device == 'GPU' })
    def 'Auto reshaping and broadcasting works and the result can be back propagated.'(// TODO: Cover more broadcasting operations!
            Class<Object> type, boolean whichGrad, List<Integer> bShape,
            BiFunction<Tsr<?>, Tsr<?>, Tsr<?>> operation, String cValue, String wGradient, String device
    ) {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and :
            String wValue = whichGrad
                                ? "8.0" + ( bShape.inject(1, {x,y->x*y}) > 1 ? ", 9.0" : "" )
                                : "1.0, 2.0, 3.0, 4.0"
        and :
            def aShape = [2, 2]
        and :
            Tsr<Double> a = Tsr.of(aShape, 1d..5d).setRqsGradient(!whichGrad).to(Device.find(device))
            Tsr<Double> b = Tsr.of(bShape, 8d..9d).setRqsGradient(whichGrad).to(Device.find(device))
        and :
            a.mutate.toType(type)
            b.mutate.toType(type)
        and :
            String wShape = ( whichGrad ? bShape : aShape ).join("x")
            Tsr    w      = ( whichGrad ? b      : a      )

        expect :
            a.valueClass == type
            b.valueClass == type

        when :
            Tsr c = operation.apply(a, b)
        then :
            c.toString().startsWith("[2x2]:($cValue)")
            w.toString() == "[$wShape]:($wValue):g:(null)"

        when :
            c.backward(Tsr.of([2, 2], [5, -2, 7, 3]))
        then :
            w.toString() == "[$wShape]:($wValue):g:($wGradient)"

        when :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
        then :
            c.toString() == "(2x2):[$cValue]"

        where:
            type   | whichGrad | bShape |    operation      ||     cValue              | wGradient                 | device

            Double | false     | [1]    | { x, y -> x + y } || "9.0, 10.0, 11.0, 12.0" | "5.0, -2.0, 7.0, 3.0"     | 'CPU'
            Double | false     | [1]    | { x, y -> x + y } || "9.0, 10.0, 11.0, 12.0" | "5.0, -2.0, 7.0, 3.0"     | 'GPU'
            Float  | false     | [1]    | { x, y -> x + y } || "9.0, 10.0, 11.0, 12.0" | "5.0, -2.0, 7.0, 3.0"     | 'CPU'
            Float  | false     | [1]    | { x, y -> x + y } || "9.0, 10.0, 11.0, 12.0" | "5.0, -2.0, 7.0, 3.0"     | 'GPU'

            Double | false     | [1]    | { x, y -> x * y } || "8.0, 16.0, 24.0, 32.0" | "40.0, -16.0, 56.0, 24.0" | 'CPU'
            Double | false     | [1]    | { x, y -> x * y } || "8.0, 16.0, 24.0, 32.0" | "40.0, -16.0, 56.0, 24.0" | 'GPU'
            Float  | false     | [1]    | { x, y -> x * y } || "8.0, 16.0, 24.0, 32.0" | "40.0, -16.0, 56.0, 24.0" | 'CPU'
            Float  | false     | [1]    | { x, y -> x * y } || "8.0, 16.0, 24.0, 32.0" | "40.0, -16.0, 56.0, 24.0" | 'GPU'

            Double | true      | [2,1]  | { x, y -> x + y } || "9.0, 10.0, 12.0, 13.0" | "3.0, 10.0"               | 'CPU'
            Double | true      | [1]    | { x, y -> x + y } || "9.0, 10.0, 11.0, 12.0" | "13.0"                    | 'CPU'
            //Double |  true     | [2,1]  | { x, y -> x + y } || "9.0, 10.0, 12.0, 13.0" | "3.0, 10.0"               | 'GPU'
            Double | true      | [1]    | { x, y -> x + y } || "9.0, 10.0, 11.0, 12.0" | "13.0"                    | 'GPU'

            Double | true      | [1,2]  | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'CPU'
            Double | true      | [2]    | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'CPU'
            Double | true      | [1,2]  | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'GPU'
            Double | true      | [2]    | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'GPU'
            Float  | true      | [1,2]  | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'CPU'
            Float  | true      | [2]    | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'CPU'
            Float  | true      | [1,2]  | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'GPU'
            Float  | true      | [2]    | { x, y -> x + y } || "9.0, 11.0, 11.0, 13.0" | "12.0, 1.0"               | 'GPU'

            Double | true      | [1,2]  | { x, y -> x - y } || "-7.0, -7.0, -5.0, -5.0"| "-12.0, -1.0"             | 'CPU'
            Double | true      | [2]    | { x, y -> x - y } || "-7.0, -7.0, -5.0, -5.0"| "-12.0, -1.0"             | 'CPU'
            Double | true      | [1,2]  | { x, y -> x - y } || "-7.0, -7.0, -5.0, -5.0"| "-12.0, -1.0"             | 'GPU'
            Double | true      | [2]    | { x, y -> x - y } || "-7.0, -7.0, -5.0, -5.0"| "-12.0, -1.0"             | 'GPU'

            Double | true      | [1,2]  | { x, y -> y - x } || "7.0, 7.0, 5.0, 5.0"    | "12.0, 1.0"               | 'CPU'
            Double | true      | [2]    | { x, y -> y - x } || "7.0, 7.0, 5.0, 5.0"    | "12.0, 1.0"               | 'CPU'
            Double | true      | [1,2]  | { x, y -> y - x } || "7.0, 7.0, 5.0, 5.0"    | "12.0, 1.0"               | 'GPU'
            Double | true      | [2]    | { x, y -> y - x } || "7.0, 7.0, 5.0, 5.0"    | "12.0, 1.0"               | 'GPU'
    }


    void 'A new transposed version of a given tensor will be returned by the "T()" method.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)

        when : 'A three by two matrix is being transposed...'
            Tsr t = Tsr.of([2, 3], [
                                1, 2, 3,
                                4, 5, 6
                            ]).T()

        then : t.toString().contains("[3x2]:(1.0, 4.0, 2.0, 5.0, 3.0, 6.0)")
    }


    def 'Operators "+,*,**,^" produce expected results with gradients which can be accessed via a "Ig[0]" Function instance'()
    {
        given : 'Neurekas view is set to legacy and three tensors of which one requires gradients.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Tsr x = Tsr.of(3).setRqsGradient(true)
            Tsr b = Tsr.of(-4)
            Tsr w = Tsr.of(2)

        when : Tsr y = ( (x+b)*w )**2

        then : y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        when : y = ((x+b)*w)^2
        then : y.toString().contains("[1]:(4.0); ->d[1]:(-8.0)")

        and : Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(true)

        when :
            y.backward(Tsr.of(1))
        and :
            Tsr t1 = Tsr.of( "Ig[0]", [y] )
            Tsr t2 = Tsr.of( "Ig[0]", [x] )

        then :
            t1 == null
        and :
            t2.toString() == "[1]:(-8.0)"
        and :
            t2 == x.gradient

        and : Neureka.get().settings().debug().setIsKeepingDerivativeTargetPayloads(false)

        when :
            Tsr[] trs = new Tsr[]{x}
        and :
            def fun = new FunctionBuilder( Neureka.get().backend() ).build("Ig[0]", false)
        then :
            fun(trs).toString() == "[1]:(-8.0)"

        when :
            trs[0] = y
        and :
            fun = new FunctionBuilder( Neureka.get().backend() ).build("Ig[0]", false)

        then :
            fun(trs) == null
    }


    def 'Activation functions work across types on slices and non sliced tensors.'(
            Class<?> type, String funExpression
    ) {
        given : 'We create a function based on the provided expression.'
            var func = Function.of(funExpression)
        and : 'We create 2 tensors storing the same values, one sliced and the other a normal tensor.'
            var t1 = Tsr.of(type).withShape(2, 3).andSeed("Tempeh")
            var t2 = Tsr.of(type).withShape(4, 5).all(0)[1..2, 1..3]
            t2[0..1, 0..2] = t1

        expect : 'The types of both tensors should match what was provided during instantiation.'
            t1.dataType == DataType.of(type)
            t1.valueClass == type
            t2.dataType == DataType.of(type)
            t2.valueClass == type

        when : 'We apply the function to both tensors...'
            var result1 = func(t1)
            var result2 = func(t2)
        then : 'The data of the first (non slice) tensor should be as expected.'
            result1.data == expected
        and : 'As well the value of the slice tensor (Its data would be a sparse array).'
            result2.value == expected

        where :
            type   |  funExpression || expected
            Double |   'tanh(i0)'   || [-0.2608431635405718, -0.6400224689534015, -0.15255723053856546, 0.1566537867655921, 0.5489211983894932, -0.17031712209680225] as double[]
            Float  |   'tanh(i0)'   || [-0.26084316, -0.64002246, -0.15255724, 0.15665378, 0.54892117, -0.17031713] as float[]
            Double |   'relu(i0)'   || [-0.0027019706408068795, -0.008329762613111082, -0.001543641184315801, 0.15861207834235577, 0.6567031992927272, -0.001728424711189524] as double[]
            Float  |   'relu(i0)'   || [-0.0027019705, -0.008329763, -0.0015436412, 0.15861207, 0.6567032, -0.0017284247] as float[]

    }

}
