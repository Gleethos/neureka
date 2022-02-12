package it.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.args.Arg
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
            Class<?> type, String expected
    ) {
        reportInfo """
            The 'x' operator performs convolution on the provided operands.
        """

        given: 'Gradient auto apply for tensors in ue is set to false.'
            Neureka.get().settings().autograd().setIsApplyingGradientWhenTensorIsUsed(false)
        and: 'Tensor legacy view is set to true.'
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
        and: 'Two new 3D tensor instances with the shapes: [2x3x1] & [1x3x2].'
            var x = Tsr.of(new int[]{2, 3, 1},
                                new double[]{
                                        3,  2, -1,
                                        -2,  2,  4
                                }
                            )
                            .unsafe.toType(type)

            var y = Tsr.of(new int[]{1, 3, 2},
                                    new double[]{
                                            4, -1,
                                            3,  2,
                                            3, -1
                                    }
                                )
                                .unsafe.toType(type)

        when : 'The x-mul result is being instantiated by passing a simple equation to the tensor constructor.'
            var z = Tsr.of("I0xi1", x, y)
        then: 'The result contains the expected String.'
            z.toString().contains(expected)

        when: 'The x-mul result is being instantiated by passing a object array containing equation parameters and syntax.'
            z = Tsr.of(new Object[]{x, "x", y})
        then: 'The result contains the expected String.'
            z.toString().contains(expected)

        where :
            type   || expected
            Double || "[2x1x2]:(15.0, 2.0, 10.0, 2.0)"
            Float  || "[2x1x2]:(15.0, 2.0, 10.0, 2.0)"
    }


    def 'The "dot" operation reshapes and produces valid "x" operation result.'( Class<?> type )
    {
        given : 'Two multi-dimensional tensors.'
            var a = Tsr.of([1, 4, 4, 1   ], 4f..12f).unsafe.toType(type)
            var b = Tsr.of([1, 3, 5, 2, 1], -5d..3d).unsafe.toType(type)

        when : 'The "dot" method is being called on "a" receiving "b"...'
            var c = a.convDot(b)

        then : 'The result tensor contains the expected shape.'
            c.toString().contains("(4x2x5x2)")
        and :
            c.valueClass == type

        where :
            type << [Double, Float]
    }


    def 'The "matMul" operation produces the expected result.'(
            Class<?> type, Double[] A, Double[] B, int M, int K, int N, double[] expectedC
    ) {
        given : 'Two 2-dimensional tensors.'
            var a = Tsr.of(Double.class).withShape(M, K).andFill(A).unsafe.toType(type)
            var b = Tsr.of(Double.class).withShape(K, N).andFill(B).unsafe.toType(type)

        when : 'The "matMul" method is being called on "a" receiving "b"...'
            var c = a.matMul(b)

        then : 'The result tensor contains the expected shape and values.'
            c.toString() == "(${M}x${N}):$expectedC"
        and :
            c.valueClass == type

        where : 'We use the following data and matrix dimensions!'
            type   | A            | B                  | M | K | N || expectedC
            Double | [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 1, 0, 0, 1 ]
            Double | [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 0.5 ]
            Double | [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2.0, 3.0, -1.0, -1.5 ]
            Float  | [4, 3, 2, 1] | [-0.5, 1.5, 1, -2] | 2 | 2 | 2 || [ 1, 0, 0, 1 ]
            Float  | [-2, 1]      | [-1, -1.5]         | 1 | 2 | 1 || [ 0.5 ]
            Float  | [-2, 1]      | [-1, -1.5]         | 2 | 1 | 2 || [ 2.0, 3.0, -1.0, -1.5 ]
    }

    def 'The "random" function populates tensors randomly.'() {

        given :
            var t = Tsr.of(Double).withShape(2,4).all(-42)
        and :
            var f = Function.of('random(I[0])')

        when :
            var r = f(t)

        then :
            r === t
        and :
            r.data == [1.08932458081836, 1.2280912763651217, -0.8432688409559622, 0.16634425538710282, -0.3162408914866152, -0.20064937375580177, 0.5205000859982427, 2.8909977024398703]

        when :
            r = f.callWith(Arg.Seed.of(42)).call(t)

        then :
            r === t
        and :
            r.data == [0.5162831402652289, 0.7981311275445736, -0.9077652533826679, -0.09080253321088107, -0.5486405426203189, 1.7498437096821327, -0.20705593029533073, 0.14433135828802282]
    }

    def 'New method "asFunction" of String added at runtime is callable by groovy and also works.'(
            Class<?> type, String code, String expected
    ) {
        given : 'We create two tensors and convert them to a desired type.'
            var a = Tsr.of([1,2], [3d, 2d]).unsafe.toType(type)
            var b = Tsr.of([2,1], [-1f, 4f]).unsafe.toType(type)
        and : 'We prepare bindings for the Groovy shell.'
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        expect : 'The tensors have the type...'
            a.valueClass == type
            b.valueClass == type

        when : 'The groovy code is being evaluated.'
            var c = new GroovyShell(binding).evaluate((code)) as Tsr

        then : 'The resulting tensor (toString) will contain the expected String.'
            c.toString().contains(expected)
        and :
            c.valueClass == type

        where :
            type   | code                               || expected
            Double | '"I[0]xI[1]".asFunction()([a, b])' || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            Double | '"I[0]xI[1]"[a, b]'                || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            Double | '"i0 x i1"%[a, b]'                 || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            Double | '"i0"%a'                           || "(1x2):[3.0, 2.0]"
            Float  | '"I[0]xI[1]".asFunction()([a, b])' || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            Float  | '"I[0]xI[1]"[a, b]'                || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            Float  | '"i0 x i1"%[a, b]'                 || "(2x2):[-3.0, -2.0, 12.0, 8.0]"
            Float  | '"i0"%a'                           || "(1x2):[3.0, 2.0]"
    }

    def 'New operator methods added to "SDK-types" at runtime are callable by groovy and also work.'(
            Class<?> type, String code, String expected
    ) {
        given :
            Neureka.get().settings().view().tensors({it.hasSlimNumbers=true})
            Tsr a = Tsr.of(5d).unsafe.toType(type)
            Tsr b = Tsr.of(3f).unsafe.toType(type)
            Binding binding = new Binding()
            binding.setVariable('a', a)
            binding.setVariable('b', b)

        when : '...calling methods on types like Double and Integer that receive Tsr instances...'
            Tsr c = new GroovyShell(binding).evaluate((code)) as Tsr

        then : 'The resulting tensor (toString) will contain the expected String.'
            c.toString().endsWith("[$expected]")

        where :
            type   | code       || expected
            Double | '(2+a)'    || "7"
            Double | '(2*b)'    || "6"
            Double | '(6/b)'    || "2"
            Double | '(2^b)'    || "8"
            Double | '(2**b)'   || "8"
            Double | '(4-a)'    || "-1"
            Double | '(2.0+a)'  || "7"
            Double | '(2.0*b)'  || "6"
            Double | '(6.0/b)'  || "2"
            Double | '(2.0^b)'  || "8"
            Double | '(2.0**b)' || "8"
            Double | '(4.0-a)'  || "-1"
            Float  | '(2+a)'    || "7"
            Float  | '(2*b)'    || "6"
            Float  | '(6/b)'    || "2"
            Float  | '(2^b)'    || "8"
            Float  | '(2**b)'   || "8"
            Float  | '(4-a)'    || "-1"
            Float  | '(2.0+a)'  || "7"
            Float  | '(2.0*b)'  || "6"
            Float  | '(6.0/b)'  || "2"
            Float  | '(2.0^b)'  || "8"
            Float  | '(2.0**b)' || "8"
            Float  | '(4.0-a)'  || "-1"

    }

    def 'Overloaded operation methods on tensors produce expected results when called.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)
            Tsr a = Tsr.of(2d).setRqsGradient(true)
            Tsr b = Tsr.of(-4d)
            Tsr c = Tsr.of(3d).setRqsGradient(true)

        expect :
            ( a / a                     ).toString().contains("[1]:(1.0)")
            ( c % a                     ).toString().contains("[1]:(1.0)")
            ( ( ( b / b ) ^ c % a ) * 3 ).toString().contains("[1]:(3.0)")
            ( a *= b                    ).toString().contains("(-8.0)")
            ( a += -c                   ).toString().contains("(-11.0)")
            ( a -= c                    ).toString().contains("(-14.0)")
            ( a /= Tsr.of(2d)     ).toString().contains("(-7.0)")
            ( a %= c                    ).toString().contains("(-1.0)")
    }

    def 'Manual convolution produces expected result.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
            Tsr a = Tsr.of([100, 100], 3d..19d)
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
            Tsr a = Tsr.of([4, 4], 0d..16d).to( device )

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
            Tsr a = Tsr.of([11, 11], 3d..19d).to( device )
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
                                ? "8" + ( bShape.inject(1, {x,y->x*y}) > 1 ? ", 9" : "" )
                                : "1, 2, 3, 4"
        and :
            def aShape = [2, 2]
        and :
            Tsr<Double> a = Tsr.of(aShape, 1d..5d).setRqsGradient(!whichGrad).to(Device.find(device))
            Tsr<Double> b = Tsr.of(bShape, 8d..9d).setRqsGradient(whichGrad).to(Device.find(device))
        and :
            a.unsafe.toType(type)
            b.unsafe.toType(type)
        and :
            String wShape = ( whichGrad ? bShape : aShape ).join("x")
            Tsr    w      = ( whichGrad ? b      : a      )

        expect :
            a.valueClass == type
            b.valueClass == type

        when :
            Tsr c = operation.apply(a, b)
        then :
            c.toString({it.hasSlimNumbers = true}).startsWith("[2x2]:($cValue)")
            w.toString({it.hasSlimNumbers = true}) == "[$wShape]:($wValue):g:(null)"

        when :
            c.backward(Tsr.of([2, 2], [5, -2, 7, 3]).unsafe.toType(type))
        then :
            w.toString({it.hasSlimNumbers = true}) == "[$wShape]:($wValue):g:($wGradient)"

        when :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(false)
        then :
            c.toString({it.hasSlimNumbers = true}) == "(2x2):[$cValue]"

        where:
            device | type   | whichGrad | bShape |    operation      ||     cValue      | wGradient

            'CPU'  | Double | false     | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "5, -2, 7, 3"
            'GPU'  | Double | false     | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "5, -2, 7, 3"
            'CPU'  | Float  | false     | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "5, -2, 7, 3"
            'GPU'  | Float  | false     | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "5, -2, 7, 3"

            'CPU'  | Double | false     | [1]    | { x, y -> x * y } || "8, 16, 24, 32" | "40, -16, 56, 24"
            'GPU'  | Double | false     | [1]    | { x, y -> x * y } || "8, 16, 24, 32" | "40, -16, 56, 24"
            'CPU'  | Float  | false     | [1]    | { x, y -> x * y } || "8, 16, 24, 32" | "40, -16, 56, 24"
            'GPU'  | Float  | false     | [1]    | { x, y -> x * y } || "8, 16, 24, 32" | "40, -16, 56, 24"

            'CPU'  | Double | true      | [2,1]  | { x, y -> x + y } || "9, 10, 12, 13" | "3, 10"
            //'GPU'  | Double |  true     | [2,1]  | { x, y -> x + y } || "9, 10, 12, 13" | "3, 10"
            'CPU'  | Float  | true      | [2,1]  | { x, y -> x + y } || "9, 10, 12, 13" | "3, 10"
            //'GPU'  | Float  |  true     | [2,1]  | { x, y -> x + y } || "9, 10, 12, 13" | "3, 10"

            'CPU'  | Double | true      | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "13"
            'GPU'  | Double | true      | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "13"
            'CPU'  | Float  | true      | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "13"
            'GPU'  | Float  | true      | [1]    | { x, y -> x + y } || "9, 10, 11, 12" | "13"

            'CPU'  | Double | true      | [1,2]  | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"
            'GPU'  | Double | true      | [1,2]  | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"
            'CPU'  | Float  | true      | [1,2]  | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"
            'GPU'  | Float  | true      | [1,2]  | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"

            'CPU'  | Double | true      | [2]    | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"
            'GPU'  | Double | true      | [2]    | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"
            'CPU'  | Float  | true      | [2]    | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"
            'GPU'  | Float  | true      | [2]    | { x, y -> x + y } || "9, 11, 11, 13" | "12, 1"

            'CPU'  | Double | true      | [1,2]  | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"
            'GPU'  | Double | true      | [1,2]  | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"
            'CPU'  | Float  | true      | [1,2]  | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"
            'GPU'  | Float  | true      | [1,2]  | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"

            'CPU'  | Double | true      | [2]    | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"
            'GPU'  | Double | true      | [2]    | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"
            'CPU'  | Float  | true      | [2]    | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"
            'GPU'  | Float  | true      | [2]    | { x, y -> x - y } || "-7, -7, -5, -5"| "-12, -1"

            'CPU'  | Double | true      | [1,2]  | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
            'GPU'  | Double | true      | [1,2]  | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
            'CPU'  | Float  | true      | [1,2]  | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
            'GPU'  | Float  | true      | [1,2]  | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"

            'CPU'  | Double | true      | [2]    | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
            'GPU'  | Double | true      | [2]    | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
            'CPU'  | Float  | true      | [2]    | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
            'GPU'  | Float  | true      | [2]    | { x, y -> y - x } || "7, 7, 5, 5"    | "12, 1"
    }


    void 'A new transposed version of a given tensor will be returned by the "T()" method.'()
    {
        given :
            Neureka.get().settings().view().getTensorSettings().setIsLegacy(true)

        when : 'A three by two matrix is being transposed...'
            Tsr t = Tsr.of([2, 3], [
                                1d, 2d, 3d,
                                4d, 5d, 6d
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
            Tsr t2 = Tsr.of( "Ig[0]", [x] )
            Tsr t1 = Tsr.of( "Ig[0]", [y] ) // The input does not have a gradient!

        then :
            thrown(IllegalArgumentException)
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
        and :
            fun(trs)

        then :
            thrown(IllegalArgumentException)
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
        then :
            result1.valueClass == type
            result2.valueClass == type

        and : 'The data of the first (non slice) tensor should be as expected.'
            result1.data == expected
        and : 'As well the value of the slice tensor (Its data would be a sparse array).'
            result2.value == expected

        where :
            type   |  funExpression     || expected

            Double | 'tanh(i0)'         || [-0.2608431635405718, -0.6400224689534015, -0.15255723053856546, 0.1566537867655921, 0.5489211983894932, -0.17031712209680225] as double[]
            Float  | 'tanh(i0)'         || [-0.26084316, -0.64002246, -0.15255724, 0.15665378, 0.54892117, -0.17031713] as float[]
            Integer| 'tanh(i0)'         || [-1, -1, 1, -1, 1, -1] as int[]

            Double | 'relu(i0)'         || [-0.0027019706408068795, -0.008329762613111082, -0.001543641184315801, 0.15861207834235577, 0.6567031992927272, -0.001728424711189524] as double[]
            Float  | 'relu(i0)'         || [-0.0027019705, -0.008329763, -0.0015436412, 0.15861207, 0.6567032, -0.0017284247] as float[]
            Integer| 'relu(i0)'         || [-7156386, -18495716, 248181051, -13634228, 919305478, -15169971] as int[]

            Double | 'relu(i0*i0)'      || [0.07300645343782339, 0.6938494519078316, 0.023828281059158886, 0.025157791396081604, 0.43125909196130346, 0.029874519822505895] as double[]
            Float  | 'relu(i0*i0)'      || [0.07300645, 0.6938495, 0.023828283, 0.025157789, 0.43125907, 0.02987452] as float[]
            Integer| 'relu(i0*i0)'      || [988699588, -17870520, 141304729, 1260971300, 210951204, 1018550276] as int[]

            Double | 'relu(i0-i0)'      || [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] as double[]
            Float  | 'relu(i0-i0)'      || [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] as float[]
            Integer| 'relu(i0-i0)'      || [0, 0, 0, 0, 0, 0] as int[]

            Double | 'relu(i0)-i0'      || [0.26749509343988104, 0.8246464986979971, 0.1528204772472643, 0.0, 0.0, 0.17111404640776287] as double[]
            Float  | 'relu(i0)-i0'      || [0.2674951, 0.82464653, 0.15282048, 0.0, 0.0, 0.17111404] as float[]
            Integer| 'relu(i0)-i0'      || [708482220, 1831075919, 0, 1349788550, 0, 1501827147] as int[]

            Double | 'relu(-i0)+i0'     || [0.0, 0.0, 0.0, 0.15702595755893223, 0.6501361672998, 0.0] as double[]
            Float  | 'relu(-i0)+i0'     || [0.0, 0.0, 0.0, 0.15702595174312592, 0.650136142373085, 0.0] as float[]
            Integer| 'relu(-i0)+i0'     || [0, 0, 245699240, 0, 910112423, 0] as int[]

            Double | 'relu(-i0)/i0'     || [-1.0, -1.0, -1.0, -0.01, -0.01, -1.0] as double[]
            Float  | 'relu(-i0)/i0'     || [-1.0, -1.0, -1.0, -0.01, -0.01, -1.0] as float[]
            Integer| 'relu(-i0)/i0'     || [-1, -1, 0, -1, 0, -1] as int[]

            Double | 'relu(-i0-5)+i0*3' || [-0.857889221601257, -2.540599021320214, -0.5115487141104245, 0.42425011424364373, 1.9135425658852545, -0.5667989886456676] as double[]
            Float  | 'relu(-i0-5)+i0*3' || [-0.85788924, -2.540599, -0.51154876, 0.4242501, 1.9135424, -0.566799] as float[]
            Integer| 'relu(-i0-5)+i0*3' || [-1431277217, 595824021, 742061342, 1568121735, -1546243917, 1260973055] as int[]

            Double | 'abs(i0*10)%3'     || [2.7019706408068793, 2.3297626131110825, 1.5436411843158009, 1.5861207834235578, 0.5670319929272729, 1.7284247111895241] as double[]
            Float  | 'abs(i0*10)%3'     || [2.7019706, 2.3297625, 1.5436412, 1.5861207, 0.56703186, 1.7284248] as float[]
            Integer| 'abs(i0*10)%3'     || [2, 0, 1, 1, 2, 1] as int[]

            Double | 'gaus(i0)*100%i0'  || [0.011693048642643422, 0.8192993419907051, 0.08721424132459399, 0.1277867634692304, 0.6121424058303924, 0.09210498892686086] as double[]
            Float  | 'gaus(i0)*100%i0'  || [0.011690378, 0.81929654, 0.087213635, 0.12778962, 0.6121441, 0.09210494] as float[]
            Integer| 'gaus(i0)*100%i0'  || [0, 0, 0, 0, 0, 0] as int[]

    }

}
