package it

import neureka.Neureka
import neureka.Tensor
import neureka.backend.ocl.CLBackend
import neureka.math.Function
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.dtype.DataType
import neureka.view.NDPrintSettings
import spock.lang.IgnoreIf
import spock.lang.Specification

class Calculus_Stress_Test extends Specification
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

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null }) // We need to assure that this system supports OpenCL!
    def 'Stress test runs error free and produces expected result'(
        Device device
    ) {
        given : 'For this test we tell the CL-Backend to auto-convert to floats.'
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = true }
        and:
            def stress = (Tensor t ) -> {
                t = t + Tensor.of( t.shape(), -3d..12d )
                t = t * Tensor.of( t.shape(),  2d..3d  )
                t = t / Tensor.of( t.shape(),  1d..2d  )
                t = t **Tensor.of( t.shape(),  2d..1d  )
                t = t - Tensor.of( t.shape(), -2d..2d  )
                return t
            }
        and :
            Tensor source = Tensor.of( [3, 3, 3, 3], -1d ).to( device )

        when :
            source.mut[1..2, 0..2, 1..1, 0..2] = Tensor.of( [2, 3, 1, 3], -4d..2d )
            Tensor s = source[1..2, 0..2, 1..1, 0d..2d]

        then :
            s.toString() == Tensor.of( [2, 3, 1, 3], -4d..2d ).toString()

        when :
            s = stress(s)

        then :
            s.toString({it.hasSlimNumbers = true}) ==
                    "(2x3x1x3):[" +
                        "198, -6.5, " +
                        "36, -2.5, " +
                        "2, 6.5, " +
                        "" +
                        "101, 0, " +
                        "15, 4, " +
                        "146, 13, " +
                        "" +
                        "400, 17, " +
                        "194, 15.5, " +
                        "101, -4.5" +
                    "]"
        and :
            (device instanceof OpenCLDevice) || s.mut.data.get() == [198.0, -6.5, 36.0, -2.5, 2.0, 6.5, 101.0, 0.0, 15.0, 4.0, 146.0, 13.0, 400.0, 17.0, 194.0, 15.5, 101.0, -4.5]
            (device instanceof OpenCLDevice) || source.mut.data.get() == [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -4.0, -3.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, -4.0, -3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -3.0, -2.0, -1.0, -1.0, -1.0, -1.0]

        cleanup :
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = false }

        where :
           device << [CPU.get(), Device.get('gpu')]
    }


    def 'Dot operation stress test runs error free and produces expected result'(
            List<Integer> shape, String expected
    ) {
        given:
            Tensor<Double> t = Tensor.of( shape, -4d..2d )

        when :
            t = t.convDot( t.T() )

        then :
            t.toString() == expected

        where :
            shape        || expected
            [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
            [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
            [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
            [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null }) // We need to assure that this system supports OpenCL!
    def 'The broadcast operation stress test runs error free and produces expected result'(
            Device device,
            List<Integer> shape1, List<Integer> shape2,
            String operation,
            String expected
    ) {
        given : 'For this test we tell the CL-Backend to auto-convert to floats.'
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = true }
        and :
            Tensor<Double> t1 = Tensor.of( shape1, -4d..2d ).to( device )
            Tensor<Double> t2 = Tensor.of( shape2, -3d..5d ).to( device )

        when :
            Tensor t = Tensor.of( operation, [t1, t2] )

        then :
            t.toString() == expected

        cleanup :
            Neureka.get().backend.find(CLBackend).ifPresent { it.settings.autoConvertToFloat = false }

        where :
            device             | shape1    | shape2    | operation || expected
            CPU.get()          | [2, 1]    | [2, 2]    | 'i0%i1' || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0%i1' || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"

            //Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0%i1'   || "(2x2):[-1.0, -0.0, -0.0, NaN]"
            //Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0%i1'   || "(2x3x2):[-1.0, -0.0, -0.0, NaN, -0.0, -0.0, -1.0, -1.0, 0.0, NaN, 0.0, 1.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0*i1' || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0*i1' || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"

            Device.get('gpu')  | [2, 1]    | [2, 2]    | 'i0*i1' || "(2x2):[12.0, 8.0, 3.0, -0.0]"
            Device.get('gpu')  | [2, 3, 1] | [1, 3, 2] | 'i0*i1' || "(2x3x2):[12.0, 8.0, 3.0, -0.0, -2.0, -4.0, 3.0, 2.0, -0.0, 0.0, 1.0, 2.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0+i1' || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0+i1' || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            Device.get('gpu')  | [2, 1]    | [2, 2]    | 'i0+i1' || "(2x2):[-7.0, -6.0, -4.0, -3.0]"
            Device.get('gpu')  | [2, 3, 1] | [1, 3, 2] | 'i0+i1' || "(2x3x2):[-7.0, -6.0, -4.0, -3.0, -1.0, 0.0, -4.0, -3.0, -1.0, 0.0, 2.0, 3.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0-i1' || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0-i1' || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"

            Device.get('gpu')  | [2, 1]    | [2, 2]    | 'i0-i1' || "(2x2):[-1.0, -2.0, -2.0, -3.0]"
            Device.get('gpu')  | [2, 3, 1] | [1, 3, 2] | 'i0-i1' || "(2x3x2):[-1.0, -2.0, -2.0, -3.0, -3.0, -4.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0]"

            CPU.get()          | [2, 1]    | [2, 2]    | 'i0/i1' || "(2x2):[1.33333, 2.0, 3.0, -∞]"
            CPU.get()          | [2, 3, 1] | [1, 3, 2] | 'i0/i1' || "(2x3x2):[1.33333, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333, 0.5, -0.0, NaN, 1.0, 0.5]"

            //WIP: fix derivative! -> Make multiple kernels!
            //Device.find('gpu') | [2, 1]    | [2, 2]    | 'i0/i1'   || "(2x2):[1.33333E0, 2.0, 3.0, -∞]"
            //Device.find('gpu') | [2, 3, 1] | [1, 3, 2] | 'i0/i1'   || "(2x3x2):[1.33333E0, 2.0, 3.0, -∞, -2.0, -1.0, 0.33333E0, 0.5, -0.0, NaN, 1.0, 0.5]"
    }

    def 'Activation functions work across types, on large prime sized 1D slices and non sliced 1D tensors.'(
            Class<?> type, String funExpression
    ) {
        given : 'We create a function based on the provided expression.'
            var func = Function.of(funExpression)
        and : 'We use a large prime number to size our tensors in order to stress workload divisibility.'
            var PRIME_SIZE_1 = 7907
            var PRIME_SIZE_2 = 7919
        and : 'We create 2 tensors storing the same values, one sliced and the other a normal tensor.'
            var t1 = Tensor.of(type).withShape(PRIME_SIZE_1).andSeed("Tempeh")
            var t2 = Tensor.of(type).withShape(PRIME_SIZE_2).all(0)[9..7915]
            t2.mut[0..t2.size-1] = t1

        expect : 'The types of both tensors should match what was provided during instantiation.'
            t1.dataType == DataType.of(type)
            t1.itemType == type
            t2.dataType == DataType.of(type)
            t2.itemType == type

        when : 'We apply the function to both tensors...'
            var result1 = func(t1)
            var result2 = func(t2)
            var data1 = result1.mut.data.get()
            var data2 = result2.mut.data.get()

        then : 'First we ensure that both tensors have the correct value/element type.'
            result1.itemType == type
            result2.itemType == type
        and : 'The underlying data object should match the data array type as is defined by the data type!'
            data1.class == result1.dataType.dataArrayType()
            data2.class == result2.dataType.dataArrayType()

        and : 'The data of the first non slice tensor as well as its slice should be as expected.'
            data1.collect({(it as BigDecimal).round(3)}).sum().round(3) == expected[0]
            data2.collect({(it as BigDecimal).round(3)}).sum().round(3) == expected[1]

        where :
            type   |  funExpression            || expected

            Double | 'gaus(i0)*100 % i0'       || [2840.044, 2840.044]
            Float  | 'gaus(i0)*100 % i0'       || [2840.043, 2840.043]
            Integer| 'gaus(i0)*100 % i0'       || [0.000, 0.000]

            Double | 'tanh(i0)*100 % i0'       || [625.914, 625.914]
            Float  | 'tanh(i0)*100 % i0'       || [625.910, 625.910]
            Integer| 'tanh(i0)*100 % i0'       || [-20900.000, -20900.000]

            Double | 'fast_tanh(i0)*100 % i0'  || [637.151, 637.151]
            Float  | 'fast_tanh(i0)*100 % i0'  || [637.011, 637.011]
            Integer| 'fast_tanh(i0)*100 % i0'  || [-20900.000, -20900.000]

            Double | 'fast_gaus(i0)+i0'        || [6632.311, 6632.311]
            Float  | 'fast_gaus(i0)+i0'        || [6632.311, 6632.311]
            Integer| 'fast_gaus(i0)+i0'        || [-233722189118.000, -233722189118.000]

            Double | 'softsign(i0)*100 % i0'   || [647.562, 647.562]
            Float  | 'softsign(i0)*100 % i0'   || [647.564, 647.564]
            Integer| 'softsign(i0)*100 % i0'   || [-20900.000, -20900.000]

            Double | 'random(i0)'              || [93.662, 93.662]
            Float  | 'random(i0)'              || [93.662, 93.662]
    }

    def 'Activation functions work across types.'(
            Class<?> type, String funExpression, boolean derive, Object expected
    ) {
        given : 'We create a function based on the provided expression.'
            var func = Function.of(funExpression)
        and : 'We use a large prime number to size our tensors in order to stress workload divisibility.'
            var PRIME_SIZE_1 = 3
            var PRIME_SIZE_2 = 5
        and : 'We create 2 tensors storing the same values, one sliced and the other a normal tensor.'
            var t1 = Tensor.of(type).withShape(PRIME_SIZE_1).andSeed("Seitan")
            var t2 = Tensor.of(type).withShape(PRIME_SIZE_2).all(0)[1..3]
            t2[0..t2.size-1] = t1

        expect : 'The types of both tensors should match what was provided during instantiation.'
            t1.dataType   == DataType.of(type)
            t1.itemType == type
            t2.dataType   == DataType.of(type)
            t2.itemType == type

        when : 'We apply the function to both tensors...'
            var result1 = ( !derive ? func(t1) : func.derive([t1], 0) )
            var result2 = ( !derive ? func(t2) : func.derive([t2], 0) )
        then : 'First we ensure that both tensors have the correct value/element type.'
            result1.itemType == type
            result2.itemType == type
        and : 'The underlying data object should match the data array type as is defined by the data type!'
            result1.mut.data.get().class == result1.dataType.dataArrayType()
            result2.mut.data.get().class == result2.dataType.dataArrayType()

        and : 'The data of the first non slice tensor as well as its slice should be as expected.'
            result1.items == expected
            result2.items == expected

        where :
        type   |  funExpression   | derive || expected

        Double | 'silu(i0)'       | false  || [1.0985150624263118, 2.331551300795844, 0.08745752408303246] as double[]
        Float  | 'silu(i0)'       | false  || [1.098515, 2.3315513, 0.08745752] as float[]
        Integer| 'silu(i0)'       | false  || [2124371342, 0, 0] as int[]

        Double | 'silu(i0)'       | true   || [1.0198659569612678, 1.0992238228008295, 0.5805713104936336] as double[]
        Float  | 'silu(i0)'       | true   || [1.019866, 1.0992239, 0.5805713] as float[]
        Integer| 'silu(i0)'       | true   || [1, 0, 0] as int[]

        Double | 'gelu(i0)'       | false  || [1.2553019101258691, 2.48514859714065, 0.09199892841280806] as double[]
        Float  | 'gelu(i0)'       | false  || [1.255302, 2.4851487, 0.09199893] as float[]
        Integer| 'gelu(i0)'       | false  || [2124371342, 0, 0] as int[]

        Double | 'gelu(i0)'       | true   || [1.09968352899801, 1.043758795430269, 0.6360091016582581] as double[]
        Float  | 'gelu(i0)'       | true   || [1.0996835, 1.0437589, 0.6360091] as float[]
        Integer| 'gelu(i0)'       | true   || [1, 0, 0] as int[]

        Double | 'selu(i0)'       | false  || [1.4457526798842053, 2.6470118580557593, 0.17005220305511268] as double[]
        Float  | 'selu(i0)'       | false  || [1.4457527, 2.647012, 0.1700522] as float[]
        Integer| 'selu(i0)'       | false  || [-2062888229, -2, -2] as int[]

        Double | 'selu(i0)'       | true   || [1.0507009873554805, 1.0507009873554805, 1.0507009873554805] as double[]
        Float  | 'selu(i0)'       | true   || [1.050701, 1.050701, 1.050701] as float[]
        Integer| 'selu(i0)'       | true   || [1, 0, 0] as int[]

        Double | 'gatu(i0)'       | false  || [0.9891407665275838, 0.9999999999999742, 0.004239423130809827] as double[]
        Float  | 'gatu(i0)'       | false  || [0.98914075, 1.0, 0.0042394227] as float[]
        Integer| 'gatu(i0)'       | false  || [1, -1, -1] as int[]

        Double | 'gatu(i0)'       | true   || [0.1226918386004856, 9.805489753489383E-13, 0.07858138767615172] as double[]
        Float  | 'gatu(i0)'       | true   || [0.12269211, 0.0, 0.078581385] as float[]
        Integer| 'gatu(i0)'       | true   || [0, 0, 0] as int[]

        Double | 'gasu(i0)'       | false  || [0.7226245060456667, 0.9411395236107959, 0.004221551478848414] as double[]
        Float  | 'gasu(i0)'       | false  || [0.72262454, 0.9411396, 0.004221551] as float[]
        Integer| 'gasu(i0)'       | false  || [1, -1, -1] as int[]

        Double | 'gasu(i0)'       | true   || [0.4370057619908791, 0.06596632547000601, 0.07792071781374522] as double[]
        Float  | 'gasu(i0)'       | true   || [0.43700573, 0.06596632, 0.07792072] as float[]
        Integer| 'gasu(i0)'       | true   || [0, 0, 0] as int[]

    }


}
