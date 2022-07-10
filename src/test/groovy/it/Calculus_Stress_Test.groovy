package it

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.devices.Device
import neureka.devices.host.CPU
import neureka.devices.opencl.OpenCLDevice
import neureka.dtype.DataType
import neureka.view.TsrStringSettings
import spock.lang.IgnoreIf
import spock.lang.Specification

class Calculus_Stress_Test extends Specification
{
    def setup() {
        Neureka.get().reset()
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

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() && data.device == null }) // We need to assure that this system supports OpenCL!
    def 'Stress test runs error free and produces expected result'(
        Device device
    ) {
        given:
            def stress = ( Tsr t ) -> {
                t = t + Tsr.of( t.shape(), -3d..12d )
                t = t * Tsr.of( t.shape(),  2d..3d  )
                t = t / Tsr.of( t.shape(),  1d..2d  )
                t = t ^ Tsr.of( t.shape(),  2d..1d  )
                t = t - Tsr.of( t.shape(), -2d..2d  )
                return t
            }
        and :
            Tsr source = Tsr.of( [3, 3, 3, 3], -1d ).to( device )

        when :
            source[1..2, 0..2, 1..1, 0..2] = Tsr.of( [2, 3, 1, 3], -4d..2d )
            Tsr t = source[1..2, 0..2, 1..1, 0d..2d]

        then :
            t.toString() == Tsr.of( [2, 3, 1, 3], -4d..2d ).toString()

        when :
            t = stress(t)

        then :
            t.toString({it.hasSlimNumbers = true}) ==
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
            (device instanceof OpenCLDevice) || t.unsafe.data == [198.0, -6.5, 36.0, -2.5, 2.0, 6.5, 101.0, 0.0, 15.0, 4.0, 146.0, 13.0, 400.0, 17.0, 194.0, 15.5, 101.0, -4.5]
            (device instanceof OpenCLDevice) || source.unsafe.data == [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -4.0, -3.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 2.0, -4.0, -3.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, -4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -3.0, -2.0, -1.0, -1.0, -1.0, -1.0]

        where :
           device << [CPU.get(), Device.get('gpu')]
    }


    def 'Dot operation stress test runs error free and produces expected result'(
            List<Integer> shape, String expected
    ) {
        given:
            Tsr<Double> t = Tsr.of( shape, -4d..2d )

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
        given:
            Tsr<Double> t1 = Tsr.of( shape1, -4d..2d ).to( device )
            Tsr<Double> t2 = Tsr.of( shape2, -3d..5d ).to( device )

        when :
            Tsr t = Tsr.of( operation, [t1,t2] )

        then :
            t.toString() == expected

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
            var t1 = Tsr.of(type).withShape(PRIME_SIZE_1).andSeed("Tempeh")
            var t2 = Tsr.of(type).withShape(PRIME_SIZE_2).all(0)[9..7915]
            t2[0..t2.size-1] = t1

        expect : 'The types of both tensors should match what was provided during instantiation.'
            t1.dataType == DataType.of(type)
            t1.valueClass == type
            t2.dataType == DataType.of(type)
            t2.valueClass == type

        when : 'We apply the function to both tensors...'
            var result1 = func(t1)
            var result2 = func(t2)
        then : 'First we ensure that both tensors have the correct value/element type.'
            result1.valueClass == type
            result2.valueClass == type
        and : 'The underlying data object should match the data array type as is defined by the data type!'
            result1.unsafe.data.class == result1.dataType.dataArrayType()
            result2.unsafe.data.class == result2.dataType.dataArrayType()

        and : 'The data of the first non slice tensor as well as its slice should be as expected.'
            Arrays.hashCode(result1.unsafe.data) == expected[0]
            Arrays.hashCode(result2.unsafe.data) == expected[1]

        where :
            type   |  funExpression            || expected

            Double | 'gaus(i0)*100 % i0'       || [-853255121,  -853255121]
            Float  | 'gaus(i0)*100 % i0'       || [-1410818458, -1410818458]
            Integer| 'gaus(i0)*100 % i0'       || [ 566202463,   566202463]

            Double | 'tanh(i0)*100 % i0'       || [361719754, 361719754]
            Float  | 'tanh(i0)*100 % i0'       || [-1213389248, -1213389248]
            Integer| 'tanh(i0)*100 % i0'       || [-634381565,  -634381565]

            Double | 'fast_tanh(i0)*100 % i0'  || [-41932067, -41932067]
            Float  | 'fast_tanh(i0)*100 % i0'  || [1791804151, 1791804151]
            Integer| 'fast_tanh(i0)*100 % i0'  || [-634381565,  -634381565]

            Double | 'fast_gaus(i0)+i0'        || [1214521048, 1214521048]
            Float  | 'fast_gaus(i0)+i0'        || [614466683, 614466683]
            Integer| 'fast_gaus(i0)+i0'        || [-1351535318, -1351535318]

            Double | 'softsign(i0)*100 % i0'   || [-1627096169, -1627096169]
            Float  | 'softsign(i0)*100 % i0'   || [-1112662242, -1112662242]
            Integer| 'softsign(i0)*100 % i0'   || [-634381565,  -634381565]

            Double | 'random(i0)'              || [-2059799883, 276852681]
            Float  | 'random(i0)'              || [-2100773274, -590726536]
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
            var t1 = Tsr.of(type).withShape(PRIME_SIZE_1).andSeed("Seitan")
            var t2 = Tsr.of(type).withShape(PRIME_SIZE_2).all(0)[1..3]
            t2[0..t2.size-1] = t1

        expect : 'The types of both tensors should match what was provided during instantiation.'
            t1.dataType   == DataType.of(type)
            t1.valueClass == type
            t2.dataType   == DataType.of(type)
            t2.valueClass == type

        when : 'We apply the function to both tensors...'
            var result1 = ( !derive ? func(t1) : func.derive([t1], 0) )
            var result2 = ( !derive ? func(t2) : func.derive([t2], 0) )
        then : 'First we ensure that both tensors have the correct value/element type.'
            result1.valueClass == type
            result2.valueClass == type
        and : 'The underlying data object should match the data array type as is defined by the data type!'
            result1.unsafe.data.class == result1.dataType.dataArrayType()
            result2.unsafe.data.class == result2.dataType.dataArrayType()

        and : 'The data of the first non slice tensor as well as its slice should be as expected.'
            result1.value == expected
            result2.value == expected

        where :
        type   |  funExpression   | derive || expected

        Double | 'silu(i0)'       | false  || [2.1135352025452363, -0.23397468505726102, 1.0477889390530153] as double[]
        Float  | 'silu(i0)'       | false  || [2.1135352, -0.23397468, 1.047789] as float[]
        Integer| 'silu(i0)'       | false  || [0, 1766941311, 0] as int[]

        Double | 'silu(i0)'       | true   || [1.099546086132034, 0.17280584920363537, 1.0100270613327957] as double[]
        Float  | 'silu(i0)'       | true   || [1.0995461, 0.17280586, 1.010027] as float[]
        Integer| 'silu(i0)'       | true   || [0, 1, 0] as int[]

        Double | 'gelu(i0)'       | false  || [2.277196975563254, -0.16328256348702017, 1.200361889401775] as double[]
        Float  | 'gelu(i0)'       | false  || [2.2771971, -0.16328256, 1.2003618] as float[]
        Integer| 'gelu(i0)'       | false  || [0, 1766941311, 0] as int[]

        Double | 'gelu(i0)'       | true   || [1.0543057750427323, 0.016077333534953053, 1.0988343517880281] as double[]
        Float  | 'gelu(i0)'       | true   || [1.0543058, 0.01607734, 1.0988343] as float[]
        Integer| 'gelu(i0)'       | true   || [0, 1, 0] as int[]

        Double | 'selu(i0)'       | false  || [2.4387050838616458, -0.8935373895024364, 1.3932426393325066] as double[]
        Float  | 'selu(i0)'       | false  || [2.4387052, -0.8935374, 1.3932426] as float[]
        Integer| 'selu(i0)'       | false  || [-2, 1856526980, -2] as int[]

        Double | 'selu(i0)'       | true   || [1.0507009873554805, 0.8645619513449402, 1.0507009873554805] as double[]
        Float  | 'selu(i0)'       | true   || [1.050701, 0.864562, 1.050701] as float[]
        Integer| 'selu(i0)'       | true   || [0, 1, 0] as int[]

        Double | 'gatu(i0)'       | false  || [0.9999999999724318, -0.34306006539262235, 0.9813017567906377] as double[]
        Float  | 'gatu(i0)'       | false  || [1.0, -0.34306008, 0.9813018] as float[]
        Integer| 'gatu(i0)'       | false  || [-1, 1, -1] as int[]

        Double | 'gatu(i0)'       | true   || [8.91084539489384E-10, 1.3334362950250542, 0.19541950892083282] as double[]
        Float  | 'gatu(i0)'       | true   || [0.0, 1.3334364, 0.19541931] as float[]
        Integer| 'gatu(i0)'       | true   || [0, 0, 0] as int[]

        Double | 'gasu(i0)'       | false  || [0.9259465197053056, -0.2633824974845539, 0.6998385035320221] as double[]
        Float  | 'gasu(i0)'       | false  || [0.92594653, -0.2633825, 0.6998385] as float[]
        Integer| 'gasu(i0)'       | false  || [-1, 1, -1] as int[]

        Double | 'gasu(i0)'       | true   || [0.08862831427395856, 0.8200403847459871, 0.47525473491026016] as double[]
        Float  | 'gasu(i0)'       | true   || [0.0886283, 0.8200404, 0.47525474] as float[]
        Integer| 'gasu(i0)'       | true   || [0, 0, 0] as int[]

    }


}