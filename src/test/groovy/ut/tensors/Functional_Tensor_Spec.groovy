package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.common.utility.SettingsLoader
import neureka.devices.opencl.CLContext
import neureka.dtype.DataType
import neureka.view.NDPrintSettings
import spock.lang.IgnoreIf
import spock.lang.Specification

import java.util.function.Predicate
import java.util.stream.Collectors

class Functional_Tensor_Spec extends Specification
{
    def setup() {
        // The following is similar to Neureka.get().reset() however it uses a groovy script for library settings:
        SettingsLoader.tryGroovyScriptsOn(Neureka.get(), script -> new GroovyShell(getClass().getClassLoader()).evaluate(script))
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

        if ( Neureka.get().backend().has(CLContext) )
            Neureka.get().backend().get(CLContext).getSettings().autoConvertToFloat = false
    }

    def cleanup() {
        if ( Neureka.get().backend().has(CLContext) )
            Neureka.get().backend().get(CLContext).getSettings().autoConvertToFloat = true
    }

    def 'Tensor initialization lambdas produce expected tensors.'()
    {
        when : 'Instantiating a tensor using an initializer lambda...'
            Tsr t = Tsr.of(
                        DataType.of( Integer.class ),
                        [ 2, 3 ],
                        ( int i, int[] indices ) -> { i - 2 }
                    )

        then : 'The tensor has been initialized with the expected values:'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"

        when :
            t = Tsr.of(
                    DataType.of( String.class ),
                    [ 2, 3 ],
                    ( int i, int[] indices ) -> { i + ':' + indices.toString() }
                )

        then :
            t.toString() == "(2x3):[0:[0, 0], 1:[0, 1], 2:[0, 2], 3:[1, 0], 4:[1, 1], 5:[1, 2]]"
    }


    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'We can analyse the values of a tensor using various predicate receiving methods'(
            String device
    ) {
        given : 'We create 2 tensors, where one is a slice of the other.'
            var a = Tsr.ofInts().withShape(3, 2).andFill(2, 0, 1, 1, 8, 3)
            var b = a[1, 0..1]

        expect :
            !a.every((Predicate<Integer>){it == 1})
            a.any((Predicate<Integer>){it == 1})
            a.any((Predicate<Integer>){it == 8})
            !a.any((Predicate<Integer>){it == 42})
        and :
            b.every((Predicate<Integer>){it == 1})
            !b.any((Predicate<Integer>){it == 2})
            b.none((Predicate<Integer>){it == 2})
        and :
            a.count((Predicate<Integer>){it == 0}) == 1
            a.count((Predicate<Integer>){it == 1}) == 2
            b.count((Predicate<Integer>){it == 1}) == 2
            b.count((Predicate<Integer>){it == 3}) == 0
        and : 'We can also easily turn a tensor into a stream!'
            a.stream()
                    .filter({it < 3})
                    .map({(it-4)*2})
                    .collect(Collectors.toList()) == [-4, -8, -6, -6]
        where :
            device << ['CPU', 'GPU']
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Tensor mapping lambdas produce expected tensors.'(
       String device
    ) {
        when : 'Instantiating a tensor using an initializer lambda...'
            Tsr t = Tsr.of(
                    DataType.of( Double.class ),
                        [ 2, 3 ],
                        ( int i, int[] indices ) -> { (i - 2) as Double }
                    )
                    .to(device)

        then : 'The tensor has been initialized with the expected values:'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"

        when : 'We want to perform an element wise mapping to a new tensor...'
            def b = t.mapTo(String, (it) -> {"<$it>".replace(".0", "")})

        then : 'We expect the returned tensor to be a String container whose Strings are formatted according to our mapping lambda.'
            b.toString() == "(2x3):[<-2>, <-1>, <0>, <1>, <2>, <3>]"
            b.itemType == String.class
        and : 'The original tensor should not have changed because no inline operation occurred.'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"

        where :
            device << ['CPU', 'GPU']
    }

    @IgnoreIf({ !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'The "map" method is a shorter convenience method for mapping to the same type.'(
        String device
    ) {
        given : 'We create a tensor with a single element.'
            var t = Tsr.of(
                                    DataType.of( Integer.class ),
                                    [ 1 ],
                                    ( int i, int[] indices ) -> { 1 }
                                )
                                .to(device)

        when : 'We map the tensor to a new tensor of the same type.'
            var b = t.map((it) -> {it + 1})
        then : 'The new tensor should have the same value as the original tensor.'
            b.toString() == "(1):[2.0]"
            b.itemType == Integer.class
        and : 'The original tensor should not have changed because no inline operation occurred.'
            t.toString() == "(1):[1.0]"

        where :
            device << ['CPU', 'GPU']
    }

}
