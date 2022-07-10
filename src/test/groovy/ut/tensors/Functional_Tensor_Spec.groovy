package ut.tensors

import neureka.Tsr
import neureka.dtype.DataType
import spock.lang.Specification

import java.util.function.Predicate
import java.util.stream.Collectors

class Functional_Tensor_Spec extends Specification
{

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


    def 'We can analyse the values of a tensor using various predicate receiving methods'()
    {
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
    }


    def 'Tensor mapping lambdas produce expected tensors.'()
    {
        when : 'Instantiating a tensor using an initializer lambda...'
            Tsr t = Tsr.of(
                    DataType.of( Double.class ),
                        [ 2, 3 ],
                        ( int i, int[] indices ) -> { (i - 2) as Double }
                    )

        then : 'The tensor has been initialized with the expected values:'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"

        when : 'We want to perform an element wise mapping to a new tensor...'
            def b = t.mapTo(String, (it) -> {"<$it>".replace(".0", "")})

        then : 'We expect the returned tensor to be a String container whose Strings are formatted according to our mapping lambda.'
            b.toString() == "(2x3):[<-2>, <-1>, <0>, <1>, <2>, <3>]"
            b.valueClass == String.class
        and : 'The original tensor should not have changed because no inline operation occurred.'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"
    }


}