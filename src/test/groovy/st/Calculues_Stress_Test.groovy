package st

import neureka.Neureka
import neureka.Tsr
import spock.lang.Specification

class Calculues_Stress_Test extends Specification
{


    def 'Stress test runs error free and produces expected result'(
        boolean arrayIndexing, boolean legacyIndexing
    ) {
        given: 'Neureka is being reset.'
            Neureka.instance().reset()
        and:
            Neureka.instance().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing
            Neureka.instance().settings().indexing().isUsingLegacyIndexing = legacyIndexing
        and :
            def stress = ( Tsr t ) -> {
                t = t + new Tsr( t.shape(), -3..12 )
                t = t * new Tsr( t.shape(), 2..3 )
                t = t / new Tsr( t.shape(), 1..2 )
                t = t ^ new Tsr( t.shape(), 2..1 )
                t = t - new Tsr( t.shape(), -2..2 )
                return t
            }
        and :
            Tsr t = new Tsr( [2, 3, 1, 3], -4..2 )

        when :
            t = stress(t)
            def x = new Tsr( [2, 1, 3], -4..2 )
            println(x.dot(x.T()))

        then :
            t.toString() == "(2x3x1x3):[198.0, -6.5, 36.0, -2.5, 2.0, 6.5, 101.0, 0.0, 15.0, 4.0, 146.0, 13.0, 400.0, 17.0, 194.0, 15.5, 101.0, -4.5]"

        where :
            arrayIndexing | legacyIndexing
            true          | true
            true          | false
            false         | true
            false         | false
    }


    def 'Dot operation stress test runs error free and produces expected result'(
            boolean arrayIndexing, boolean legacyIndexing, List<Integer> shape, String expected
    ) {
        given: 'Neureka is being reset.'
            Neureka.instance().reset()
        and:
            Neureka.instance().settings().indexing().isUsingArrayBasedIndexing = arrayIndexing
            Neureka.instance().settings().indexing().isUsingLegacyIndexing = legacyIndexing

        and :
            Tsr t = new Tsr( shape, -4..2 )

        when :
            t = t.dot(t.T())

        then :
            t.toString() == expected

        where :
        arrayIndexing | legacyIndexing | shape        || expected
        true          | true           | [2, 3]       || "(2x1x2):[20.0, 14.0, 14.0, 11.0]"
        true          | false          | [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
        false         | true           | [2, 3]       || "(2x1x2):[20.0, 14.0, 14.0, 11.0]"
        false         | false          | [2, 3]       || "(2x1x2):[29.0, 2.0, 2.0, 2.0]"
        true          | true           | [2, 1, 3]    || "(2x1x1x1x2):[20.0, 14.0, 14.0, 11.0]"
        true          | false          | [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
        false         | true           | [2, 1, 3]    || "(2x1x1x1x2):[20.0, 14.0, 14.0, 11.0]"
        false         | false          | [2, 1, 3]    || "(2x1x1x1x2):[29.0, 2.0, 2.0, 2.0]"
    }



}
