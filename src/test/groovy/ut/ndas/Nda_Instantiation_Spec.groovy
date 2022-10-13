package ut.ndas

import neureka.Nda
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

import java.util.function.Predicate

@Title("ND-Array Instantiation")
@Narrative('''

    In this specification we cover how ND-arrays can be instantiated.
    
''')
@Subject([Nda])
class Nda_Instantiation_Spec extends Specification
{
    def 'A vector can be created from an array of values through the "of" method.'()
    {
        given : 'We simply pass an array of ints to the "of" factory method to create an nd-array.'
            var nda = Nda.of(1, 2, 3)
        expect : 'The nd-array will then have the expected shape, items and data type.'
            nda.shape == [3]
            nda.items == [1, 2, 3]
            nda.itemType == Integer

        when : 'We use doubles instead of ints...'
            nda = Nda.of(4.0d, 5.0d)
        then : '...the nd-array will be an array of doubles!'
            nda.shape == [2]
            nda.items == [4.0, 5.0]
            nda.itemType == Double

        when : 'We want to use booleans instead of numeric data types...'
            nda = Nda.of(true, false)
        then : '...the nd-array will be an array of booleans!'
            nda.shape == [2]
            nda.items == [true, false]
            nda.itemType == Boolean
    }

    def 'ND-arrays can be created fluently.'(
            Class<Object> type, Object value, Object data
    ) {
        reportInfo "This feature is based on a fluent builder API!"

        given : 'We create a new homogeneously filled Nda instance using a fluent builder API.'
            Nda<?> t = Nda.of( type )
                                 .withShape( 3, 2 )
                                 .all( value )

        expect : 'This new instance will have the expected data type!'
            t.itemType() == type

        and : '...all items of the array will have the same value, which is the one we passed to the fluent builder.'
            t.every((Predicate<Object>){ it == value })

        and : 'The nd-array will have the shape we passed to the builder.'
            t.shape == [3, 2]

        and : 'The size of the tensor will be the product of all shape entries: 2 * 3 = 6.'
            t.size == 6

        where : 'The following data is being used to populate the builder API:'
            type      | value          || data
            Integer   |  42  as int    || new int[]   { 42  }
            Double    |  4.0 as double || new double[]{ 4.0 }
            Float     |  4f  as float  || new float[] { 4f  }
            Long      |  42L as Long   || new long[]  { 42L }
            Boolean   |  false         || new boolean[] { false }
            Character | '°' as char    || new char[] { '°' as char }
    }


}
