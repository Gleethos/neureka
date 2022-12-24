package ut.tensors

import neureka.Nda
import neureka.Neureka
import neureka.common.utility.SettingsLoader
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

import java.util.function.Predicate
import java.util.stream.Collectors
import java.util.stream.Stream

@Title("Functional ND-Arrays")
@Narrative('''

    ND-Arrays expose a powerful API for performing operations on them
    in a functional style.

''')
class Functional_Nda_Spec extends Specification
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
    }

    def 'We can initialize an ND-Array using a filler lambda mapping indices to items.'()
    {
        when : 'We instantiate the ND-Array using an initializer lambda which explains the indices of each item.'
            var t = Nda.of(String).withShape(2, 3).andWhere(( int i, int[] indices ) -> { i + ':' + indices.toString() })

        then : 'The ND-Array will have the expected items:'
            t.items == ["0:[0, 0]", "1:[0, 1]", "2:[0, 2]", "3:[1, 0]", "4:[1, 1]", "5:[1, 2]"]
        and : 'We can also recognise them when printed as string:'
            t.toString() == "(2x3):[0:[0, 0], 1:[0, 1], 2:[0, 2], 3:[1, 0], 4:[1, 1], 5:[1, 2]]"
    }


    def 'We can find both min and max items in an ND-array by providing a comparator.'()
    {
        given : 'We create an ND-array of strings for which we want to find a min and max value:'
            var n = Nda.of(String)
                                .withShape(2, 3)
                                .andWhere(( int i, int[] indices ) -> { "a" + i } )

        when : 'We find the min value by passing a comparator (which takes 2 values and returns an int):'
            var min = n.minItem( ( String a, String b ) -> a.compareTo( b ) )

        then : 'The resulting min value is the last item in the array:'
            min == "a0"

        when : 'We now try to find the max value:'
            var max = n.maxItem( ( String a, String b ) -> a.compareTo( b ) )

        then : 'The max value is the last item in the array:'
            max == "a5"
    }


    def 'We can analyse the values of a nd-array using various predicate receiving methods'() {
        given : 'We create 2 tensors, where one is a slice of the other.'
            var a = Nda.of(Integer).withShape(3, 2).andFill(2, 0, 1, 1, 8, 3)
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
    }

    def 'We can use the "filter" method as a shortcut for "stream().filter(..)".'()
    {
        when : 'We create a tensor...'
            Nda<Integer> t = Nda.of(Integer).withShape(3, 2).andFill(9, 1, 0, 1, -4, 8)

        then : 'The filter method returns a filtered stream which we can collect!'
            t.filter({it < 3})
                    .collect(Collectors.toList()) == [1, 0, 1, -4]
    }

    def 'We can use the "flatMap" method as a shortcut for "stream().flatMap(..)".'()
    {
        when : 'We create a nd-array...'
            Nda<Integer> t = Nda.of(Integer).withShape(3, 2).andFill(9, 1, 0, 1, -4, 8)

        then : 'We can use the "flatMap" method as a shortcut for "stream().flatMap(..)"'
            t.flatMap({it < 3 ? [it, it] : []})
                    .collect(Collectors.toList()) == [1, 1, 0, 0, 1, 1, -4, -4]
    }

    def 'ND-Array mapping lambdas produce expected nd-arrays.'() {
        when : 'Instantiating a nd-array using an initializer lambda...'
            var t = Nda.of(Double.class)
                        .withShape(2,3)
                        .andWhere(( int i, int[] indices ) -> { (i - 2) as Double })

        then : 'The nd-array has been initialized with the expected values:'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"

        when : 'We want to perform an element wise mapping to a new nd-array...'
            def b = t.mapTo(String, (it) -> {"<$it>".replace(".0", "")})

        then : 'We expect the returned nd-array to be a String container whose Strings are formatted according to our mapping lambda.'
            b.toString() == "(2x3):[<-2>, <-1>, <0>, <1>, <2>, <3>]"
            b.itemType == String.class
        and : 'The original nd-array should not have changed because no inline operation occurred.'
            t.toString() == "(2x3):[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]"
    }

    def 'The "map" method is a shorter convenience method for mapping to the same type.'() {
        given : 'We create a nd-array with a single element.'
            var t = Nda.of(1d)

        when : 'We map the nd-array to a new nd-array of the same type.'
            var b = t.map( it -> it + 1 )
        then : 'The new nd-array should have the same value as the original nd-array.'
            b.toString() == "(1):[2.0]"
            b.itemType == Double.class
        and : 'The original nd-array should not have changed because no inline operation occurred.'
            t.toString() == "(1):[1.0]"
    }

    def 'We can find both min and max items in a tensor by providing a comparator.'()
    {
        given : 'We create a tensor of chars for which we want to find a min and max values:'
            var t = Nda.of(Character)
                                .withShape(2, 13)
                                .andWhere(( int i, int[] indices ) -> { (i+65) as char } )

        when : 'We find the min value by passing a comparator (which takes 2 values and returns an int):'
            var min = t.minItem( ( Character a, Character b ) -> a.compareTo( b ) )

        then : 'The resulting min value is the last item in the tensor, the letter A:'
            min == 'A'

        when : 'We now try to find the max value:'
            var max = t.maxItem( ( Character a, Character b ) -> a.compareTo( b ) )

        then : 'The max value is the last item in the tensor, the letter Z:'
            max == 'Z'
    }

    def 'We can collect a stream into a nd-array.'()
    {
        given : 'We create a stream of integers.'
            var stream = Stream.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

        when : 'We collect the stream into a nd-array.'
            var n = stream.collect(Nda.shaped(2, 5))

        then : 'The resulting nd-array should have the same values as the stream.'
            n.toString() == "(2x5):[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"
    }

}
