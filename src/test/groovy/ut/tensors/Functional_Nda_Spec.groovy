package ut.tensors

import neureka.Nda
import neureka.Neureka
import neureka.common.utility.SettingsLoader
import neureka.view.NDPrintSettings
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

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

}
