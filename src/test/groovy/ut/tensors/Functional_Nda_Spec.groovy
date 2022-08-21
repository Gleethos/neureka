package ut.tensors

import neureka.Nda
import neureka.Neureka
import neureka.common.utility.SettingsLoader
import neureka.devices.opencl.CLContext
import neureka.view.NDPrintSettings
import spock.lang.Specification


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
