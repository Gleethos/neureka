package st

import neureka.Neureka
import neureka.view.TsrStringSettings
import st.tests.BroadSystemTest
import spock.lang.Specification

class Broad_System_Test extends Specification
{
    def setupSpec() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().tensors({ TsrStringSettings it ->
            it.scientific( true )
            it.multiline( false )
            it.withGradient( true )
            it.withCellSize( 1 )
            it.withValue( true )
            it.withRecursiveGraph( false )
            it.withDerivatives( true )
            it.withShape( true )
            it.cellBound( false )
            it.withPostfix(  "" )
            it.withPrefix(  ""  )
            it.withSlimNumbers(  false )  
        })
    }

    def 'Test integration broadly.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadSystemTest.on()
    }


}
