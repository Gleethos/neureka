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

    def 'Test integration broadly.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadSystemTest.on()
    }


}
