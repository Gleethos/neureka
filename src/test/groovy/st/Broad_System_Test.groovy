package st

import neureka.Neureka
import st.tests.BroadSystemTest
import spock.lang.Specification

class Broad_System_Test extends Specification
{
    def setupSpec() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().asString = "dgc"
    }

    def 'Test integration broadly.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadSystemTest.on()
    }


}
