package st

import neureka.Neureka
import neureka.utility.TsrAsString
import st.tests.BroadSystemTest
import spock.lang.Specification

class Broad_System_Test extends Specification
{
    def setupSpec() {
        Neureka.instance().reset()
        // Configure printing of tensors to be more compact:
        Neureka.instance().settings().view().asString = TsrAsString.configFromCode("dgc")
    }

    def 'Test integration broadly.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadSystemTest.on()
    }


}
