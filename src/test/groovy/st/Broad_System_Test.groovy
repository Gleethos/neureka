package st

import st.tests.BroadSystemTest
import spock.lang.Specification

class Broad_System_Test extends Specification
{
    def 'Test integration broadly.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadSystemTest.on()
    }


}
