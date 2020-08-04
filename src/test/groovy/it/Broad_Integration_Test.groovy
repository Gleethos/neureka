package it

import it.tests.BroadIntegrationTest
import spock.lang.Specification

class Broad_Integration_Test extends Specification
{
    def 'Test integration broadly.'()
    {
        expect : 'The integration test runs without exceptions or assertion errors.'
            BroadIntegrationTest.on()
    }


}
