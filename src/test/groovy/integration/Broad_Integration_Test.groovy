package integration

import integration.tests.BroadIntegrationTest
import spock.lang.Specification

class Broad_Integration_Test extends Specification
{
    def 'Test integration broadly.'()
    {
        expect : BroadIntegrationTest.on()
    }


}
