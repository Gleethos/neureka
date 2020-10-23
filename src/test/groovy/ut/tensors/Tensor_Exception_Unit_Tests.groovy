package ut.tensors

import neureka.Tsr
import org.slf4j.Logger
import spock.lang.Shared
import spock.lang.Specification

class Tensor_Exception_Unit_Tests extends Specification
{

    @Shared def oldLogger

    def setup() {
        oldLogger = Tsr._LOGGER
        Tsr._LOGGER = Mock( Logger )
    }

    def cleanup() {
        Tsr._LOGGER = oldLogger
    }

    def "Trying to inject an empty tensor into another causes fitting exception."()
    {
        given : 'A new tensor instance used fo rexception testing.'
            Tsr t = new Tsr([6, 6], -1)

        when : 'We try to inject an empty tensor whose size does of course not match...'
            t[[1..3], [1..3]] = new Tsr()

        then : 'The expected message is being thrown, which tells us that '
            def exception = thrown(IllegalArgumentException)
            assert exception.message.contains("Provided tensor is empty! Empty tensors cannot be injected.")
        and : 'The exception has also been logged.'
            1 * Tsr._LOGGER.error( "Provided tensor is empty! Empty tensors cannot be injected." )
    }

    def 'Passing an invalid object into Tsr constructor causes descriptive exception.'()
    {
        when : 'A tensor is being instantiated with a nonsensical parameter.'
            new Tsr( new Scanner( System.in ) )
        then : 'An exception is being thrown which tells us about it.'
            def exception = thrown(IllegalArgumentException)
            exception.message.contains(
                    "Cannot create tensor from argument of type 'java.util.Scanner'!"
            )
        and : 'The logger logs the exception message!'
            1 * Tsr._LOGGER.error( "Cannot create tensor from argument of type 'java.util.Scanner'!" )
    }


    def 'Passing an invalid key object into the "getAt" method causes a descriptive exception.'()
    {
        given : 'A new test tensor is being instantiated.'
            Tsr t = new Tsr( [2, 3], -1..6 )

        when : 'A nonsensical object is being passed to the tensor.'
            t[ Integer.class ]

        then : 'An exception is being thrown which tells us about it.'
            def exception = thrown(IllegalArgumentException)
            exception.message.contains(
                    "Cannot create tensor slice from key of type 'java.lang.Class'!"
            )
        and : 'The logger logs the exception message!'
            1 * Tsr._LOGGER.error( "Cannot create tensor slice from key of type 'java.lang.Class'!" )
    }





}
