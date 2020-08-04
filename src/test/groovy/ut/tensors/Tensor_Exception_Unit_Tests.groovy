package ut.tensors

import neureka.Tsr
import spock.lang.Specification

class Tensor_Exception_Unit_Tests extends Specification
{

    def "Trying to inject an empty tensor into another causes fitting exception."()
    {
        given : Tsr t = new Tsr([6, 6], -1)

        when : t[[1..3], [1..3]] = new Tsr()

        then :
            def exception = thrown(IllegalArgumentException)
            assert exception.message.contains("Provided tensor is empty!")
    }

    def "Passing invalid object into Tsr constructor causes descriptive exception."()
    {
        when : new Tsr(new Scanner(System.in))
        then :
            def exception = thrown(IllegalArgumentException)
            exception.message.contains(
                    "Cannot create tensor from argument of type 'java.util.Scanner'!"
            )
    }

}
