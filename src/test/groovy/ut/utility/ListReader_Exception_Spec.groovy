package ut.utility

import neureka.common.utility.ListReader
import spock.lang.Specification

class ListReader_Exception_Spec extends Specification
{

    def 'The ListReader will detect inconsistent types in the provided data.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            var data = [
                [1,  2,  3],
                [4, "5", 6]
            ]
        when : 'We use the reader to internally fill 2 lists representing shape and data...'
            var result = ListReader.read(data, (o)->o)

        then : 'However this leads to an exception because of type incoherence!'
            def exception = thrown(IllegalArgumentException)
            exception.message == "Type inconsistency encountered. Not all leave elements are of the same type!\n" +
                                    "Expected type 'Integer', but encountered 'String'."
    }

    def 'The ListReader will detect inconsistent degrees of nesting in the provided data.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            var data = [
                [1,  2,  3],
                [4, 5, 6, 7]
            ]
        when : 'We use the reader to internally fill 2 lists representing shape and data...'
            var result = ListReader.read(data, (o)->o)

        then :
            def exception = thrown(IllegalArgumentException)
            exception.message == "Size inconsistency encountered at nest level '0'. Not all nested lists are equally sized.\n" +
                                    "Expected size '3', but encountered '4'."
    }

}
