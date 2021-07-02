package ut.utility

import neureka.utility.ListReader
import spock.lang.Specification

class ListReader_Exception_Tests extends Specification
{

    def 'The ListReader will detect inconsistent types in the provided data.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            def data = [
                [1,  2,  3],
                [4, "5", 6]
            ]
        and : 'An empty shape list.'
            def shape = []
        and : 'An empty list as a target for the flattened data.'
            def flattened = []

        when : 'We use the reader to fill the 2 lists defined above...'
            new ListReader(data, 0, flattened, shape)

        then :
            def exception = thrown(IllegalArgumentException)
            exception.message == "Type inconsistency encountered. Not all leave elements are of the same type!\n" +
                                    "Expected type 'Integer', but encountered 'String'."
    }

    def 'The ListReader will detect inconsistent degrees of nesting in the provided data.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            def data = [
                [1,  2,  3],
                [4, 5, 6, 7]
            ]
        and : 'An empty shape list.'
            def shape = []
        and : 'An empty list as a target for the flattened data.'
            def flattened = []

        when : 'We use the reader to fill the 2 lists defined above...'
            new ListReader(data, 0, flattened, shape)

        then :
            def exception = thrown(IllegalArgumentException)
            exception.message == "Size inconsistency encountered at nest level '0'. Not all nested lists are equally sized.\n" +
                                    "Expected size '3', but encountered '4'."
    }

}
