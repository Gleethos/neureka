package ut.utility

import neureka.utility.ListReader
import spock.lang.Specification

class ListReader_Spec extends Specification
{

    def 'The ListReader can interpret nested lists resembling a matrix into a shape list and value list.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            def data = [
                    [1, 2, 3],
                    [4, 5, 6]
            ]
        and : 'An empty shape list.'
            def shape = []
        and : 'An empty list as a target for the flattened data.'
            def flattened = []

        when : 'We use the reader to fill the 2 lists defined above...'
            new ListReader(data, 0, flattened, shape, (o) -> o)

        then : 'The shape list will have the shape of the "matrix".'
            shape == [2, 3]
        and : 'The flattened data is as expected!'
            flattened == [1, 2, 3, 4, 5, 6]
    }

    def 'The ListReader can interpret nested lists resembling a 3D tensor into a shape list and value list.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            def data = [
                            [
                                    [1, 2, 3, 0],
                                    [4, 5, 6, -1]
                            ], [
                                    [1, 2, 3, -2],
                                    [4, 5, 6, -3]
                            ]
                        ]
        and : 'An empty shape list.'
            def shape = []
        and : 'An empty list as a target for the flattened data.'
            def flattened = []

        when : 'We use the reader to fill the 2 lists defined above...'
            new ListReader(data, 0, flattened, shape, (o) -> o)

        then : 'The shape list will have the shape of the "tensor".'
            shape == [2, 2, 4]
        and : 'The flattened data is as expected!'
            flattened == [1, 2, 3, 0, 4, 5, 6, -1, 1, 2, 3, -2, 4, 5, 6, -3]
    }

    def 'The ListReader can interpret nested lists into a shape list and value list.'(
            Object data, List<Integer> expectedShape, List<Object> expectedData
    ) {
        given : 'An empty shape list.'
            def shape = []
        and : 'An empty list as a target for the flattened data.'
            def flattened = []

        when : 'We use the reader to fill the 2 lists defined above...'
            new ListReader(data, 0, flattened, shape, (o) -> o)

        then : 'The shape list will have the shape of the "matrix".'
            shape == expectedShape
        and : 'The flattened data is as expected!'
            flattened == expectedData

        where :
            data            || expectedShape | expectedData
            [42]            || [1]           | [42]
            [[43]]          || [1, 1]        | [43]
            [[-1],[+1]]     || [2, 1]        | [-1, 1]
            [[24, 42]]      || [1, 2]        | [24, 42]
            [["24", "42"]]  || [1, 2]        | ["24", "42"]
            [[true],[false]]|| [2, 1]        | [true, false]
    }

}
