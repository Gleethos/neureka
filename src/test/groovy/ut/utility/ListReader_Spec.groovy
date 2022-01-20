package ut.utility

import neureka.common.utility.ListReader
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("The Internal ListReader turning lists into flat arrays with shape and type data")
@Narrative('''
    
    This specification covers an internal class which should not be used
    outside this library, namely the ListReader class.
    This class is simply a converter which turns nested lists
    into flat arrays alongside the type of the elements and the shape of this "tensor".
    
''')
class ListReader_Spec extends Specification
{

    def 'The ListReader can interpret nested lists resembling a matrix into a shape list and value list.'()
    {
        given : 'We have a nested list whose structure resembles a matrix!'
            var data = [
                    [1, 2, 3],
                    [4, 5, 6]
            ]
        when : 'We use the reader to internally fill 2 lists representing shape and data...'
            var result = ListReader.read(data, (o)->o)

        then : 'The shape list will have the shape of the "matrix".'
            result.shape == [2, 3]
        and : 'The flattened data is as expected!'
            result.data == [1, 2, 3, 4, 5, 6]
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
        when : 'We use the reader to internally fill 2 lists representing shape and data...'
            var result = ListReader.read(data, (o)->o)

        then : 'The shape list will have the shape of the "tensor".'
            result.shape == [2, 2, 4]
        and : 'The flattened data is as expected!'
            result.data == [1, 2, 3, 0, 4, 5, 6, -1, 1, 2, 3, -2, 4, 5, 6, -3]
    }

    def 'The ListReader can interpret nested lists into a shape list and value list.'(
            Object data, List<Integer> expectedShape, List<Object> expectedData
    ) {
        when : 'We use the reader to internally fill 2 lists representing shape and data...'
            var result = ListReader.read(data, (o)->o)

        then : 'The shape list will have the shape of the "matrix".'
            result.shape == expectedShape
        and : 'The flattened data is as expected!'
            result.data == expectedData

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
