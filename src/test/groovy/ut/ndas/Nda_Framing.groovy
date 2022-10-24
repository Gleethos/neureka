package ut.ndas

import neureka.Nda
import neureka.Neureka
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("NDA Framing")
@Narrative('''

    Framing an nd-array is all about naming its axes and then using those names to
    access, read or write its values in a more convenient and human readable way.

''')
@Subject([Nda])
class Nda_Framing extends Specification
{
    def setup() {
        Neureka.get().reset()
    }

    def 'We can label the columns of a rank 2 nd-array.'()
    {
        given : 'A rank 2 nd-array with shape (2, 3).'
            def nda = Nda.of(String).withShape(2, 3).andFill("1", "a", "§", "2", "b", "%")
        when : 'We label the columns of the nd-array.'
            nda.mut.label("Framed").mut.labelAxes([null, ["Num", "Letter", "Symbol"]] as String[][])
        then : 'The columns are labeled as expected.'
            nda.toString() == "(2x3):[\n" +
                              "   (   Num )(Letter)(Symbol ):( Framed )\n" +
                              "   [    1  ,    a  ,    §   ],\n" +
                              "   [    2  ,    b  ,    %   ]\n" +
                              "]"

    }

    def 'We can label the columns and rows of a rank 3 nd-array.'()
    {
        given : 'A rank 3 nd-array with shape (2, 3, 4).'
            def nda = Nda.of(String)
                                        .withShape(2, 2, 3)
                                        .andFill("1", "a", "§", "2", "b", "%" , "3", "c", "€", "4", "d", "£")
        when : 'We label the columns and rows of the nd-array.'
            nda.mut.label("Framed").mut.labelAxes([null, ["A", "B"], ["Num", "Letter", "Symbol"]])
        then : 'The columns and rows are labeled as expected.'
            nda.toString() == "(2x2x3):[\n" +
                              "   [\n" +
                              "      (   Num )(Letter)(Symbol ):( Framed )\n" +
                              "      [    1  ,    a  ,    §   ]:( A ),\n" +
                              "      [    2  ,    b  ,    %   ]:( B )\n" +
                              "   ],\n" +
                              "   [\n" +
                              "      (   Num )(Letter)(Symbol ):( Framed )\n" +
                              "      [    3  ,    c  ,    €   ]:( A ),\n" +
                              "      [    4  ,    d  ,    £   ]:( B )\n" +
                              "   ]\n" +
                              "]"
    }

    def 'We can use labels as selectors for slicing.'()
    {
        given : 'A rank 3 nd-array with shape (2, 3, 4).'
            def nda = Nda.of(String)
                                        .withShape(2, 2, 3)
                                        .andFill("1", "a", "§", "2", "b", "%" , "3", "c", "€", "4", "d", "£")
        when : 'We label the columns and rows of the nd-array.'
            nda.mut.label("Framed").mut.labelAxes([["M1", "M2"], ["A", "B"], ["Num", "Letter", "Symbol"]])
        then : 'The columns and rows are labeled as expected.'
            nda.toString() == "(2x2x3):[\n" +
                              "   ( M1 ):[\n" +
                              "      (   Num )(Letter)(Symbol ):( Framed )\n" +
                              "      [    1  ,    a  ,    §   ]:( A ),\n" +
                              "      [    2  ,    b  ,    %   ]:( B )\n" +
                              "   ],\n" +
                              "   ( M2 ):[\n" +
                              "      (   Num )(Letter)(Symbol ):( Framed )\n" +
                              "      [    3  ,    c  ,    €   ]:( A ),\n" +
                              "      [    4  ,    d  ,    £   ]:( B )\n" +
                              "   ]\n" +
                              "]"
        when : 'We slice the nd-array using labels.'
            def slice = nda["M1", "A", "Num"]
        then : 'The slice is as expected.'
            slice.items == ["1"]
            slice.toString() == "(1x1x1):[\n" +
                                "   ( M1 ):[\n" +
                                "      (   Num  ):( Framed:slice )\n" +
                                "      [    1   ]:( A )\n" +
                                "   ]\n" +
                                "]"
    }

    def 'The slice of a labeled vector is labeled too.'()
    {
        given : 'A rank 1 nd-array with shape (6).'
            var nda = Nda.of("a", "b", "c", "d", "e", "f", "g", "h")
        when : 'We label the nd-array.'
            nda.mut.label("Framed").mut.labelAxes(["Letters":["A", "B", "C", "D", "E", "F", "G", "H"]])
        then : 'The nd-array is labeled as expected.'
            nda.toString() == "(8):(    A  )(   B  )(   C  )(   D  )(   E  )(   F  )(   G  )(   H   ):( Framed )\n" +
                              "    [    a  ,    b  ,    c  ,    d  ,    e  ,    f  ,    g  ,    h   ]"
        when : 'We slice the nd-array using labels.'
            var slice = nda.getAt(["B".."G"]:3)
        then : 'The slice is as expected.'
            slice.items == ["b", "e"]
            slice.toString() == "(2):(    B  )(   E   ):( Framed:slice )\n" +
                                "    [    b  ,    e   ]"

    }



}
