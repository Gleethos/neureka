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



}
