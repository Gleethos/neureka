package ut.ndas

import neureka.Nda
import neureka.Neureka
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Nda framing")
@Narrative('''

    Immutability is a core concept of the Neureka library.
    This means that the Nda API does not expose mutability directly.
    Instead, the API exposes methods that return new instances of Nda
    that are derived from the original instance.
    
    This is also true for labeling operations, 
    meaning that the Nda API does not directly expose methods that mutate labels of an Nda
    but instead provides methods that return new instances of Nda
    with different labels.
    
    Don't be concerned about the performance implications of this,
    because in the vast majority of cases the new instance will be backed by the same data array
    as the original instance!
    
''')
@Subject([Nda])
class Nda_Framing_Spec extends Specification
{
    def setup() {
        Neureka.get().reset()
    }

    def 'An Nda can be labeled.'()
    {
        given : 'We create a vector of Strings.'
            Nda<String> nda = Nda.of("a".."c")
        expect : 'Initially the vector is not labeled.'
            nda.label == ""
        when : 'We label the vector.'
            nda = nda.withLabel( "my-label" )
        then : 'The vector will have the expected label.'
            nda.label == "my-label"
    }

    def 'We can label the columns of a rank 2 nd-array.'()
    {
        given : 'A rank 2 nd-array with shape (2, 3).'
            def nda = Nda.of(String).withShape(2, 3).andFill("1", "a", "§", "2", "b", "%")
        when : 'We label the columns of the nd-array.'
            nda = nda.withLabel("Framed").withLabels([null, ["Num", "Letter", "Symbol"]] as String[][])
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
            nda = nda.withLabel("Framed").withLabels([null, ["A", "B"], ["Num", "Letter", "Symbol"]])
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
            nda = nda.withLabel("Framed").withLabels([["M1", "M2"], ["A", "B"], ["Num", "Letter", "Symbol"]])
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
        given : 'A rank 1 nd-array with shape (8).'
            var nda = Nda.of("a".."h")
        when : 'We label the nd-array.'
            nda = nda.withLabel("Framed").withLabels(["Letters":["A", "B", "C", "D", "E", "F", "G", "H"]])
        then : 'The nd-array is labeled as expected.'
            nda.toString() == "(8):(    A  )(   B  )(   C  )(   D  )(   E  )(   F  )(   G  )(   H   ):( Framed )\n" +
                              "    [    a  ,    b  ,    c  ,    d  ,    e  ,    f  ,    g  ,    h   ]"
        when : """
            We slice the nd-array using labels...
            Note that we are using the map syntax of `[i..j]:k` which is 
            semantically equivalent to pythons `i:j:k` syntax for indexing! (numpy)
            Here `i` is the start index (or alias),
            `j` is the end index (alias) and `k` is the step size.
        """
            var slice = nda.getAt(["B".."G"]:3)
        then : 'The slice is as expected.'
            slice.items == ["b", "e"]
            slice.toString() == "(2):(    B  )(   E   ):( Framed:slice )\n" +
                                "    [    b  ,    e   ]"
    }

    def 'Concatenating 2 labeled nd-arrays will produce a nd-array which is also labeled.'()
    {
        given : 'Two rank 2 nd-arrays with shape (2x3).'
            var nda1 = Nda.of("a", "1", "!", "b", "2", "§").withShape(2,3)
            var nda2 = Nda.of("x", "2,50", "%", "y", "4,90", "&").withShape(2,3)
        when : 'We label the nd-arrays.'
            nda1 = nda1.withLabel("Nda1").withLabels(["rows":["A", "B"], "cols":["Letter", "Num", "Symbol"]])
            nda2 = nda2.withLabel("Nda2").withLabels(["rows":["1", "2"], "cols":["Letter",  "€",  "Symbol"]])
        then : 'The nd-arrays are labeled as expected.'
            nda1.toString() == "(2x3):[\n" +
                                "   ( Letter)(  Num )(Symbol ):( Nda1 )\n" +
                                "   [    a  ,    1  ,    !   ]:( A ),\n" +
                                "   [    b  ,    2  ,    §   ]:( B )\n" +
                                "]"
            nda2.toString() == "(2x3):[\n" +
                               "   ( Letter)(   €  )(Symbol ):( Nda2 )\n" +
                               "   [    x  ,  2,50 ,    %   ]:( 1 ),\n" +
                               "   [    y  ,  4,90 ,    &   ]:( 2 )\n" +
                               "]"
        when : 'We concatenate the nd-arrays.'
            var nda = nda1.concatAt(0, nda2)
        then : """
                The concatenated nd-array is labeled as expected.
                Note that conflicting labels will simply be merged into a single label.    
            """
            nda.toString() == "(4x3):[\n" +
                              "   ( Letter)( Num+€)(Symbol ):( Nda1+Nda2 )\n" +
                              "   [    a  ,    1  ,    !   ]:( A ),\n" +
                              "   [    b  ,    2  ,    §   ]:( B ),\n" +
                              "   [    x  ,  2,50 ,    %   ]:( 1 ),\n" +
                              "   [    y  ,  4,90 ,    &   ]:( 2 )\n" +
                              "]"
    }

}
