package ut.calculus

import neureka.calculus.Function
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("Parsing Expressions into Functions")
@Narrative('''

    Neureka uses the 'Function' interface as a representation of a
    nested structure of operations.
    This means that a 'Function' is simply an abstract syntax trees made up of other 'Function' implementations
    which are assembled together by a parser receiving a string expression.
    In this specification we ensure that function expressions will be properly parsed into
    'Function' implementations.

''')
class Calculus_Parsing_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """
            This specification ensures that functions can be created from String expressions
            using factory methods on the $Function interface.
            The implementation details as to how exactly this leads to an abstract syntax tree
            will not be covered here.
            This is because the parsing procedure is rather complex and the only thing we care about 
            is the result.   
            <br>
            Within a given expression String passed to the parser, function inputs are
            recognized by 'I[j]', 'Ij' or 'ij', where j is the input index.
            Functions accept arrays as their inputs,
            which is why variables must be targeted in such a way.
            There are also many mathematical function like 'sig(..)', 'tanh(..)', 'sin(..)', 'cos(..)' 
            and many more which are recognised by the parser. 
            Other than that the syntax is rather mundane with respect to traditional
            operations like for example plus '+', minus '-', times '*', ... etc.      <br>
        """
    }

    def 'Test parsed equations when building Function instances.'(
            String equation, String expected
    ) {
        expect : 'A Function created from a given expression will be parsed as expected.'
            Function.of(equation).toString() == expected

        where : 'The following expressions and expected exception messages are being used :'
            equation                                      || expected
            "fast_tanh(i0*i1)"                            || "fast_tanh(I[0] * I[1])"
            "ig0*(igj)xI[g1]"                             || "((Ig[0] * Ig[j]) x Ig[1])"
            "sumJs(ij)"                                   || "sumJs(I[j])"
            "sumJs(1*(4-2/ij))"                           || "sumJs(1.0 * (4.0 - (2.0 / I[j])))"
            "quadratic(sftpls(Ij))"                       || "quad(softplus(I[j]))"
            "softplus(I[3]**(3/i1)/sumJs(Ij**2)-23+I0/i1)"|| "softplus((((I[3] ** (3.0 / I[1])) / sumJs(I[j] ** 2.0)) - 23.0) + (I[0] / I[1]))"
            "1+3+5-23+I0*45/(345-651**I3-6)"              || "(1.0 + 3.0 + (5.0 - 23.0) + (I[0] * (45.0 / (345.0 - (651.0 ** I[3]) - 6.0))))"
            "sin(23*i1)-cos(i0**0.3)+tanh(23)"            || "((sin(23.0 * I[1]) - cos(I[0] ** 0.3)) + tanh(23.0))"
            "4 *-2"                                       || "(4.0 * -2.0)"
            "fast_gaus(i0*i1)"                            || "fast_gaus(I[0] * I[1])"
            "2*3/1-2"                                     || "((2.0 * (3.0 / 1.0)) - 2.0)"
            "3x5xI[4]xI[3]"                               || "(((3.0 x 5.0) x I[4]) x I[3])"
            "[1,0, 5,3, 4]:(tanh(i0xi1))"                 || "([1,0,5,3,4]:(tanh(I[0] x I[1])))"
            "[0,2, 1,3, -1](sig(I0))"                     || "([0,2,1,3,-1]:(sig(I[0])))"
            "I[0]<-I[1]"                                  || "(I[0] <- I[1])"
            "quadratic(I[0]) <- (I[1] <- I[2])"           || "(quad(I[0]) <- (I[1] <- I[2]))" // '<-' is the assign operator!
            "((tanh(i0)"                                  || "tanh(I[0])"
            '($$(gaus(i0*()'                              || "gaus(I[0] * 0.0)"
            "rrlu(i0)"                                    || "relu(I[0])"
            "th(i0)*gaaus(i0+I1)"                         || "(tanh(I[0]) * gaus(I[0] + I[1]))"
            "dimtrim(I[0])"                               || "dimtrim(I[0])"
            "add(I[0], 3, 3/I[1])"                        || "(I[0] + 3.0 + (3.0 / I[1]))"
            "multiply(1, 4, -2, I[1])"                    || "(1.0 * 4.0 * -2.0 * I[1])"
            "divide(I[0], 3*I[1], I[3]-6)"                || "(I[0] / (3.0 * I[1]) / (I[3] - 6.0))"
            "i0@i1"                                       || "(I[0] @ I[1])"
            // TODO: WIP:
            //"soft( [2, 1, 0]:( I[0] )*-100 )"           || "softplus( [2,1,0]:( I[0] ) * -100.0 )"
    }


    def 'Parsed equations throw expected error messages.' (
            String equation, String expected
    ) {
        when : 'We try to instantiate a Function by passing an expression String...'
            Function.of(equation)

        then : 'An exception is being thrown that contains the expected message!'
            def error = thrown(IllegalArgumentException)
            assert error.message==expected

        where : 'The following expressions and expected exception messages are being used :'
            equation                  || expected
            "softplus(I[0],I[1],I[2])"|| "The function/operation 'softplus' expects 1 parameters, however 3 where given!"
            "sig(I[0],I[1],I[2])"     || "The function/operation 'sig' expects 1 parameters, however 3 where given!"
            "sumjs(I[0],I[1],I[2])"   || "The function/operation 'sumJs' expects 1 parameters, however 3 where given!\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration."
            "prodjs(I[0],I[1],I[2])"  || "The function/operation 'prodJs' expects 1 parameters, however 3 where given!\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration."
    }

    def 'Functions can derive themselves according to the provided index of the input which ought to be derived.'(
            String equation, int index, String expected
    ) {

        expect : 'A Function created from a given expression will produce the expected derivative String.'
            Function.of(equation).getDerivative( index ).toString() == expected

        where : 'The following expressions and derivation indices are being used :'
        equation                     | index || expected
        "1 - I[0] * 3"               | 0     || "-3.0"
        "i0 / 6"                     | 0     || "(1.0 / 6.0)"
        "ln( 4 * i0 )"               | 0     || "(4.0 / (4.0 * I[0]))"
        "4**I[0]"                    | 0     || "(ln(4.0) * (4.0 ** I[0]))"
        "i0 ** 3"                    | 0     || "(3.0 * (I[0] ** (3.0 - 1.0)))"
        "(I[0] * I[1] * I[0]) + 3"   | 0     || "((I[1] * I[0]) + (I[0] * I[1]))"
        "3 ** (i0 / 2)"              | 0     || "((1.0 / 2.0) * (ln(3.0) * (3.0 ** (I[0] / 2.0))))"
        "(2 * I[0]) / (1 - I[0] * 3)"| 0     || "((2.0 / (1.0 - (I[0] * 3.0))) - (((2.0 * I[0]) * -3.0) / ((1.0 - (I[0] * 3.0)) ** 2.0)))"
        //"(3 * I[0]) ** (I[0]/6)"   | 0     || ""

    }

}
