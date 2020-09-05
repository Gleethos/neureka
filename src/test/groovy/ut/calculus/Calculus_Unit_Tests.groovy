package ut.calculus

import neureka.calculus.Function
import neureka.calculus.frontend.assembly.FunctionBuilder
import spock.lang.Specification

class Calculus_Unit_Tests extends Specification
{

    def 'Test scalar results of Function "1/I[0]" instance.'(
            double[] inputs, Integer index, double expected
    ){
        given :
            Function f = FunctionBuilder.build("1/I[0]", false)
        expect :
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected
        where :
            inputs           | index || expected
            new double[]{2}  | 0     || -0.25
            new double[]{2}  | null  ||  0.5
    }


    def 'Test scalar results of Function "I[0]+1/I[0]" instance.'(
            double[] inputs, Integer index, double expected
    ){
        given :
            Function f = FunctionBuilder.build("I[0]+1/I[0]", false)
        expect :
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected
        where :
            inputs           | index || expected
            new double[]{2}  | null  ||  2.5
            new double[]{-1} | 0     ||  0.0
            new double[]{-3} | 0     ||  0.8888888888888888
            new double[]{0.2}| 0     ||  -23.999999999999996
    }


    def 'Test scalar results of Function "(I[0]+1/I[0])^-I[0]" instance'(
            double[] inputs, Integer index, double expected
    ){
        given :
            Function f = FunctionBuilder.build("(I[0]+1/I[0])^-I[0]", false)
        expect :
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected
        where :
            inputs           | index || expected
            new double[]{1}  | null  ||  0.5
            new double[]{0.2}| 0     ||  -0.5217778675999797
    }


    def 'test scalar results of Function "(cos(I[0]*5)/5+I[0])*(1+sin(I[0])/2)" instance.'(
            double[] inputs, Integer index, double expected
    ){
        given :
        Function f = FunctionBuilder.build("(cos(I[0]*5)/5+I[0])*(1+sin(I[0])/2)", false)
        expect :
        if (index!=null) assert f.derive( inputs, index )==expected
        else assert f.call( inputs )==expected
        where :
        inputs           | index || expected
        new double[]{3}  | null  ||  3.049021713079475
        new double[]{2.5}| null  ||  3.507365283517986
        new double[]{0}  | null  ||  0.2
        new double[]{0  }| 0     ||  1.1
        new double[]{0.5}| 0     ||  0.646867884000033
        new double[]{1.6}| 0     ||  -0.00697440343353687
        new double[]{-4 }| 0     ||  3.9174193383745917
    }


    def 'Test scalar results of Function "sumjs((cos(I[j]*5)/5+I[j])*(1+sin(I[j])/2))" instance.'(
            double[] inputs, Integer index, double expected
    ){
        given :
        Function f = FunctionBuilder.build("sumjs((cos(I[j]*5)/5+I[j])*(1+sin(I[j])/2))", false)
        expect :
        if (index!=null) assert f.derive( inputs, index )==expected
        else assert f.call( inputs )==expected
        where :
        inputs                           | index || expected
        new double[]{0.0, 0.5, 1.6, -4.0}| 0     ||  1.1
        new double[]{0.0, 0.5, 1.6, -4.0}| 1     ||  0.646867884000033
        new double[]{0.0, 0.5, 1.6, -4.0}| 2     ||  -0.00697440343353687
        new double[]{0.0, 0.5, 1.6, -4.0}| 3     ||  3.9174193383745917
    }


    def 'Test parsed equations when building Function instances.'(
            String equation, String expected
    ) {
        expect : Function.create(equation).toString() == expected

        where :
            equation                                    || expected
            "ig0*(igj)xI[g1]"                           || "((Ig[0] * Ig[j]) x Ig[1])"
            "sumJs(ij)"                                 || "sumJs(I[j])"
            "sumJs(1*(4-2/ij))"                         || "sumJs(1.0 * (4.0 - (2.0 / I[j])))"
            "quadratic(sftpls(Ij))"                    || "quad(softplus(I[j]))"
            "softplus(I[3]^(3/i1)/sumJs(Ij^2)-23+I0/i1)"|| "softplus((((I[3] ^ (3.0 / I[1])) / sumJs(I[j] ^ 2.0)) - 23.0) + (I[0] / I[1]))"
            "1+3+5-23+I0*45/(345-651^I3-6)"             || "(1.0 + 3.0 + (5.0 - 23.0) + (I[0] * (45.0 / (345.0 - (651.0 ^ I[3]) - 6.0))))"
            "sin(23*i1)-cos(i0^0.3)+tanh(23)"           || "((sin(23.0 * I[1]) - cos(I[0] ^ 0.3)) + tanh(23.0))"
            "2*3/2-1"                                   || "((2.0 * (3.0 / 2.0)) - 1.0)"
            "3x5xI[4]xI[3]"                             || "(((3.0 x 5.0) x I[4]) x I[3])"
            "[1,0, 5,3, 4]:(tanh(i0xi1))"               || "([1,0,5,3,4]:(tanh(I[0] x I[1])))"
            "[0,2, 1,3, -1](sig(I0))"                   || "([0,2,1,3,-1]:(sig(I[0])))"
            "I[0]<-I[1]->I[2]"                          || "((I[0] <- I[1]) -> I[2])"
            "quadratic(I[0]) -> I[1] -> I[2]"           || "((quad(I[0]) -> I[1]) -> I[2])"
            "((tanh(i0)"                                || "tanh(I[0])"
            '($$(gaus(i0*()'                            || "gaus(I[0] * 0.0)"
            "rrlu(i0)"                                  || "relu(I[0])"
            "th(i0)*gas(i0+I1)"                         || "(tanh(I[0]) * gaus(I[0] + I[1]))"
            "th(i0)dgus(i0+I1)"                         || "(tanh(I[0]) d gaus(I[0] + I[1]))"
            "ijdguus(i0+I1)"                            || "(I[j] d gaus(I[0] + I[1]))"
            "ijssumJs(i0+Ij)"                           || "(I[j] s sumJs(I[0] + I[j]))"
            "i[0] d>> i[1] d>> I[2]"                    || "(I[0] d"+((char)187)+" I[1] d"+((char)187)+" I[2])"
    }


    def 'Parsed equations throw expected error messages.' (
        String equation, String expected
    ) {
        when :
            Function.create(equation)

        then :
            def error = thrown(IllegalArgumentException)
            assert error.message==expected

        where :
            equation                || expected
            "i[0] d>> i[1]"         || "The function/operation 'd"+((char)187)+"' expects 3 parameters, however 2 where given!"
            "softplus(I[0],I[1],I[2])"   || "The function/operation 'softplus' expects 1 parameters, however 3 where given!"
            "sig(I[0],I[1],I[2])"   || "The function/operation 'sig' expects 1 parameters, however 3 where given!"
            "sumjs(I[0],I[1],I[2])" || "The function/operation 'sumJs' expects 1 parameters, however 3 where given!\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration."
            "prodjs(I[0],I[1],I[2])"|| "The function/operation 'prodJs' expects 1 parameters, however 3 where given!\nNote: This function is an 'indexer'. Therefore it expects to sum variable 'I[j]' inputs, where 'j' is the index of an iteration."
    }


    def 'Test scalar results of various Function instances.'(
            String equation, double[] inputs, Integer index, double expected
    ){
        given :
            Function f = FunctionBuilder.build(equation, false)
        expect :
            if ( index!=null ) assert f.derive( inputs, index ) == expected
            else assert f.call( inputs )==expected
        where :
            equation                         | inputs                           | index || expected
            "6/2*(1+2)"                      | new double[]{}                   | null  ||   9
            "sumJs(Ij)"                      | new double[]{2, 3.2, 6}          | null  ||  11.2
            "prod(Ij)"                       | new double[]{0.5, 0.5, 100}      | null  ||  25
            "prod(prod(Ij))"                 | new double[]{0.5, 0.5, 10}       | null  || (2.5 * 2.5 * 2.5)
            "I3/i[1]-I0+2+i2"                | new double[]{5, 4, 3, 12}        | null  ||   3
            "i3*i1/(i4-i0-2)-sig(0)+tanh(0)" | new double[]{-4, -2, 6, -3, -8}  | null  ||  -1.5
            "(i0*i1)*i2"                     | new double[]{2, 3, -2}           | 0     || -6
            "softplus(i0*i1)*i2"                  | new double[]{2, 3, -2}           | 0     || -5.985164261060192
            "prod(ij)"                       | new double[]{2, 3, -2}           | 1     || -4
            "relu(prod(ij))"                 | new double[]{2, 3, -2}           | null  || -0.12
            "relu(prod(ij))"                 | new double[]{2, 3, -2}           | 1     || -0.04
            "quad(prod(ij)+6)"               | new double[]{2, 3, -2}           | null  || 36
            "quad(prod(ij)+6)"               | new double[]{2, 3, -2}           | 1     || -12*-4
            "quad(abs(prod(ij))-6)"          | new double[]{2, 3, -2}           | null  || 36
            "quad(abs(prod(ij))-6)"          | new double[]{2, 3, -2}           | 1     || -12*-4
            "sumJs(ij)"                      | new double[]{2, 3, -2}           | null  || 3
            "sumJs(ij)"                      | new double[]{2, 3, -2}           | 1     || 1
            //Todo: pow inside indexer!
    }


}
