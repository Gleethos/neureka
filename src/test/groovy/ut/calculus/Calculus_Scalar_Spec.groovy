package ut.calculus

import neureka.Neureka
import neureka.calculus.Function
import neureka.calculus.assembly.FunctionBuilder
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Title

@Title("A Function as such!.")
@Narrative('''
    
    This specification defines the expected behaviour of the Function API
    with respect to receiving simple scalar values as arguments.
    
''')
class Calculus_Scalar_Spec extends Specification
{

    def 'Function "1/I[0]" instance returns expected scalar results.'(
            double[] inputs, Integer index, double expected
    ){
        given : 'We create a Function instance from expression "1/I[0]".'
            Function f = new FunctionBuilder( Neureka.get().backend() ).build("1/I[0]", false)

        expect : 'The function yields expected scalar results when called.'
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected

        where : 'The following input array, target derivative index and result scalar is used :'
            inputs | index || expected
            [2]    | 0     || -0.25
            [2]    | null  ||  0.5
    }


    def 'Function "I[0]+1/I[0]" instance returns expected scalar results.'(
            double[] inputs, Integer index, double expected
    ){
        given : 'We create a Function instance from expression "I[0]+1/I[0]".'
            Function f = new FunctionBuilder( Neureka.get().backend() ).build("I[0]+1/I[0]", false)

        expect : 'The function yields expected scalar results when called.'
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected

        where : 'The following input array, target derivative index and result scalar is used :'
            inputs | index || expected
            [2]    | null  ||  2.5
            [-1]   | 0     ||  0.0
            [-3]   | 0     ||  0.8888888888888888
            [0.2]  | 0     ||  -23.999999999999996
    }


    def 'Function "(I[0]+1/I[0])^-I[0]" instance returns expected scalar result.'(
            double[] inputs, Integer index, double expected
    ){
        given : 'We create a Function instance from expression "(I[0]+1/I[0])^-I[0]".'
            Function f = new FunctionBuilder( Neureka.get().backend() ).build("(I[0]+1/I[0])^-I[0]", false)

        expect : 'The function yields expected scalar results when called.'
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected

        where : 'The following input array, target derivative index and result scalar is used :'
            inputs  | index || expected
            [  1  ] | null  ||  0.5
            [ 0.2 ] | 0     ||  -0.5217778675999797
    }


    def 'Function "(cos(I[0]*5)/5+I[0])*(1+sin(I[0])/2)" instance returns expected scalars.'(
            double[] inputs, Integer index, double expected
    ){
        given :
            Function f = new FunctionBuilder( Neureka.get().backend() ).build("(cos(I[0]*5)/5+I[0])*(1+sin(I[0])/2)", false)

        expect :
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected

        where : 'The following input array, target derivative index and result scalar is used :'
            inputs | index || expected
            [ 3  ] | null  ||  3.049021713079475
            [ 2.5] | null  ||  3.507365283517986
            [ 0  ] | null  ||  0.2
            [ 0  ] | 0     ||  1.1
            [ 0.5] | 0     ||  0.646867884000033
            [ 1.6] | 0     ||  -0.00697440343353687
            [ -4 ] | 0     ||  3.9174193383745917
    }


    def 'Test scalar results of Function "sumjs((cos(I[j]*5)/5+I[j])*(1+sin(I[j])/2))" instance.'(
            double[] inputs, Integer index, double expected
    ){
        given :
            Function f = new FunctionBuilder( Neureka.get().backend() ).build("sumjs((cos(I[j]*5)/5+I[j])*(1+sin(I[j])/2))", false)

        expect :
            if (index!=null) assert f.derive( inputs, index )==expected
            else assert f.call( inputs )==expected

        where : 'The following input array, target derivative index and result scalar is used :'
            inputs                | index || expected
            [0.0, 0.5, 1.6, -4.0] | 0     ||  1.1
            [0.0, 0.5, 1.6, -4.0] | 1     ||  0.646867884000033
            [0.0, 0.5, 1.6, -4.0] | 2     ||  -0.00697440343353687
            [0.0, 0.5, 1.6, -4.0] | 3     ||  3.9174193383745917
    }


    def 'Test scalar results of various Function instances.'(
            String equation, double[] inputs, Integer index, double expected
    ){
        given : 'A new Function instance which is detached! (no autograd support)'
            Function f = new FunctionBuilder( Neureka.get().backend() ).build(equation, false)

        expect : 'Calling the function will yield the expected result.'
            if ( index!=null ) assert f.derive( inputs, index ) == expected
            else assert f.call( inputs )==expected

        where : 'The following parameters are used :'
            equation                         | inputs               | index || expected
            "6/2*(1+2)"                      | []                   | null  ||   9
            "sumJs(Ij)"                      | [2, 3.2, 6]          | null  ||  11.2
            "prod(Ij)"                       | [0.5, 0.5, 100]      | null  ||  25
            "prod(prod(Ij))"                 | [0.5, 0.5, 10]       | null  || (2.5 * 2.5 * 2.5)
            "I3/i[1]-I0+2+i2"                | [5, 4, 3, 12]        | null  ||   3
            "i3*i1/(i4-i0-2)-sig(0)+tanh(0)" | [-4, -2, 6, -3, -8]  | null  ||  -1.5
            "(i0*i1)*i2"                     | [2, 3, -2]           | 0     || -6
            "softplus(i0*i1)*i2"             | [2, 3, -2]           | 0     || -5.985164261060192
            "prod(ij)"                       | [2, 3, -2]           | 1     || -4
            "relu(prod(ij))"                 | [2, 3, -2]           | null  || -0.12
            "relu(prod(ij))"                 | [2, 3, -2]           | 1     || -0.04
            "quad(prod(ij)+6)"               | [2, 3, -2]           | null  || 36
            "quad(prod(ij)+6)"               | [2, 3, -2]           | 1     || -12*-4
            "quad(abs(prod(ij))-6)"          | [2, 3, -2]           | null  || 36
            "quad(abs(prod(ij))-6)"          | [2, 3, -2]           | 1     || -12*-4
            "sumJs(ij)"                      | [2, 3, -2]           | null  || 3
            "sumJs(ij)"                      | [2, 3, -2]           | 1     || 1
            "sumJs(ij^1)"                    | [2, 3, -2]           | null  || 3
            "sumJs(ij^1)"                    | [2, 3, -2]           | 1     || 1
            "I[1]^2"                         | [2, 3, -2]           | null  || 9
            "I[1]^2"                         | [2, 3, -2]           | 1     || 6
            "sumJs(ij^2)"                    | [2, 3, -2]           | null  || 17
            "sumJs(ij^2)"                    | [2, 3, -2]           | 1     || 6
            "2^I[1]"                         | [2, 3, -2]           | null  || 8
            "2^I[0]"                         | [2, 3, -2]           | null  || 4
            "2^I[2]"                         | [2, 3, -2]           | null  || 0.25
            "2^I[1]"                         | [2, 3, -2]           | 1     || 5.545177444479562
            "sumJs(2^I[j])"                  | [2, 3, -2]           | null  || 12.25
            "sumJs(2^I[j])"                  | [2, 3, -2]           | 1     || 5.545177444479562
            "I[1]%2"                         | [2, 3, -2]           | null  || 1
            "I[1]%2"                         | [2, 3, -2]           | 1     || 1
            "I[2]%2"                         | [2, 3, -5]           | null  || -1
            "I[2]%2"                         | [2, 3, -5]           | 2     || 1
            "I[2]%2"                         | [2, 3, -5]           | 1     || 0
            "7%I[1]"                         | [2, 4, -5]           | null  || 3
            "7%I[1]"                         | [2, 4, -5]           | 1     || 0 // This is in fact undefined... but let's keep things differentiable
    }


}
