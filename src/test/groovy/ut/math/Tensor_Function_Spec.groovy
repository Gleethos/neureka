package ut.math

import neureka.Neureka
import neureka.Tensor
import neureka.backend.ocl.CLBackend
import neureka.devices.Device
import neureka.math.Function
import neureka.view.NDPrintSettings
import spock.lang.*

import java.util.stream.Stream

@Title("Applying Functions to Tensors")
@Narrative('''

    A tensor would be nothing without being able to apply operations on them.
    However, calling operations manually in order to process your
    tensors can be a verbose and error prone task.
    This is where functions come into play.
    Neureka's functions are composed of operations forming an abstract syntax tree.
    Passing tensors to a function will route them trough this tree and apply
    all of the operations on the tensors for you.

''')
@Subject([Tensor, Function])
class Tensor_Function_Spec extends Specification
{
    def setupSpec()
    {
        reportHeader """ 
                This specification ensures that tensors supplied
                to functions are executed successfully and produce the expected results.
        """
    }

    def setup() {
        Neureka.get().reset()
        // Configure printing of tensors to be more compact:
        Neureka.get().settings().view().ndArrays({ NDPrintSettings it ->
            it.isScientific      = true
            it.isMultiline       = false
            it.hasGradient       = true
            it.cellSize          = 1
            it.hasValue          = true
            it.hasRecursiveGraph = false
            it.hasDerivatives    = true
            it.hasShape          = true
            it.isCellBound       = false
            it.postfix           = ""
            it.prefix            = ""
            it.hasSlimNumbers    = false
        })
    }

    def 'The tensor API has built-in methods for applying functions.'()
    {
        given : 'A simple scalar tensor containing the number "4".'
            var x = Tensor.of(4d)

        when: 'We use the following methods...'
            var sig  = x.sig()
            var tanh = x.tanh()
            var ln   = x.ln()
            var sin  = x.sin()
            var cos  = x.cos()
            var sfp  = x.softplus()
            var exp  = x.exp()
            var sqrt = x.sqrt()
            var abs  = x.abs()
            var neg  = x.neg()
            var cbrt = x.cbrt()
            var l10  = x.log10()
            var smax = x.softmax()
            var sigm = x.sigmoid()

        then: 'We get the expected results for each variable.'
            sig.toString()  == "(1):[0.98201]"
            tanh.toString() == "(1):[0.99932]"
            ln.toString()   == "(1):[1.38629]"
            sin.toString()  == "(1):[-0.75680]"
            cos.toString()  == "(1):[-0.65364]"
            sfp.toString()  == "(1):[4.01815]"
            exp.toString()  == "(1):[54.5982]"
            sqrt.toString() == "(1):[2.0]"
            abs.toString()  == "(1):[4.0]"
            neg.toString()  == "(1):[-4.0]"
            cbrt.toString() == "(1):[1.5874]"
            l10.toString()  == "(1):[0.60205]"
            smax.toString() == "(1):[1.0]"
            sigm.toString() == "(1):[0.98201]"
    }

    def 'The softmax function can be applied to tensors with more than one dimension.'()
    {
        given : 'A tensor with more than one dimension.'
            var x = Tensor.of( -3f..7f ).reshape( 2, 3 )

        when: 'We apply the softmax function to it.'
            var softmax = x.softmax()

        then: 'We get the expected results.'
            softmax.toString() == "(2x3):[0.00426, 0.01160, 0.03154, 0.08576, 0.23312, 0.63369]"
        and : 'The resulting values have the property we expect from softmax: their sum is 1!'
            softmax.sum().item() == 1.0
    }

    def 'The softmax can be calculated for a particular axis.'()
    {
        given : 'A tensor with more than one dimension.'
            var x = Tensor.of( -3f..7f ).reshape( 2, 3 )

        when: 'We apply the softmax function to it.'
            var softmax = x.softmax(1)

        then: 'We get the expected results.'
            softmax.toString() == "(2x3):[0.09003, 0.24472, 0.66524, 0.09003, 0.24472, 0.66524]"
        and : 'The resulting values have the property we expect from softmax: their sum is 1!'
            softmax.sum(1).items.collect({it.round 10}) == [1.0, 1.0]

        when :
            softmax = x.softmax(0)
        then :
            softmax.toString() == "(2x3):[0.04742, 0.04742, 0.04742, 0.95257, 0.95257, 0.95257]"
        and : 'The resulting values have the property we expect from softmax: their sum is 1!'
            softmax.sum(0).items.collect({it.round 10}) == [1.0, 1.0, 1.0]
    }

    def 'The softmax can be calculated alongside multiple axes.'()
    {
        given : 'A simple 2 by 3 by 4 matrix.'
            var m = Tensor.of(-79f..43f).reshape(2, 3, 2, 4)
        when : 'We create a softmax for every axis...'
            var s = m.softmax(1, 2)
        then : 'The resulting tensor will have the expected shape.'
            s.shape() == [2, 3, 2, 4]
        and : 'Using the sum function we can verify that the softmax is correct.'
            s.sum(1, 2).items.collect({it.round 10}) == [1, 1, 1, 1, 1, 1, 1, 1]
    }

    def 'The optimization function for the SGD algorithm produces the expected result'()
    {
        given : 'We use a common learning rate.'
            var learningRate = 0.01
        and : 'Based on that we instantiate the SGD optimization inline function.'
            var fun = Function.of("I[0] <- (I[0] * -$learningRate)")
        and : 'A tensor, which will be treated as gradient.'
            var g = Tensor.of(1.0)

        when : 'We apply the function to the gradient...'
            var result = fun(g)

        then : 'Both the result tensor and the gradient will have the expected value.'
            result.toString() == "(1):[-0.01]"
            g.toString() == "(1):[-0.01]"
        and : 'The result will be identical to the gradient, simply because its an inline function.'
            result === g
    }

    @IgnoreIf({ data.device == 'GPU' && !Neureka.get().canAccessOpenCLDevice() }) // We need to assure that this system supports OpenCL!
    def 'Tensor results of various Function instances return expected results.'(
            String equation, List<Tensor> inputs, Integer index, Map<List<Integer>,List<Double>> expected
    ) {
        given : 'We set the experimental "autoConvertToFloat" flag to true.'
            Neureka.get().backend().find(CLBackend).ifPresent({ it.settings.autoConvertToFloat=true })

        and : "A new Function instance created from ${equation}."
            Function f = Function.of(equation, true) // TODO : test with 'doAD' : false!
        and :
            inputs.each {it.to(Device.get(device))}

        and : 'The result is being calculated by invoking the Function instance.'
            Tensor<?> result = ( index != null ? f.derive( inputs, index ) : f.call( inputs ) )
            List<Double> value = result.getItemsAs(double[].class) as List<Double>

        expect : "The calculated result ${result} should be (ruffly) equal to expected ${expected}."
            (0..<value.size()).every {equals(value[it], expected.values().first()[it], 1e-6)}

        and : 'The shape is as expected as well : '
            result.shape == expected.keySet().first()

        // Todo: unrecognized operation throws exception that is not recursion error
        where :
            device | equation                         | inputs                                                           | index || expected
            'CPU'  | "quad(sumJs(Ij))"                | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -5d])]                           | null || [[2]:[16d, 9d]]
            'GPU'  | "quad(sumJs(Ij))"                | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -5d])]                           | null || [[2]:[16d, 9d]]
            'CPU'  | "tanh(sumJs(Ij))"                | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | null || [[2]:[0.9993292997390673, -0.9640275800758169]]
            'GPU'  | "tanh(sumJs(Ij))"                | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | null || [[2]:[0.9993293285369873, -0.9640275835990906]]
            'CPU'  | "tanh(i0*i1)"                    | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | 0    || [[2]:[0.0295981114963193, -1.8005623907413337E-6]]
            'GPU'  | "tanh(i0*i1)"                    | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | 0    || [[2]:[0.029597997665405273, -1.9073486328125E-6]]
            'CPU'  | "fast_tanh(i0*i1)"               | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | null || [[2]:[0.9486832980505138, -0.9922778767136677]]
            'CPU'  | "fast_tanh(i0*i1)"               | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | 0    || [[2]:[0.09486832980505137, -0.00763290674395129]]
            'CPU'  | "softsign(i0*i1)"                | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | null || [[2]:[0.75, -0.8888888888888888]]
            'CPU'  | "softsign(i0*i1)"                | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | 0    || [[2]:[0.1875, -0.04938271604938271]]
            'CPU'  | "fast_gaus(i0/i1)"               | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | null || [[2]:[0.8999999999999999, 0.8]]
            'CPU'  | "fast_gaus(i0/i1)"               | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | 0    || [[2]:[-0.18, -0.16]]
            'CPU'  | "softplus(prodJs(Ij-2))"         | [Tensor.of([2],[1d, 2d]), Tensor.of([2],[3d, -4d])]                           | null || [[2]:[0.31326168751822286, 0.6931471805599453]]
            'CPU'  | "softplus([-1, 0, -2, -2](Ij-2))"| [Tensor.of([2, 4], [10d, 12d, 16d, 21d, 33d, 66d, 222d, 15d])]                | null || [[1, 2, 2, 2]:[8.000335406372896, 10.000045398899218, 14.000000831528373, 19.000000005602796, 31.000000000000036, 64.0, 220.0, 13.000002260326852]]
            'CPU'  | "softplus(i0*i1)*i2"             | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[-0.0018221023888012908, 0.2845552390654007]]
            'CPU'  | "sumJs(ij**3)"                   | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[(-1+7*7*7+2*2*2), (3*3*3+-1+2*2*2)]]
            'CPU'  | "sumJs(ij**3)"                   | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[3*Math.pow(7, 2), 3*Math.pow(-1, 2)]]
            'CPU'  | "sumJs(ij*ij)"                   | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[(1+7*7+2*2), (3*3+1+2*2)]]
            'CPU'  | "sumJs(ij*ij)"                   | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[2*Math.pow(7, 1), 2*Math.pow(-1, 1)]]
            'CPU'  | "sumJs(ij/2)"                    | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[4, 2]]
            'CPU'  | "sumJs(ij/2)"                    | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[0.5, 0.5]]
            'CPU'  | "sumJs(ij+2)"                    | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[(1+9+4), (5+1+4)]]
            'CPU'  | "sumJs(ij+2)"                    | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[1.0, 1.0]]
            'CPU'  | "sumJs(ij-2)"                    | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[(-3+5+0), (1+-3+0)]]
            'CPU'  | "sumJs(ij-2)"                    | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[1.0, 1.0]]
            'CPU'  | "sumJs(sumJs(ij))"               | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[8.0*3.0, 4.0*3.0]]
            'CPU'  | "sumJs(sumJs(ij))"               | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[3.0, 3.0]]
            'CPU'  | "sumJs(prodJs(ij))"              | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[(-14)*3, (-6)*3]]
            'CPU'  | "(prodJs(ij))"                   | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[-2.0, 6.0]]
            'CPU'  | "-(prodJs(ij))%3"                | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | null || [[2]:[-(-14)%3, -0.0]]
            'CPU'  | "sumJs(prodJs(ij))"              | [Tensor.of([2],[-1d, 3d]), Tensor.of([2],[7d, -1d]), Tensor.of([2],[2d, 2d])] | 1    || [[2]:[-2.0*3, 6.0*3]]
            'CPU'  | "relu(I[0])"                     | [Tensor.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                               | null || [[2, 3]:[-0.04, 7.0, -0.01, 2.0, 3.0, 8.0]]
            'CPU'  | "relu(I[0])"                     | [Tensor.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                               |  0   || [[2, 3]:[0.01, 1.0, 0.01, 1.0, 1.0, 1.0]]
            'CPU'  | "quad(I[0])"                     | [Tensor.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                               | null || [[2, 3]:[16.0, 49.0, 1.0, 4.0, 9.0, 64.0]]
            'CPU'  | "quad(I[0])"                     | [Tensor.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                               |  0   || [[2, 3]:[-8.0, 14.0, -2.0, 4.0, 6.0, 16.0]]
            'CPU'  | "abs(I[0])"                      | [Tensor.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                               | null || [[2, 3]:[4.0, 7.0, 1.0, 2.0, 3.0, 8.0]]
            'CPU'  | "abs(I[0])"                      | [Tensor.of([2, 3], [-4d, 7d, -1d, 2d, 3d, 8d])]                               |  0   || [[2, 3]:[-1.0, 1.0, -1.0, 1.0, 1.0, 1.0]]
            'CPU'  | "dimtrim(I[0])"                  | [Tensor.of([1, 3, 1],    [1d, 2d, 3d])]                                       | null || [[3]:[1, 2, 3]]
            'CPU'  | "dimtrim(I[0])"                  | [Tensor.of([1, 3, 1, 1], [-4d, 2d, 5d])]                                      | null || [[3]:[-4, 2, 5]]
            'CPU'  | "ln(i0)"                         | [Tensor.of(3d)]                                                               | null || [[1]:[Math.log(3)]]
            'CPU'  | "ln(i0)"                         | [Tensor.of(3d)]                                                               |  0   || [[1]:[0.3333333333333333]]
            'CPU'  | "selu(I[0])"                     | [Tensor.ofDoubles().withShape(3).all(-1)]                                     | null || [[3]:[-1.1113307378125625, -1.1113307378125625, -1.1113307378125625]]
            'GPU'  | "selu(I[0])"                     | [Tensor.ofDoubles().withShape(3).all(-1)]                                     | null || [[3]:[-1.1113307476043701, -1.1113307476043701, -1.1113307476043701]]
    }


    def 'Reshaping on 3D tensors works by instantiate a Function instance built from a String.'()
    {
        given :
            Neureka.get().settings().view().getNDPrintSettings().setIsLegacy(true)
            var f = Function.of("[2, 0, 1]:(I[0])")

        when : var t = Tensor.of([3, 4, 2], 1d..5d)
        then : t.toString().contains("[3x4x2]:(1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0)")

        when : var r = f(t)
        then :
            r.toString().contains("[2x3x4]")
            r.toString().contains("[2x3x4]:(1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 5.0, 2.0, 4.0)")
    }


    def 'The "DimTrim" operation works forward as well as backward!'()
    {
        reportInfo """
            The "DimTrim" operation used to trim the padding of a tensor shape,
            which are dimensions with a size of 1.
            So a shape like [1, 3, 1] would be trimmed to [3].
        """

        given :
            var t = Tensor.of([1, 1, 3, 2, 1], 8d).setRqsGradient(true)

        when :
            var trimmed = Function.of("dimtrim(I[0])")(t)

        then :
            trimmed.toString().contains("(3x2):[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]; ->d(")

        when :
            var back = trimmed.backward()

        then :
            back == trimmed
        and :
            t.gradient.get().toString() == "(1x1x3x2x1):[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]"
    }

    def 'Executed tensors are intermediate tensors.'() 
    {
        reportInfo """
            Functions expose different kinds of methods for different kinds of
            purposes, however there is one species of methods with a very important role
            in ensuring memory efficiency.
            These types of methods are the `execute` methods which 
            distinguish themselves in that the tensors returned by 
            these methods are flagged as "intermediate".
            If a tensor is an intermediate one, it becomes eligible 
            for deletion when consumed by another function.
            Note that internally every function is usually a composite
            of other functions forming a syntax tree which will process
            input tensors through the execute methods, which causes
            intermediate results to be deleted automatically.
            When executing a function as a user of Neureka
            one should generally avoid using the `execute` method in order to avoid
            accidental deletion of results.
            This is mostly relevant for when designing custom operations.
        """
        given : 'We create a simple function taking one input.'
            var fun = Function.of('i0 * relu(i0) + 1')
        and : 'A vector tensor of 5 float numbers'
            var t = Tensor.of(1f, -5f, -3f, 2f, 8f)
        expect : 'Both the tensor as well as the function were created successfully.'
            t.itemType == Float
            fun.toString() == "((I[0] * relu(I[0])) + 1.0)"

        when : 'We try different kinds of ways of passing the tensor to the function...'
            var result1 = fun.call(t)
            var result2 = fun.invoke(t)
            var result3 = fun.execute(t)

        then : 'The "call" method will not return an intermediate result.'
            !result1.isIntermediate()
        and : 'The functionally identical synonym method "invoke" will also yield a non-intermediate result.'
            !result2.isIntermediate()
        and : 'As expected, the tensor of the "execute" method is indeed intermediate.'
            result3.isIntermediate()
        and : 'Otherwise all 3 tensors are basically the same.'
            result1.toString() == "(5):[2.0, 1.25, 1.09, 5.0, 65.0]"
            result2.toString() == "(5):[2.0, 1.25, 1.09, 5.0, 65.0]"
            result3.toString() == "(5):[2.0, 1.25, 1.09, 5.0, 65.0]"
    }


    def 'We can collect a stream into a tensor.'()
    {
        given : 'We create a stream of integers.'
            var stream = Stream.of(1, 2, 3, 4, 5, 6)

        when : 'We collect the stream into a tensor.'
            var t = stream.collect(Tensor.shaped(2, 3))

        then : 'The resulting tensor should have the same values as the stream.'
            t.toString() == "(2x3):[1, 2, 3, 4, 5, 6]"
    }


    private static boolean equals(Number v1, Number v2, delta) {
        return (v1 == v2) || (v1 - v2).abs() <= delta
    }

}
