package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import spock.lang.IgnoreIf
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor Number Statistics")
@Subject([Tsr])
class Tensor_Stats_Spec extends Specification
{
    @IgnoreIf({ data.device == "GPU" && !Neureka.get().canAccessOpenCLDevice() })
    def 'We can use the max operation as a function'(
            String reduceType, Class<?> dataType, String device, Number expected
    ) {
        given : 'We create a max function:'
            var fun = Function.of(reduceType.toLowerCase() + "(I[0])")
        and : 'A seed, for some variability:'
            var seed = dataType.getSimpleName().hashCode() + reduceType.hashCode()
        and :
            var a = Tsr.of(dataType)
                                .withShape(19, 7)
                                .andWhere({ i, _ -> ((seed+31**(i+13))%301)-151})
        and :
            a.to(device)

        when : 'We apply the function to the tensor:'
            var result = fun(a)
        then : 'The result is correct:'
            result.items[0] == expected

        where :
            reduceType  | dataType | device || expected
            'MIN'       | Float    | 'CPU'  || -148.0
            'MAX'       | Float    | 'CPU'  ||  141.0
            'MIN'       | Double   | 'CPU'  ||  -143.0
            'MAX'       | Double   | 'CPU'  ||  149.0
            'MIN'       | Integer  | 'CPU'  || -121
            'MAX'       | Integer  | 'CPU'  ||  148
            'MIN'       | Long     | 'CPU'  || -146
            'MAX'       | Long     | 'CPU'  ||  147
            'MIN'       | Short    | 'CPU'  || -148
            'MAX'       | Short    | 'CPU'  ||  146
            'MIN'       | Byte     | 'CPU'  || -127
            'MAX'       | Byte     | 'CPU'  ||  124
            'MIN'       | Float    | 'GPU'  || -148.0
            'MAX'       | Float    | 'GPU'  ||  141.0
    }

    def 'We can get pre-instantiated min and max functions from the library context.'()
    {
        given : 'We access the pre-instantiated max function:'
            var min = Neureka.get().backend.autogradFunction.min
        and : 'We access the pre-instantiated min function:'
            var max = Neureka.get().backend.autogradFunction.max

        expect :
            min.toString() == "min(I[0])"
            max.toString() == "max(I[0])"
    }

    @IgnoreIf({ data.device == "GPU" && !Neureka.get().canAccessOpenCLDevice() })
    def 'There is no need to use a function, we can use the min() and max() methods on tensors instead.'(
        String device
    ) {
        given : 'We create a tensor:'
            var a = Tsr.of(Float)
                                .withShape(19, 7)
                                .andWhere({ i, _ -> ((31**(i+42))%301)-151})
        and :
            a.to(device)

        and : 'We access the min and max methods:'
            var min = a.min()
            var max = a.max()

        expect : 'The results are correct:'
            min.item(0) == -150.0
            max.item(0) == 147.0

        where :
            device << ['CPU', 'GPU']
    }

    def 'Both the min and max operation support autograd (back-propagation).'()
    {
        given : 'We create a simple tensor of floats which requires gradients:'
            var a = Tsr.of(-3f, 6.42f, 2.065f, -8f, 0.2, 7.666f, 3.39f).setRqsGradient(true)
        and : 'We first do a simple operation to get another tensor:'
            var b = a * 3
        and : 'We then do min and max operations as well as 2 divisions:'
            var x = b.min() / 2
            var y = b.max() / 4

        when : 'We back-propagate both paths by adding them and calling backward:'
            (x+y).backward()

        then : 'The gradient is correct:'
            a.gradient.items == [0.0, 0.0, 0.0, 1.5, 0.0, 0.75, 0.0]
    }

}
