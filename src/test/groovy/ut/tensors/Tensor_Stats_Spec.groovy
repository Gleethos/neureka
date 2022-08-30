package ut.tensors

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import spock.lang.IgnoreIf
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Reducing Tensors")
@Narrative('''

    Various kinds of operations reduce tensors to scalars,
    the most common ones being the min and max operations 
    which find the smallest as well as largest number among all 
    items of a tensor.
    Neureka exposes various different ways to achieve this,
    all of which are also differential (autograd support).

''')
@Subject([Tsr])
class Tensor_Stats_Spec extends Specification
{
    @IgnoreIf({ data.device == "GPU" && !Neureka.get().canAccessOpenCLDevice() })
    def 'We can use the max operation as a function'(
            String reduceType, Class<?> dataType, String device, Number expected
    ) {
        given : 'We create a min/max function:'
            var fun = Function.of(reduceType.toLowerCase() + "(I[0])")
        and : 'A seed, for some variability:'
            var seed = dataType.getSimpleName().hashCode() + reduceType.hashCode()
        and :
            var a = Tsr.of(dataType)
                                .withShape(19, 7)
                                .andWhere({ i, _ -> ((seed+31**(i+13))%301)-151})
        and : 'Before applying the function, we copy the tensor to the device:'
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

        expect : 'The 2 functions are indeed min and max'
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
        and : 'Before applying the function, we copy the tensor to the device:'
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
            a.gradient.items == [0f, 0f, 0f, 1.5f, 0f, 0.75f, 0f]
    }

    def 'We can use the "sum" method to sum the items of a tensor.'()
    {
        given : 'We create a tensor:'
            var a = Tsr.of(Float)
                                .withShape(13, 73, 11)
                                .andWhere({ i, _ -> ((7**i) % 11)-5})
        and : 'We sum the items of the tensor:'
            var sum = a.sum()

        expect : 'The result is correct:'
            sum.item() == 5217.0
        and : 'The result can be verified using other methods:'
            sum.item() == sum.items.stream().reduce(0,(x,y)->x+y)
            sum.item() == a.unsafe.data.ref.sum()
    }

    def 'The sum operation support autograd (back-propagation).'() {
        given : 'We create a simple tensor of floats which requires gradients:'
            var a = Tsr.of(1f, 2f, 3f, 4f).setRqsGradient(true)
        and : 'We first do a simple operation to get another tensor:'
            var b = a * 3
        and : 'We then do a sum operation:'
            var x = b.sum()

        when : 'We back-propagate the path by calling backward:'
            x.backward()

        then : 'The gradient is correct:'
            a.gradient.items == [3f, 3f, 3f, 3f]
    }

}
