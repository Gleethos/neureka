package ut.tensors

import neureka.Tsr
import neureka.calculus.Function
import neureka.devices.Device
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

@Title("Tensor Number Statistics")
@Subject([Tsr])
class Tensor_Stats_Spec extends Specification
{
    def 'We can use the max operation as a function'(
            String reduceType, Class<?> dataType, String device, Number expected
    ) {
        given : 'We create a max function:'
            var fun = Function.of(reduceType.toLowerCase() + "(I[0])")
        and :
            var seed = dataType.getSimpleName().hashCode() + reduceType.hashCode()
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

}
