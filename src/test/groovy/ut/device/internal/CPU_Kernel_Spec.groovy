package ut.device.internal

import neureka.Tsr
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.backend.main.operations.other.internal.CPUReduce
import neureka.devices.host.CPU
import spock.lang.Specification
import spock.lang.Subject

@Subject([CPUReduce])
class CPU_Kernel_Spec extends Specification
{

    def 'The Reduce implementation for the CPU has realistic behaviour'(
            CPUReduce.Type reduceType, Class<?> dataType, Number expected
    ) {
        given :
            var seed = dataType.getSimpleName().hashCode() + reduceType.name().hashCode()
            var a = Tsr.of(dataType)
                                .withShape(19, 7)
                                .andWhere({ i, _ -> ((seed+31**(i+13))%301)-151})
            var call = ExecutionCall.of(a).running(Mock(Operation)).on(CPU.get())
        when :
            var index = new CPUReduce(reduceType).run( call )
            var result = a.getItemAt(index.getItemAt(0) as int)
            println(result+"|"+a.items.min()+"|"+a.items.max())

        then :
            result == expected

        where :
            reduceType          | dataType | expected
            CPUReduce.Type.MIN  | Float    | -148.0
            CPUReduce.Type.MAX  | Float    |  141.0
            CPUReduce.Type.MIN  | Double   |  -143.0
            CPUReduce.Type.MAX  | Double   |  149.0
            CPUReduce.Type.MIN  | Integer  | -121
            CPUReduce.Type.MAX  | Integer  |  148
            CPUReduce.Type.MIN  | Long     | -146
            CPUReduce.Type.MAX  | Long     |  147
            CPUReduce.Type.MIN  | Short    | -148
            CPUReduce.Type.MAX  | Short    |  146
            CPUReduce.Type.MIN  | Byte     | -127
            CPUReduce.Type.MAX  | Byte     |  124
    }

}
