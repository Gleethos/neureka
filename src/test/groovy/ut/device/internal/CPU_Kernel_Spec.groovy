package ut.device.internal


import neureka.Tensor
import neureka.backend.api.ExecutionCall
import neureka.backend.api.Operation
import neureka.backend.main.operations.other.internal.CPUReduce
import neureka.backend.main.operations.other.internal.CPUSum
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
            var a = Tensor.of(dataType)
                                .withShape(19, 7)
                                .andWhere({ i, _ -> ((seed+31**(i+13))%301)-151})
            var call = ExecutionCall.of(a).running(Mock(Operation)).on(CPU.get())
        when :
            var index = new CPUReduce(reduceType).run( call )
            var result = a.item(index.item() as int)

        then :
            result == expected

        where :
            reduceType          | dataType || expected
            CPUReduce.Type.MIN  | Float    || -148.0
            CPUReduce.Type.MAX  | Float    ||  141.0
            CPUReduce.Type.MIN  | Double   ||  -143.0
            CPUReduce.Type.MAX  | Double   ||  149.0
            CPUReduce.Type.MIN  | Integer  || -121
            CPUReduce.Type.MAX  | Integer  ||  148
            CPUReduce.Type.MIN  | Long     || -146
            CPUReduce.Type.MAX  | Long     ||  147
            CPUReduce.Type.MIN  | Short    || -148
            CPUReduce.Type.MAX  | Short    ||  146
            CPUReduce.Type.MIN  | Byte     || -127
            CPUReduce.Type.MAX  | Byte     ||  124
    }

    def 'The Sum implementation for the CPU has realistic behaviour'(
            Class<?> dataType, Number expected
    ) {
        given :
            var seed = dataType.getSimpleName().hashCode()
            var a = Tensor.of(dataType)
                                .withShape(19, 7)
                                .andWhere({ i, _ -> ((seed+31**(i+13))%301)-151})
            var call = ExecutionCall.of(a).running(Mock(Operation)).on(CPU.get())
        when :
            var result = new CPUSum().run( call )

        then :
            result.item() == expected
            result.item() == result.items.stream().reduce(0, (x, y) -> x + y)

        where :
            dataType   || expected
            Float      || -1222.0
            Double     || -1026.0
            Integer    || -2251
            Long       || -2083
            Short      ||  1018
            Byte       ||  34
            BigInteger ||  3930
            BigDecimal || -1502.0
    }

}
