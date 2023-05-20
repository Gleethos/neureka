package ut.math

import neureka.Tensor
import neureka.math.Function
import spock.lang.Specification

class Function_Exception_Spec extends Specification {

    def 'Function throws exception when not enough inputs provided.'() {

        given :
            var fun = Function.of('I[0] + I[1]')
        and :
            var t = Tensor.of(4d)

        when :
            fun(t)

        then :
            var exception = thrown(IllegalArgumentException)
        and :
            exception.message == "Function input '1' not satisfied! Please supply at least 2 input tensors."
    }


    def 'Function throws exception when arity does not match input number.'() {

        given :
            var fun = Function.of('I[0] @ I[1]')
        and :
            var t = Tensor.of(4d)

        when :
            fun(t)

        then :
            var exception = thrown(IllegalArgumentException)
        and :
            exception.message == "Trying to instantiate an 'ExecutionCall' " +
                                 "with an arity of 1, which is not suitable for " +
                                 "the targeted operation 'MatMul' with the expected " +
                                 "arity of 2."
    }


}
