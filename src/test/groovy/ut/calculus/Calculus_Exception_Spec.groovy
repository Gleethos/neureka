package ut.calculus

import neureka.Tsr
import neureka.calculus.Function
import spock.lang.Specification

class Calculus_Exception_Spec extends Specification {

    def 'Function throws exception when not enough inputs provided.'() {

        given :
            var fun = Function.of('I[0] + I[1]')
        and :
            var t = Tsr.of(4d)

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
            var t = Tsr.of(4d)

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
