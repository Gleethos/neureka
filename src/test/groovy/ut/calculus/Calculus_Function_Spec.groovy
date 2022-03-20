package ut.calculus

import neureka.Neureka
import neureka.Tsr
import neureka.calculus.Function
import neureka.calculus.args.Args
import spock.lang.Specification

class Calculus_Function_Spec extends Specification {

    def 'Function implementations ensure that internally created tensors are flagged as "intermediate" initially!'()
    {
        given :
            var fun1 = new DummyFunction((Args args, Tsr<?>[] tensors) -> {
                                            var outputs = [Tsr.of(1)]
                                            tensors.length.times { outputs.add(tensors[it]) }
                                            return outputs[0]
                                        })
        and :
            var fun2 = new DummyFunction((Args args, Tsr<?>[] tensors) -> {
                                            var outputs = [Tsr.of(1)]
                                            tensors.length.times { outputs.add(tensors[it]) }
                                            return outputs[0].unsafe.setIsIntermediate(true)
                                        })
        and :
            var a = Tsr.of(3)
            var b = Tsr.of(-2.5)
        expect :
            !a.isIntermediate()
            !b.isIntermediate()

        when :
            caller(a, b, fun1)
        then :
            thrown(IllegalStateException)

        when :
            caller(a, b, fun2)
        then :
            noExceptionThrown()

        where :
            caller << [
                        {t1, t2, fun -> fun.call(t1, t2)},
                        {t1, t2, fun -> fun.invoke(t1, t2)},
                        {t1, t2, fun -> fun.execute(t1, t2)}
                    ]
    }

    def 'Function implementations ensure that outputs which are input members are not flagged as "intermediate"!'()
    {
        given :
            var fun1 = new DummyFunction((Args args, Tsr<?>[] tensors) -> tensors[0] )
        and :
            var fun2 = new DummyFunction((Args args, Tsr<?>[] tensors) -> {
                                    return tensors[0].unsafe.setIsIntermediate( true ) // This should fail!
                                })
        and :
            var a = Tsr.of(3)
            var b = Tsr.of(-2.5)
        expect :
            !a.isIntermediate()
            !b.isIntermediate()

        when :
            caller(a, b, fun1)
        then :
            noExceptionThrown()

        when :
            caller(a, b, fun2)
        then :
            thrown(IllegalStateException)

        where :
            caller << [
                        {t1, t2, fun -> fun.call(t1, t2)},
                        {t1, t2, fun -> fun.invoke(t1, t2)},
                        {t1, t2, fun -> fun.execute(t1, t2)}
                    ]
    }


    def 'Function implementations will ensure the "call" and "invoke" does not return tensors flagged as "intermediate".'()
    {
        given :
            var fun = new DummyFunction((Args args, Tsr<?>[] tensors) -> {
                                        return Tsr.of(42f).unsafe.setIsIntermediate(true)
                                    })

        and :
            var a = Tsr.of(3)
            var b = Tsr.of(-2.5)
        expect :
            !a.isIntermediate()
            !b.isIntermediate()

        when :
            var t1 = fun.call(a, b)
            var t2 = fun.invoke(a, b)
            var t3 = fun.execute(a, b)
        then :
            !t1.isIntermediate()
            !t2.isIntermediate()
            t3.isIntermediate()
    }

    def 'The library context exposes a set of useful functions.'(
            Function function
    ) {
        expect :
            function.toString() == expected
        where :
            function                                     || expected
            Neureka.get().backend.function.ln            || 'ln(I[0])'
            Neureka.get().backend.function.ln()          || 'ln(I[0])'
            Neureka.get().backend.function.gaus          || 'gaus(I[0])'
            Neureka.get().backend.function.gaus()        || 'gaus(I[0])'
            Neureka.get().backend.function.fastGaus      || 'fast_gaus(I[0])'
            Neureka.get().backend.function.fastGaus()    || 'fast_gaus(I[0])'
            Neureka.get().backend.function.sigmoid       || 'sig(I[0])'
            Neureka.get().backend.function.sigmoid()     || 'sig(I[0])'
            Neureka.get().backend.function.tanh          || 'tanh(I[0])'
            Neureka.get().backend.function.tanh()        || 'tanh(I[0])'
            Neureka.get().backend.function.fastTanh      || 'fast_tanh(I[0])'
            Neureka.get().backend.function.fastTanh()    || 'fast_tanh(I[0])'
            Neureka.get().backend.function.softsign || 'softsign(I[0])'
            Neureka.get().backend.function.softsign() || 'softsign(I[0])'
            Neureka.get().backend.function.softsign || 'softsign(I[0])'
            Neureka.get().backend.function.quad()        || 'quad(I[0])'
            Neureka.get().backend.function.quad          || 'quad(I[0])'
            Neureka.get().backend.function.relu()        || 'relu(I[0])'
            Neureka.get().backend.function.relu          || 'relu(I[0])'
            Neureka.get().backend.function.abs()         || 'abs(I[0])'
            Neureka.get().backend.function.abs           || 'abs(I[0])'
            Neureka.get().backend.function.sin()         || 'sin(I[0])'
            Neureka.get().backend.function.sin           || 'sin(I[0])'
            Neureka.get().backend.function.cos()         || 'cos(I[0])'
            Neureka.get().backend.function.cos           || 'cos(I[0])'
            Neureka.get().backend.function.softplus()    || 'softplus(I[0])'
            Neureka.get().backend.function.softplus      || 'softplus(I[0])'

    }

}
