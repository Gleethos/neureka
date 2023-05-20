package ut.math

import neureka.Neureka
import neureka.Tensor
import neureka.math.Functions
import neureka.math.args.Args
import spock.lang.Narrative
import spock.lang.Specification
import spock.lang.Subject
import spock.lang.Title

import java.util.function.Function

@Title("Testing Default Methods on Functions")
@Narrative('''

    This specification tests the default methods on functions
    through a simple dummy implementation of the Function interface.

''')
@Subject([Function])
class Function_Spec extends Specification
{

    def 'Function implementations ensure that internally created tensors are flagged as "intermediate" initially!'()
    {
        given :
            var fun1 = new DummyFunction((Args args, Tensor<?>[] tensors) -> {
                                            var outputs = [Tensor.of(1d)]
                                            tensors.length.times { outputs.add(tensors[it]) }
                                            return outputs[0]
                                        })
        and :
            var fun2 = new DummyFunction((Args args, Tensor<?>[] tensors) -> {
                                            var outputs = [Tensor.of(1d)]
                                            tensors.length.times { outputs.add(tensors[it]) }
                                            return outputs[0].mut.setIsIntermediate(true)
                                        })
        and :
            var a = Tensor.of(3d)
            var b = Tensor.of(-2.5)
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

    def 'Function implementations ensure that outputs which are input members are not flagged as "intermediate"!'(
        Closure caller
    ) {
        given :
            var fun1 = new DummyFunction((Args args, Tensor<?>[] tensors) -> tensors[0] )
        and :
            var fun2 = new DummyFunction((Args args, Tensor<?>[] tensors) -> {
                                    return tensors[0].mut.setIsIntermediate( true ) // This should fail!
                                })
        and :
            var a = Tensor.of(3.0)
            var b = Tensor.of(-2.5)
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
            var fun = new DummyFunction((Args args, Tensor<?>[] tensors) -> {
                                        return Tensor.of(42f).mut.setIsIntermediate(true)
                                    })

        and :
            var a = Tensor.of(3d)
            var b = Tensor.of(-2.5)
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
            String expected,
            Function<Functions, neureka.math.Function> fun
    ) {
        expect :
            fun.apply(Neureka.get().backend.function).toString() == expected
            !fun.apply(Neureka.get().backend.function).isDoingAD()
        and :
            fun.apply(Neureka.get().backend.autogradFunction).toString() == expected
            fun.apply(Neureka.get().backend.autogradFunction).isDoingAD()
        where :
             expected          || fun
             'ln(I[0])'        || {Functions it -> it.ln            }
             'ln(I[0])'        || {Functions it -> it.ln()          }
             'gaus(I[0])'      || {Functions it -> it.gaus          }
             'gaus(I[0])'      || {Functions it -> it.gaus()        }
             'fast_gaus(I[0])' || {Functions it -> it.fastGaus      }
             'fast_gaus(I[0])' || {Functions it -> it.fastGaus()    }
             'sig(I[0])'       || {Functions it -> it.sigmoid       }
             'sig(I[0])'       || {Functions it -> it.sigmoid()     }
             'tanh(I[0])'      || {Functions it -> it.tanh          }
             'tanh(I[0])'      || {Functions it -> it.tanh()        }
             'fast_tanh(I[0])' || {Functions it -> it.fastTanh      }
             'fast_tanh(I[0])' || {Functions it -> it.fastTanh()    }
             'softsign(I[0])'  || {Functions it -> it.softsign      }
             'softsign(I[0])'  || {Functions it -> it.softsign()    }
             'softsign(I[0])'  || {Functions it -> it.softsign      }
             'quad(I[0])'      || {Functions it -> it.quad()        }
             'quad(I[0])'      || {Functions it -> it.quad          }
             'relu(I[0])'      || {Functions it -> it.relu()        }
             'relu(I[0])'      || {Functions it -> it.relu          }
             'abs(I[0])'       || {Functions it -> it.abs()         }
             'abs(I[0])'       || {Functions it -> it.abs           }
             'sin(I[0])'       || {Functions it -> it.sin()         }
             'sin(I[0])'       || {Functions it -> it.sin           }
             'cos(I[0])'       || {Functions it -> it.cos()         }
             'cos(I[0])'       || {Functions it -> it.cos           }
             'softplus(I[0])'  || {Functions it -> it.softplus()    }
             'softplus(I[0])'  || {Functions it -> it.softplus      }
             'silu(I[0])'      || {Functions it -> it.silu()        }
             'silu(I[0])'      || {Functions it -> it.silu          }
             'gelu(I[0])'      || {Functions it -> it.gelu()        }
             'gelu(I[0])'      || {Functions it -> it.gelu          }
             'selu(I[0])'      || {Functions it -> it.selu()        }
             'selu(I[0])'      || {Functions it -> it.selu          }

    }

}
