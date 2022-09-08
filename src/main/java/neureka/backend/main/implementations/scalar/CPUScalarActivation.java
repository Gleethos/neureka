package neureka.backend.main.implementations.scalar;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.algorithms.Functions;
import neureka.backend.main.functions.CPUFun;
import neureka.backend.main.functions.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;

public class CPUScalarActivation implements ImplementationFor<CPU>
{
    private final ImplementationFor<CPU> _impl;
    private final ScalarFun _fun;

    public CPUScalarActivation(ScalarFun fun) {
        _impl = Functions.implementation( 2, call -> 1, (call, funs)->this._workloadFor(call) ).get();

        _fun = fun;
    }

    private CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call
    ) {
        int d = call.getValOf(Arg.DerivIdx.class);
        CPUFun f = d < 0 ? _fun.getActivation() : _fun.getDerivative();
        return (i, end) -> {
            double      in  = call.input( Number.class, 1 ).item(0).doubleValue();
            Tsr<Number> out = call.input( Number.class, 0 );
            Number result =  f.activate(in);
            out.getUnsafe().setDataAt(0, result);
        };
    }

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        return _impl.run(call);
    }

}
