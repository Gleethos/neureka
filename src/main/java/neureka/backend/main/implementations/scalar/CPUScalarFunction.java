package neureka.backend.main.implementations.scalar;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;

public class CPUScalarFunction implements ImplementationFor<CPU>
{
    private final ScalarFun _fun;

    public CPUScalarFunction(ScalarFun fun ) { _fun = fun; }

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        int d = call.getValOf(Arg.DerivIdx.class);
        CPUFun f = d < 0 ? _fun.getActivation() : _fun.getDerivative();
        double      in  = call.input( Number.class, 1 ).item(0).doubleValue();
        Tsr<Number> out = call.input( Number.class, 0 );
        Number result =  f.invoke(in);
        out.mut().setDataAt(0, result);
        return call.input(0);
    }

}
