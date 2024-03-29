package neureka.backend.main.implementations.scalar;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.math.args.Arg;
import neureka.devices.host.CPU;

public class CPUScalarFunction implements ImplementationFor<CPU>
{
    private final ScalarFun _fun;

    public CPUScalarFunction(ScalarFun fun ) { _fun = fun; }

    @Override
    public Tensor<?> run(ExecutionCall<CPU> call) {
        int d = call.getValOf(Arg.DerivIdx.class);
        CPUFun f = d < 0 ? _fun.getActivation() : _fun.getDerivative();
        double      in  = call.input( Number.class, 1 ).item(0).doubleValue();
        Tensor<Number> out = call.input( Number.class, 0 );
        Number result =  f.invoke(in);
        out.mut().setDataAt(0, result);
        return call.input(0);
    }

}
