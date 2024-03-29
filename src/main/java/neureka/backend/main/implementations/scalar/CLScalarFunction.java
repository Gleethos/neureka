package neureka.backend.main.implementations.scalar;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.implementations.fun.api.CPUFun;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.math.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLScalarFunction implements ImplementationFor<OpenCLDevice>
{
    private final ScalarFun _fun;

    public CLScalarFunction(ScalarFun fun) {
        _fun = fun;
    }

    @Override
    public Tensor<?> run(ExecutionCall<OpenCLDevice> call) {
        int d = call.getValOf(Arg.DerivIdx.class);
        CPUFun f = d < 0 ? _fun.getActivation() : _fun.getDerivative();
        Number value =  f.invoke(call.input( Number.class, 1 ).item(0).doubleValue());
        Tensor<Number> out = call.input( Number.class, 0 );
        out.mut().setDataAt(0, value);
        return call.input(0);
    }
}
