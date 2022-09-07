package neureka.backend.main.implementations.scalar;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.functions.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLScalarActivation implements ImplementationFor<OpenCLDevice>
{
    private final ScalarFun _fun;

    public CLScalarActivation(ScalarFun fun) {
        _fun = fun;
    }

    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        Fun.F64ToF64 f = call.getValOf(Arg.DerivIdx.class) < 0 ? _fun::activate : _fun::derive;
        Number value =  f.invoke(call.input( Number.class, 1 ).item(0).doubleValue());
        Tsr<Number> out = call.input( Number.class, 0 );
        out.getUnsafe().setDataAt(0, value);
        return call.input(0);
    }
}
