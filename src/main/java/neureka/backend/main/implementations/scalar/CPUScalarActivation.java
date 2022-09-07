package neureka.backend.main.implementations.scalar;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.api.ImplementationFor;
import neureka.backend.main.algorithms.Functions;
import neureka.backend.main.algorithms.internal.Fun;
import neureka.backend.main.functions.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.host.CPU;

public class CPUScalarActivation implements ImplementationFor<CPU>
{
    private final ImplementationFor<CPU> _impl;

    public CPUScalarActivation(ScalarFun fun) {
        _impl = Functions.implementation( 2, call -> 1, CPUScalarActivation::_workloadFor )
                .with(Fun.F64ToF64.pair(fun::activate, fun::derive))
                .with(Fun.F32ToF32.pair(fun::activate, fun::derive))
                .with(Fun.I32ToI32.pair(fun::activate, fun::derive))
                .with(Fun.I64ToI64.pair(fun::activate, fun::derive))
                .with(Fun.I8ToI8.pair(fun::activate, fun::derive))
                .with(Fun.I16ToI16.pair(fun::activate, fun::derive))
                .with(Fun.BoolToBool.pair(fun::activate, fun::derive))
                .with(Fun.CharToChar.pair(fun::activate, fun::derive))
                .get();
    }

    private static CPU.RangeWorkload _workloadFor(
            ExecutionCall<CPU> call,
            Functions<Fun> functions
    ) {
        return (i, end) -> {
            double      in  = call.input( Number.class, 1 ).item(0).doubleValue();
            Tsr<Number> out = call.input( Number.class, 0 );
            Number result =  functions.get(Fun.F64ToF64.class).get( call.get( Arg.DerivIdx.class ) ).invoke(in);
            out.getUnsafe().setDataAt(0, result);
        };
    }

    @Override
    public Tsr<?> run(ExecutionCall<CPU> call) {
        return _impl.run(call);
    }

}
