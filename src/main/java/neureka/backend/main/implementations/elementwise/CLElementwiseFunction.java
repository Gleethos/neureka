package neureka.backend.main.implementations.elementwise;

import neureka.Neureka;
import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.backend.main.implementations.fun.api.ScalarFun;
import neureka.calculus.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLElementwiseFunction extends ParsedCLImplementation
{
    public CLElementwiseFunction( ScalarFun fun )
    {
        super(
            CLElementwiseFunction::_run,
            2,
            Neureka.get().utility().readResource("kernels/activation_template.cl"),
            fun.activationCode(),
            fun.derivationCode(),
            fun.id()
        );
    }

    private static Tsr<?> _run( ExecutionCall<OpenCLDevice> call )
    {
        int offset = call.input( Number.class, 0 ) != null ? 0 : 1;
        int gwz = call.input( Number.class, 0 ) != null ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size();
        // Drain tensor needs to be 'actual'! :
        if ( !call.input( Number.class, offset + 1).isVirtual() ) call.input( Number.class, offset).setIsVirtual( false );
        call.getDevice()
                .getKernel(call)
                .passAllOf( call.input( Number.class, offset ) )
                .passAllOf( call.input( Number.class, offset + 1 ) )
                .pass( call.input( Number.class, 0 ).rank() )
                .pass( call.getValOf( Arg.DerivIdx.class ) )
                .call( gwz );

        return call.input( 0 );
    }

}
