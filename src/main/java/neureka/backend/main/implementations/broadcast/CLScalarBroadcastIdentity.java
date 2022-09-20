package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLScalarBroadcastIdentity extends CLScalarBroadcast
{
    public CLScalarBroadcastIdentity(String id) {
        super(
             id,
             "output = value;\n",
             "output = value;\n"
        );
    }

    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        Tsr<Number> t = call.input( Number.class, 0 );
        int gwz = t.size();
        call.getDevice()
                .getKernel(call)
                .passAllOf( t )
                .passAllOf( t )
                .pass( call.input( Number.class, 1 ).at(0).get().floatValue() )
                .pass( t.rank() )
                .pass( call.getValOf( Arg.DerivIdx.class ) )
                .call( gwz );

        return call.input(0);
    }
}
