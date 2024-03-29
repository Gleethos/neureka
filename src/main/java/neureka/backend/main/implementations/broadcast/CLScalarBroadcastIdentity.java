package neureka.backend.main.implementations.broadcast;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.math.args.Arg;
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
    public Tensor<?> run(ExecutionCall<OpenCLDevice> call) {
        Tensor<Number> t = call.input( Number.class, 0 );
        int gwz = t.size();
        call.getDevice()
                .getKernel(call)
                .passAllOf( t )
                .passAllOf( t )
                .pass( call.input( Number.class, 1 ).at(0).get() )
                .pass( t.rank() )
                .pass( call.getValOf( Arg.DerivIdx.class ) )
                .call( gwz );

        return call.input(0);
    }
}
