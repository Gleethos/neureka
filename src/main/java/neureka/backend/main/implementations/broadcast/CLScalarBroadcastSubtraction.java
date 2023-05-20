package neureka.backend.main.implementations.broadcast;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.math.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLScalarBroadcastSubtraction extends CLScalarBroadcast
{
    public CLScalarBroadcastSubtraction(String id) {
        super( id,  "output = input1 - value;\n", "if (d==0) { output = 1; } else { output = -1; }" );
    }

    @Override
    public Tensor<?> run(ExecutionCall<OpenCLDevice> call) {
        int offset = (call.input( Number.class, 2 ).isVirtual() || call.input( Number.class, 2 ).size() == 1)?1:0;
        int gwz = call.input( Number.class, 0 ).size();
        call.getDevice()
                .getKernel(call)
                .passAllOf(call.input( Number.class, 0 ))
                .passAllOf(call.input( Number.class, 0 ))
                .pass((float)call.input( Number.class, 1+offset).at(0).get().doubleValue())
                .pass( call.input( Number.class, 0 ).rank() )
                .pass( call.getValOf( Arg.DerivIdx.class ) )
                .call( gwz );

        return call.input(0);
    }
}
