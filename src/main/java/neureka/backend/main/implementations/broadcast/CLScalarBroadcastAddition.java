package neureka.backend.main.implementations.broadcast;

import neureka.Tensor;
import neureka.backend.api.ExecutionCall;
import neureka.devices.opencl.OpenCLDevice;
import neureka.math.args.Arg;

public class CLScalarBroadcastAddition extends CLScalarBroadcast
{
    public CLScalarBroadcastAddition(String id) {
        super( id,  "output = input1 + value;\n", "output = 1;\n" );
    }

    @Override
    public Tensor<?> run(ExecutionCall<OpenCLDevice> call) {
        assert call.arity() == 3;
        if ( call.getDerivativeIndex() == 0 )
            return Tensor.of( call.input(1).shape(), 1d ).mut().setIsIntermediate( true );
        else if ( call.getDerivativeIndex() == 1 )
            return Tensor.of( call.input( 2 ).shape(), 1d ).mut().setIsIntermediate( true );
        else {
            int gwz = call.input(Number.class, 0).size();
            float value = call.input(Number.class, 2).item(0).floatValue();
            call.getDevice()
                    .getKernel(call)
                    .passAllOf(call.input(Number.class, 0))
                    .passAllOf(call.input(Number.class, 1))
                    .pass(value)
                    .pass(call.input(Number.class, 0).rank())
                    .pass(call.getValOf(Arg.DerivIdx.class))
                    .call(gwz);
        }
        return call.input( 0 );
    }
}
