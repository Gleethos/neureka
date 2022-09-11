package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLScalarBroadcastMultiplication extends CLScalarBroadcast
{
    public CLScalarBroadcastMultiplication(String id) {
        super( id,  "output = input1 * value;\n", "if ( d == 0 ) {output = value;}else{output = input1;}\n" );
    }

    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        if ( call.getDerivativeIndex() == 0 )
            return call.input( 2 ).shallowCopy().getUnsafe().setIsIntermediate( true );
        else if ( call.getDerivativeIndex() == 1 )
            return call.input( 1 ).shallowCopy().getUnsafe().setIsIntermediate( true );
        else {
            int offset = (call.input(Number.class, 2).isVirtual() || call.input(Number.class, 2).size() == 1) ? 1 : 0;
            int gwz = call.input(Number.class, 0).size();
            call.getDevice()
                    .getKernel(call)
                    .passAllOf(call.input(Number.class, 0))
                    .passAllOf(call.input(Number.class, 0 + offset))
                    .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                    .pass(call.input(Number.class, 0).rank())
                    .pass(call.getValOf(Arg.DerivIdx.class))
                    .call(gwz);
        }
        return call.input( 0 );
    }
}
