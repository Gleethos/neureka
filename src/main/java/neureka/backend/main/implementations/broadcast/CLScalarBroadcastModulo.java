package neureka.backend.main.implementations.broadcast;

import neureka.Tsr;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.args.Arg;
import neureka.devices.opencl.OpenCLDevice;

public class CLScalarBroadcastModulo extends CLScalarBroadcast
{
    public CLScalarBroadcastModulo(String id ) {
        super(
            id,
                "output = ("+TYPE+")(((int)input1) % ((int)value));                 \n",
                "   if ( d == 0 ) {                                           \n" +
                "       output = ("+TYPE+")(1/value);                                  \n" +
                "   } else {                                                           \n" +
                "       output = ("+TYPE+")(-value /(float)pow((float)input1, 2.0f));  \n" +
                "   }"
        );
    }

    @Override
    public Tsr<?> run(ExecutionCall<OpenCLDevice> call) {
        int offset = (call.input( Number.class, 2 ).isVirtual() || call.input( Number.class, 2 ).size() == 1)?1:0;
        int gwz = call.input( Number.class, 0 ).size();
        call.getDevice()
                .getKernel(call)
                .passAllOf(call.input( Number.class, 0 ))
                .passAllOf(call.input( Number.class, 0 ))
                .pass( call.input( Number.class, 1 + offset ).at( 0 ).get().floatValue() )
                .pass( call.input( Number.class, 0 ).rank() )
                .pass( call.getValOf( Arg.DerivIdx.class ) )
                .call( gwz );

        return call.input( 0 );
    }
}
