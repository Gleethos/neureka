package neureka.backend.main.implementations.broadcast;

import neureka.Neureka;
import neureka.backend.main.implementations.ParsedCLImplementation;
import neureka.calculus.args.Arg;
import neureka.devices.opencl.KernelCode;

public class CLBroadcast extends ParsedCLImplementation
{
    protected CLBroadcast(String postfix, String forward, String backward) {
        super(
            call -> {
                int offset = ( call.input( Number.class, 0 ) != null ? 0 : 1 );
                int gwz    = ( call.input( Number.class, 0 ) != null ? call.input( Number.class, 0 ).size() : call.input( Number.class, 1 ).size() );
                call.getDevice()
                    .getKernel(call)
                    .passAllOf( call.input( Number.class, offset ) )
                    .passAllOf( call.input( Number.class, offset + 1 ) )
                    .passAllOf( call.input( Number.class, offset + 2 ) )
                    .pass( call.input( Number.class, 0 ).rank() )
                    .pass( call.getValOf( Arg.DerivIdx.class ) )
                    .call( gwz );

                return call.input( 0 );
            },
            3,
            Neureka.get().utility().readResource("kernels/broadcast_template.cl"),
            forward,
            backward,
            postfix,
            kernelCode -> new KernelCode[]{kernelCode}
        );
    }

}
